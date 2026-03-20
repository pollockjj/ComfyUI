from __future__ import annotations

import asyncio
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any


COMFYUI_ROOT = Path(__file__).resolve().parents[2]
UV_SEALED_WORKER_MODULE = COMFYUI_ROOT / "tests" / "isolation" / "uv_sealed_worker" / "__init__.py"
FORBIDDEN_MINIMAL_SEALED_MODULES = (
    "torch",
    "folder_paths",
    "comfy.utils",
    "comfy.model_management",
    "main",
    "comfy.isolation.extension_wrapper",
)
FORBIDDEN_SEALED_SINGLETON_MODULES = (
    "torch",
    "folder_paths",
    "comfy.utils",
    "comfy_execution.progress",
)


def _load_module_from_path(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to build import spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def matching_modules(prefixes: tuple[str, ...], modules: set[str]) -> list[str]:
    return sorted(
        module_name
        for module_name in modules
        if any(
            module_name == prefix or module_name.startswith(f"{prefix}.")
            for prefix in prefixes
        )
    )


async def _capture_minimal_sealed_worker_imports() -> dict[str, object]:
    from pyisolate.sealed import SealedNodeExtension

    module_name = "tests.isolation.uv_sealed_worker_boundary_probe"
    before = set(sys.modules)
    extension = SealedNodeExtension()
    module = _load_module_from_path(module_name, UV_SEALED_WORKER_MODULE)
    try:
        await extension.on_module_loaded(module)
        node_list = await extension.list_nodes()
        node_details = await extension.get_node_details("UVSealedRuntimeProbe")
        imported = set(sys.modules) - before
        return {
            "mode": "minimal_sealed_worker",
            "node_names": sorted(node_list),
            "runtime_probe_function": node_details["function"],
            "modules": sorted(imported),
            "forbidden_matches": matching_modules(FORBIDDEN_MINIMAL_SEALED_MODULES, imported),
        }
    finally:
        sys.modules.pop(module_name, None)


def capture_minimal_sealed_worker_imports() -> dict[str, object]:
    return asyncio.run(_capture_minimal_sealed_worker_imports())


class FakeSingletonCaller:
    def __init__(self, methods: dict[str, Any], calls: list[dict[str, Any]], object_id: str):
        self._methods = methods
        self._calls = calls
        self._object_id = object_id

    def __getattr__(self, name: str):
        if name not in self._methods:
            raise AttributeError(name)

        async def method(*args: Any, **kwargs: Any) -> Any:
            self._calls.append(
                {
                    "object_id": self._object_id,
                    "method": name,
                    "args": list(args),
                    "kwargs": dict(kwargs),
                }
            )
            result = self._methods[name]
            return result(*args, **kwargs) if callable(result) else result

        return method


class FakeSingletonRPC:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self._services: dict[str, dict[str, Any]] = {
            "FolderPathsProxy": {
                "rpc_get_temp_directory": lambda: "/sandbox/temp",
                "rpc_get_input_directory": lambda: "/sandbox/input",
                "rpc_get_output_directory": lambda: "/sandbox/output",
                "rpc_get_user_directory": lambda: "/sandbox/user",
                "rpc_get_annotated_filepath": self._get_annotated_filepath,
                "rpc_exists_annotated_filepath": lambda _name: False,
                "rpc_add_model_folder_path": lambda *_args, **_kwargs: None,
                "rpc_get_folder_paths": lambda folder_name: [f"/sandbox/models/{folder_name}"],
                "rpc_get_filename_list": lambda folder_name: [f"{folder_name}_fixture.safetensors"],
                "rpc_get_full_path": lambda folder_name, filename: f"/sandbox/models/{folder_name}/{filename}",
                "rpc_snapshot": lambda: {
                    "models_dir": "/sandbox/models",
                    "input_directory": "/sandbox/input",
                    "output_directory": "/sandbox/output",
                    "temp_directory": "/sandbox/temp",
                    "user_directory": "/sandbox/user",
                    "supported_pt_extensions": [".ckpt", ".safetensors"],
                    "folder_names_and_paths": {
                        "checkpoints": {
                            "paths": ["/sandbox/models/checkpoints"],
                            "extensions": [".ckpt", ".safetensors"],
                        }
                    },
                    "extension_mimetypes_cache": {"webp": "image"},
                    "filename_list_cache": {},
                },
            },
            "UtilsProxy": {
                "progress_bar_hook": lambda value, total, preview=None, node_id=None: {
                    "value": value,
                    "total": total,
                    "preview": preview,
                    "node_id": node_id,
                }
            },
            "ProgressProxy": {
                "rpc_set_progress": lambda value, max_value, node_id=None, image=None: {
                    "value": value,
                    "max_value": max_value,
                    "node_id": node_id,
                    "image": image,
                }
            },
        }

    @staticmethod
    def _get_annotated_filepath(name: str, default_dir: str | None = None) -> str:
        if name.endswith("[output]"):
            return f"/sandbox/output/{name[:-8]}"
        if name.endswith("[input]"):
            return f"/sandbox/input/{name[:-7]}"
        if name.endswith("[temp]"):
            return f"/sandbox/temp/{name[:-6]}"
        base_dir = default_dir or "/sandbox/input"
        return f"{base_dir}/{name}"

    def create_caller(self, cls: Any, object_id: str):
        methods = self._services.get(object_id) or self._services.get(getattr(cls, "__name__", object_id))
        if methods is None:
            raise KeyError(object_id)
        return FakeSingletonCaller(methods, self.calls, object_id)


def _clear_proxy_rpcs() -> None:
    from comfy.isolation.proxies.folder_paths_proxy import FolderPathsProxy
    from comfy.isolation.proxies.progress_proxy import ProgressProxy
    from comfy.isolation.proxies.utils_proxy import UtilsProxy

    FolderPathsProxy.clear_rpc()
    ProgressProxy.clear_rpc()
    UtilsProxy.clear_rpc()


def prepare_sealed_singleton_proxies(fake_rpc: FakeSingletonRPC) -> None:
    os.environ["PYISOLATE_CHILD"] = "1"
    os.environ["PYISOLATE_IMPORT_TORCH"] = "0"
    _clear_proxy_rpcs()

    from comfy.isolation.proxies.folder_paths_proxy import FolderPathsProxy
    from comfy.isolation.proxies.progress_proxy import ProgressProxy
    from comfy.isolation.proxies.utils_proxy import UtilsProxy

    FolderPathsProxy.set_rpc(fake_rpc)
    ProgressProxy.set_rpc(fake_rpc)
    UtilsProxy.set_rpc(fake_rpc)


def reset_forbidden_singleton_modules() -> None:
    for module_name in (
        "folder_paths",
        "comfy.utils",
        "comfy_execution.progress",
        "torch",
    ):
        sys.modules.pop(module_name, None)


def capture_sealed_singleton_imports() -> dict[str, object]:
    reset_forbidden_singleton_modules()
    fake_rpc = FakeSingletonRPC()
    before = set(sys.modules)
    prepare_sealed_singleton_proxies(fake_rpc)

    from comfy.isolation.proxies.folder_paths_proxy import FolderPathsProxy
    from comfy.isolation.proxies.progress_proxy import ProgressProxy
    from comfy.isolation.proxies.utils_proxy import UtilsProxy

    folder_proxy = FolderPathsProxy()
    progress_proxy = ProgressProxy()
    utils_proxy = UtilsProxy()

    folder_path = folder_proxy.get_annotated_filepath("demo.png[input]")
    temp_dir = folder_proxy.get_temp_directory()
    models_dir = folder_proxy.models_dir
    asyncio.run(utils_proxy.progress_bar_hook(2, 5, node_id="node-17"))
    progress_proxy.set_progress(1.5, 5.0, node_id="node-17")

    imported = set(sys.modules) - before
    return {
        "mode": "sealed_singletons",
        "folder_path": folder_path,
        "temp_dir": temp_dir,
        "models_dir": models_dir,
        "rpc_calls": fake_rpc.calls,
        "modules": sorted(imported),
        "forbidden_matches": matching_modules(FORBIDDEN_SEALED_SINGLETON_MODULES, imported),
    }
