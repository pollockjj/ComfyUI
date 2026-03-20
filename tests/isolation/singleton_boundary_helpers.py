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
FORBIDDEN_EXACT_SMALL_PROXY_MODULES = FORBIDDEN_SEALED_SINGLETON_MODULES


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


def _load_helper_proxy_service() -> Any | None:
    try:
        from comfy.isolation.proxies.helper_proxies import HelperProxiesService
    except (ImportError, AttributeError):
        return None
    return HelperProxiesService


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
                "rpc_get_models_dir": lambda: "/sandbox/models",
                "rpc_get_folder_names_and_paths": lambda: {
                    "checkpoints": {
                        "paths": ["/sandbox/models/checkpoints"],
                        "extensions": [".ckpt", ".safetensors"],
                    }
                },
                "rpc_get_extension_mimetypes_cache": lambda: {"webp": "image"},
                "rpc_get_filename_list_cache": lambda: {},
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
            "HelperProxiesService": {
                "rpc_restore_input_types": lambda raw: raw,
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
    helper_proxy_service = _load_helper_proxy_service()
    if helper_proxy_service is not None:
        helper_proxy_service.clear_rpc()


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
    helper_proxy_service = _load_helper_proxy_service()
    if helper_proxy_service is not None:
        helper_proxy_service.set_rpc(fake_rpc)


def reset_forbidden_singleton_modules() -> None:
    for module_name in (
        "folder_paths",
        "comfy.utils",
        "comfy_execution.progress",
        "torch",
    ):
        sys.modules.pop(module_name, None)


class FakeExactRelayCaller:
    def __init__(self, methods: dict[str, Any], transcripts: list[dict[str, Any]], object_id: str):
        self._methods = methods
        self._transcripts = transcripts
        self._object_id = object_id

    def __getattr__(self, name: str):
        if name not in self._methods:
            raise AttributeError(name)

        async def method(*args: Any, **kwargs: Any) -> Any:
            self._transcripts.append(
                {
                    "phase": "child_call",
                    "object_id": self._object_id,
                    "method": name,
                    "args": list(args),
                    "kwargs": dict(kwargs),
                }
            )
            impl = self._methods[name]
            self._transcripts.append(
                {
                    "phase": "host_invocation",
                    "object_id": self._object_id,
                    "method": name,
                    "target": impl["target"],
                    "args": list(args),
                    "kwargs": dict(kwargs),
                }
            )
            result = impl["result"](*args, **kwargs) if callable(impl["result"]) else impl["result"]
            self._transcripts.append(
                {
                    "phase": "result",
                    "object_id": self._object_id,
                    "method": name,
                    "result": result,
                }
            )
            return result

        return method


class FakeExactRelayRPC:
    def __init__(self) -> None:
        self.transcripts: list[dict[str, Any]] = []
        self._services: dict[str, dict[str, Any]] = {
            "FolderPathsProxy": {
                "rpc_get_models_dir": {
                    "target": "folder_paths.models_dir",
                    "result": "/sandbox/models",
                },
                "rpc_get_temp_directory": {
                    "target": "folder_paths.get_temp_directory",
                    "result": "/sandbox/temp",
                },
                "rpc_get_input_directory": {
                    "target": "folder_paths.get_input_directory",
                    "result": "/sandbox/input",
                },
                "rpc_get_output_directory": {
                    "target": "folder_paths.get_output_directory",
                    "result": "/sandbox/output",
                },
                "rpc_get_user_directory": {
                    "target": "folder_paths.get_user_directory",
                    "result": "/sandbox/user",
                },
                "rpc_get_folder_names_and_paths": {
                    "target": "folder_paths.folder_names_and_paths",
                    "result": {
                        "checkpoints": {
                            "paths": ["/sandbox/models/checkpoints"],
                            "extensions": [".ckpt", ".safetensors"],
                        }
                    },
                },
                "rpc_get_extension_mimetypes_cache": {
                    "target": "folder_paths.extension_mimetypes_cache",
                    "result": {"webp": "image"},
                },
                "rpc_get_filename_list_cache": {
                    "target": "folder_paths.filename_list_cache",
                    "result": {},
                },
                "rpc_get_annotated_filepath": {
                    "target": "folder_paths.get_annotated_filepath",
                    "result": lambda name, default_dir=None: FakeSingletonRPC._get_annotated_filepath(name, default_dir),
                },
                "rpc_exists_annotated_filepath": {
                    "target": "folder_paths.exists_annotated_filepath",
                    "result": False,
                },
                "rpc_add_model_folder_path": {
                    "target": "folder_paths.add_model_folder_path",
                    "result": None,
                },
                "rpc_get_folder_paths": {
                    "target": "folder_paths.get_folder_paths",
                    "result": lambda folder_name: [f"/sandbox/models/{folder_name}"],
                },
                "rpc_get_filename_list": {
                    "target": "folder_paths.get_filename_list",
                    "result": lambda folder_name: [f"{folder_name}_fixture.safetensors"],
                },
                "rpc_get_full_path": {
                    "target": "folder_paths.get_full_path",
                    "result": lambda folder_name, filename: f"/sandbox/models/{folder_name}/{filename}",
                },
            },
            "UtilsProxy": {
                "progress_bar_hook": {
                    "target": "comfy.utils.PROGRESS_BAR_HOOK",
                    "result": lambda value, total, preview=None, node_id=None: {
                        "value": value,
                        "total": total,
                        "preview": preview,
                        "node_id": node_id,
                    },
                },
            },
            "ProgressProxy": {
                "rpc_set_progress": {
                    "target": "comfy_execution.progress.get_progress_state().update_progress",
                    "result": None,
                },
            },
            "HelperProxiesService": {
                "rpc_restore_input_types": {
                    "target": "comfy.isolation.proxies.helper_proxies.restore_input_types",
                    "result": lambda raw: raw,
                }
            },
        }

    def create_caller(self, cls: Any, object_id: str):
        methods = self._services.get(object_id) or self._services.get(getattr(cls, "__name__", object_id))
        if methods is None:
            raise KeyError(object_id)
        return FakeExactRelayCaller(methods, self.transcripts, object_id)


def capture_exact_small_proxy_relay() -> dict[str, object]:
    reset_forbidden_singleton_modules()
    fake_rpc = FakeExactRelayRPC()
    prepare_sealed_singleton_proxies(fake_rpc)

    from comfy.isolation.proxies.folder_paths_proxy import FolderPathsProxy
    from comfy.isolation.proxies.helper_proxies import restore_input_types
    from comfy.isolation.proxies.progress_proxy import ProgressProxy
    from comfy.isolation.proxies.utils_proxy import UtilsProxy

    folder_proxy = FolderPathsProxy()
    utils_proxy = UtilsProxy()
    progress_proxy = ProgressProxy()
    before = set(sys.modules)

    restored = restore_input_types(
        {
            "required": {
                "image": {"__pyisolate_any_type__": True, "value": "*"},
            }
        }
    )
    folder_path = folder_proxy.get_annotated_filepath("demo.png[input]")
    models_dir = folder_proxy.models_dir
    folder_names_and_paths = folder_proxy.folder_names_and_paths
    asyncio.run(utils_proxy.progress_bar_hook(2, 5, node_id="node-17"))
    progress_proxy.set_progress(1.5, 5.0, node_id="node-17")

    imported = set(sys.modules) - before
    return {
        "mode": "exact_small_proxy_relay",
        "folder_path": folder_path,
        "models_dir": models_dir,
        "folder_names_and_paths": folder_names_and_paths,
        "restored_any_type": str(restored["required"]["image"]),
        "transcripts": fake_rpc.transcripts,
        "modules": sorted(imported),
        "forbidden_matches": matching_modules(FORBIDDEN_EXACT_SMALL_PROXY_MODULES, imported),
    }


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
