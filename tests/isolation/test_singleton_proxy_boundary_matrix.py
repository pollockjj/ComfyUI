from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from pyisolate.sealed import SealedNodeExtension


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


def _load_module_from_path(module_name: str, module_path: Path) -> ModuleType:
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


def _matching_modules(prefixes: tuple[str, ...], modules: set[str]) -> list[str]:
    return sorted(
        module_name
        for module_name in modules
        if any(
            module_name == prefix or module_name.startswith(f"{prefix}.")
            for prefix in prefixes
        )
    )


async def _capture_minimal_sealed_worker_imports() -> dict[str, object]:
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
            "forbidden_matches": _matching_modules(FORBIDDEN_MINIMAL_SEALED_MODULES, imported),
        }
    finally:
        sys.modules.pop(module_name, None)


def capture_minimal_sealed_worker_imports() -> dict[str, object]:
    return asyncio.run(_capture_minimal_sealed_worker_imports())


def test_minimal_sealed_worker_forbidden_imports() -> None:
    payload = capture_minimal_sealed_worker_imports()

    assert payload["mode"] == "minimal_sealed_worker"
    assert payload["runtime_probe_function"] == "inspect"
    assert payload["forbidden_matches"] == []


def test_torch_share_subset_scope() -> None:
    minimal = capture_minimal_sealed_worker_imports()

    allowed_torch_share_only = {
        "torch",
        "folder_paths",
        "comfy.utils",
        "comfy.model_management",
        "main",
        "comfy.isolation.extension_wrapper",
    }

    assert minimal["forbidden_matches"] == []
    assert all(
        module_name not in minimal["modules"] for module_name in sorted(allowed_torch_share_only)
    )


def test_capture_payload_is_json_serializable() -> None:
    payload = capture_minimal_sealed_worker_imports()

    encoded = json.dumps(payload, sort_keys=True)

    assert "\"minimal_sealed_worker\"" in encoded
