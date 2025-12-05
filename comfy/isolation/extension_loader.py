from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import yaml

import pyisolate
from pyisolate import ExtensionManager, ExtensionManagerConfig

from .extension_wrapper import ComfyNodeExtension
from .manifest_loader import is_cache_valid, load_from_cache, save_to_cache


class ExtensionLoadError(RuntimeError):
    pass


def augment_dependencies_with_pyisolate(
    dependencies: List[str], editable_path: Path, extension_name: str, logger
) -> List[str]:
    deps = list(dependencies)
    if editable_path.exists():
        needs_injection = True
        for idx in range(len(deps) - 1):
            if deps[idx] == "-e" and deps[idx + 1] == str(editable_path):
                needs_injection = False
                break
        if needs_injection:
            deps = ["-e", str(editable_path)] + deps
    else:
        logger.error(
            "[I][Loader] Missing pyisolate source at %s; isolated node %s cannot mirror host runtime",
            editable_path,
            extension_name,
        )
        raise ExtensionLoadError(
            f"PyIsolate source missing at {editable_path}; cannot load isolated node {extension_name}"
        )
    return deps


def register_dummy_module(extension_name: str, node_dir: Path) -> None:
    normalized_name = extension_name.replace("-", "_").replace(".", "_")
    if normalized_name not in sys.modules:
        dummy_module = types.ModuleType(normalized_name)
        dummy_module.__file__ = str(node_dir / "__init__.py")
        dummy_module.__path__ = [str(node_dir)]
        dummy_module.__package__ = normalized_name
        sys.modules[normalized_name] = dummy_module


async def load_isolated_node(
    node_dir: Path,
    manifest_path: Path,
    logger,
    build_stub_class: Callable[[str, Dict[str, object], ComfyNodeExtension], type],
    pyisolate_editable_path: Path,
    venv_root: Path,
    extension_managers: List[ExtensionManager],
) -> List[Tuple[str, str, type]]:
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle) or {}

    if not manifest.get("isolated", False):
        return []

    dependencies = list(manifest.get("dependencies", []) or [])
    share_torch = manifest.get("share_torch", True)
    extension_name = manifest.get("name", node_dir.name)
    dependencies = augment_dependencies_with_pyisolate(
        dependencies, pyisolate_editable_path, extension_name, logger
    )

    specs: List[Tuple[str, str, type]] = []

    # Check cache FIRST - if valid, build stubs without spawning process
    if is_cache_valid(node_dir, manifest_path):
        cached_data = load_from_cache(node_dir)
        if cached_data:
            # Lazy setup: create Extension object but don't start process yet
            manager_config = ExtensionManagerConfig(venv_root_path=str(venv_root))
            manager: ExtensionManager = pyisolate.ExtensionManager(ComfyNodeExtension, manager_config)
            extension_managers.append(manager)

            extension_config = {
                "name": extension_name,
                "module_path": str(node_dir),
                "isolated": True,
                "dependencies": dependencies,
                "share_torch": share_torch,
                "apis": [],
            }

            extension = manager.load_extension(extension_config)
            register_dummy_module(extension_name, node_dir)

            for node_name, details in cached_data.items():
                stub_cls = build_stub_class(node_name, details, extension)
                specs.append((node_name, details.get("display_name", node_name), stub_cls))
            return specs

    # Cache miss - need to spawn process and interrogate
    manager_config = ExtensionManagerConfig(venv_root_path=str(venv_root))
    manager: ExtensionManager = pyisolate.ExtensionManager(ComfyNodeExtension, manager_config)
    extension_managers.append(manager)

    extension_config = {
        "name": extension_name,
        "module_path": str(node_dir),
        "isolated": True,
        "dependencies": dependencies,
        "share_torch": share_torch,
        "apis": [],
    }

    extension = manager.load_extension(extension_config)
    register_dummy_module(extension_name, node_dir)

    remote_nodes: Dict[str, str] = await extension.list_nodes()
    if not remote_nodes:
        raise ExtensionLoadError(
            f"[I][Loader] Isolated node {extension_name} at {node_dir} reported zero NODE_CLASS_MAPPINGS"
        )

    cache_data_to_save = {}

    for node_name, display_name in remote_nodes.items():
        details = await extension.get_node_details(node_name)
        details["display_name"] = display_name
        cache_data_to_save[node_name] = details

        stub_cls = build_stub_class(node_name, details, extension)
        specs.append((node_name, display_name, stub_cls))

    save_to_cache(node_dir, cache_data_to_save)
    return specs


__all__ = [
    "ExtensionLoadError",
    "augment_dependencies_with_pyisolate",
    "register_dummy_module",
    "load_isolated_node",
]
