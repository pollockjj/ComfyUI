from __future__ import annotations

import logging
import sys
import types
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import yaml
import pyisolate
from pyisolate import ExtensionManager, ExtensionManagerConfig

from .extension_wrapper import ComfyNodeExtension

logger = logging.getLogger(__name__)


class ExtensionLoadError(RuntimeError):
    pass


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
    logger: logging.Logger,
    build_stub_class: Callable[[str, Dict[str, object], ComfyNodeExtension], type],
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

    from comfy.isolation import _RUNNING_EXTENSIONS
    _RUNNING_EXTENSIONS[extension_name] = extension

    remote_nodes: Dict[str, str] = await extension.list_nodes()
    if not remote_nodes:
        return []

    specs: List[Tuple[str, str, type]] = []
    for node_name, display_name in remote_nodes.items():
        details = await extension.get_node_details(node_name)
        details["display_name"] = display_name
        stub_cls = build_stub_class(node_name, details, extension)
        specs.append((node_name, display_name, stub_cls))

    return specs


__all__ = ["ExtensionLoadError", "register_dummy_module", "load_isolated_node"]
