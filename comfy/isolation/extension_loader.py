from __future__ import annotations

import logging
import os
import sys
import types
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import yaml
import pyisolate
from pyisolate import ExtensionManager, ExtensionManagerConfig
from .vae_proxy import VAERegistry
from .clip_proxy import CLIPRegistry
from .model_patcher_proxy import ModelPatcherRegistry
from .model_sampling_proxy import ModelSamplingRegistry

from .extension_wrapper import ComfyNodeExtension
from .manifest_loader import is_cache_valid, load_from_cache, save_to_cache

logger = logging.getLogger(__name__)


def get_enforcement_policy() -> Dict[str, bool]:
    # Return enforcement policy (PYISOLATE_ENFORCE_ISOLATED, PYISOLATE_ENFORCE_SANDBOX)
    return {
        "force_isolated": os.environ.get("PYISOLATE_ENFORCE_ISOLATED") == "1",
        "force_sandbox": os.environ.get("PYISOLATE_ENFORCE_SANDBOX") == "1",
    }


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

    # Read manifest values
    isolated = manifest.get("isolated", False)
    sandbox = manifest.get("sandbox", False)

    policy = get_enforcement_policy()
    if policy["force_sandbox"]:
        sandbox = True
        isolated = True
        logger.debug("Enforcement: sandbox=True for %s", node_dir.name)
    elif policy["force_isolated"]:
        isolated = True
        logger.debug("Enforcement: isolated=True for %s", node_dir.name)

    if not isolated:
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
        "sandbox": sandbox,  # NEW: Pass sandbox config to pyisolate
        # APIs are auto-populated from adapter.provide_rpc_services() if not specified here
    }

    extension = manager.load_extension(extension_config)
    register_dummy_module(extension_name, node_dir)

    # Try cache first (lazy spawn)
    if is_cache_valid(node_dir, manifest_path, venv_root):
        cached_data = load_from_cache(node_dir, venv_root)
        if cached_data:
            logger.info(f"][ {extension_name} loaded from cache")
            specs: List[Tuple[str, str, type]] = []
            for node_name, details in cached_data.items():
                stub_cls = build_stub_class(node_name, details, extension)
                specs.append((node_name, details.get("display_name", node_name), stub_cls))
            return specs

    # Cache miss - spawn process and get metadata
    logger.info(f"][ {extension_name} cache miss, spawning process for metadata")

    remote_nodes: Dict[str, str] = await extension.list_nodes()
    if not remote_nodes:
        return []

    specs: List[Tuple[str, str, type]] = []
    cache_data: Dict[str, Dict] = {}

    for node_name, display_name in remote_nodes.items():
        details = await extension.get_node_details(node_name)
        details["display_name"] = display_name
        cache_data[node_name] = details
        stub_cls = build_stub_class(node_name, details, extension)
        specs.append((node_name, display_name, stub_cls))

    # Save metadata to cache for future runs
    save_to_cache(node_dir, venv_root, cache_data, manifest_path)
    logger.info(f"][ {extension_name} metadata cached")

    # EJECT: Kill process after getting metadata (will respawn on first execution)
    logger.info(f"][ {extension_name} ejecting after metadata extraction")
    extension.stop()

    return specs


__all__ = ["ExtensionLoadError", "register_dummy_module", "load_isolated_node"]
