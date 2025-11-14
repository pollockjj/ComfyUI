"""PyIsolate isolation system for ComfyUI custom nodes.

Provides process isolation for custom_nodes via PyIsolate, enabling:
- Dependency conflict resolution (isolated venvs)
- Security sandboxing
- Zero-copy tensor sharing (share_torch=True)
- ProxiedSingleton for shared ComfyUI services
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import yaml

try:
    import pyisolate
    from pyisolate import ExtensionManager, ExtensionManagerConfig
except ImportError:  # pragma: no cover - pyisolate always available in target env
    pyisolate = None
    ExtensionManager = None  # type: ignore
    ExtensionManagerConfig = None  # type: ignore

from .extension_wrapper import ComfyNodeExtension

LOG_PREFIX = "� [PyIsolate]"
PYISOLATE_EDITABLE_PATH = Path("/home/johnj/pyisolate")


def get_isolation_logger(name: str) -> logging.Logger:
    """Get logger with PyIsolate prefix for consistent log formatting."""

    return logging.getLogger(name)


logger = get_isolation_logger(__name__)
logger.info(f"{LOG_PREFIX}[System] Isolation system initialized")


def initialize_proxies() -> None:
    """Placeholder until ProxiedSingletons are required for active nodes."""

    logger.info(f"{LOG_PREFIX}[System] ProxiedSingleton initialization skipped (bootstrap mode)")


@dataclass(frozen=True)
class IsolatedNodeSpec:
    """Description of a node exposed through a PyIsolate extension."""

    node_name: str
    display_name: str
    stub_class: type
    module_path: Path


_ISOLATED_NODE_SPECS: List[IsolatedNodeSpec] = []
_ISOLATION_SCAN_ATTEMPTED = False
_EXTENSION_MANAGERS: List[ExtensionManager] = []  # Keep alive so subprocesses persist


async def initialize_isolation_nodes() -> List[IsolatedNodeSpec]:
    """Discover isolated custom nodes via pyisolate.yaml manifests."""

    global _ISOLATED_NODE_SPECS, _ISOLATION_SCAN_ATTEMPTED

    if _ISOLATED_NODE_SPECS:
        logger.info(f"{LOG_PREFIX}[Loader] Returning cached isolated node specs (%d entries)", len(_ISOLATED_NODE_SPECS))
        return _ISOLATED_NODE_SPECS

    if _ISOLATION_SCAN_ATTEMPTED:
        logger.warning(f"{LOG_PREFIX}[Loader] initialize_isolation_nodes already attempted; skipping re-scan")
        return []

    _ISOLATION_SCAN_ATTEMPTED = True

    if pyisolate is None:
        logger.debug(f"{LOG_PREFIX}[Loader] pyisolate unavailable, skipping isolated nodes")
        return []

    manifest_entries = _find_manifest_directories()
    if not manifest_entries:
        logger.info(f"{LOG_PREFIX}[Loader] No pyisolate manifests detected under custom_nodes")
        return []

    specs: List[IsolatedNodeSpec] = []
    for node_dir, manifest in manifest_entries:
        try:
            spec_list = await _load_isolated_node(node_dir, manifest)
        except Exception as exc:  # pragma: no cover - defensive logging only
            logger.error(
                "%s[Loader] Failed to initialize isolated node at %s: %s",
                LOG_PREFIX,
                node_dir,
                exc,
            )
            raise
        specs.extend(spec_list)

    _ISOLATED_NODE_SPECS = specs
    return list(_ISOLATED_NODE_SPECS)


def _find_manifest_directories() -> List[tuple[Path, Path]]:
    import folder_paths

    manifest_dirs: List[tuple[Path, Path]] = []
    for base_path in folder_paths.get_folder_paths("custom_nodes"):
        base = Path(base_path)
        if not base.exists() or not base.is_dir():
            continue
        for entry in base.iterdir():
            if not entry.is_dir():
                continue
            manifest = entry / "pyisolate.yaml"
            if not manifest.exists():
                manifest = entry / "pyisolate.yml"
            if manifest.exists():
                logger.info("%s[Loader] Found pyisolate manifest: %s", LOG_PREFIX, manifest)
                manifest_dirs.append((entry, manifest))
    return manifest_dirs


async def _load_isolated_node(node_dir: Path, manifest_path: Path) -> List[IsolatedNodeSpec]:
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle) or {}

    if not manifest.get("isolated", False):
        logger.debug(
            "%s[Loader] %s manifest present but isolation disabled",
            LOG_PREFIX,
            node_dir,
        )
        return []

    dependencies = list(manifest.get("dependencies", []) or [])
    share_torch = manifest.get("share_torch", True)
    extension_name = manifest.get("name", node_dir.name)
    dependencies = _augment_dependencies_with_pyisolate(dependencies, extension_name)

    venv_root = node_dir / ".venv"
    venv_root.mkdir(parents=True, exist_ok=True)

    manager_config = ExtensionManagerConfig(venv_root_path=str(venv_root))
    logger.info(
        "%s[Loader] Using manager config: venv_root=%s",
        LOG_PREFIX,
        manager_config["venv_root_path"],
    )
    manager: ExtensionManager = pyisolate.ExtensionManager(ComfyNodeExtension, manager_config)
    _EXTENSION_MANAGERS.append(manager)

    extension_config = {
        "name": extension_name,
        "module_path": str(node_dir),
        "isolated": True,
        "dependencies": dependencies,
        "apis": [],
        "share_torch": share_torch,
    }

    logger.info(
        "%s[Loader] Invoking ExtensionManager.load_extension with config=%s",
        LOG_PREFIX,
        extension_config,
    )

    try:
        extension = manager.load_extension(extension_config)
    except Exception as exc:
        logger.error(
            "%s[Loader] ExtensionManager.load_extension failed for %s: %s",
            LOG_PREFIX,
            extension_name,
            exc,
        )
        raise
    logger.info(
        "%s[Loader] Loading isolated node %s from %s",
        LOG_PREFIX,
        extension_name,
        node_dir,
    )

    specs: List[IsolatedNodeSpec] = []
    remote_nodes: Dict[str, str] = await extension.list_nodes()
    if not remote_nodes:
        raise RuntimeError(
            f"{LOG_PREFIX}[Loader] Isolated node {extension_name} at {node_dir} reported zero NODE_CLASS_MAPPINGS"
        )
    logger.info(
        "%s[Loader] %s reported %d node(s): %s",
        LOG_PREFIX,
        extension_name,
        len(remote_nodes),
        list(remote_nodes.keys()),
    )
    for node_name, display_name in remote_nodes.items():
        info = await extension.get_node_info(node_name)
        stub_cls = _build_stub_class(node_name, info, extension)
        specs.append(
            IsolatedNodeSpec(
                node_name=node_name,
                display_name=display_name,
                stub_class=stub_cls,
                module_path=node_dir,
            )
        )
    return specs


def _augment_dependencies_with_pyisolate(dependencies: List[str], extension_name: str) -> List[str]:
    deps = list(dependencies)
    if PYISOLATE_EDITABLE_PATH.exists():
        needs_injection = True
        for idx in range(len(deps) - 1):
            if deps[idx] == "-e" and deps[idx + 1] == str(PYISOLATE_EDITABLE_PATH):
                needs_injection = False
                break
        if needs_injection:
            deps = ["-e", str(PYISOLATE_EDITABLE_PATH)] + deps
            logger.info(
                "%s[Loader] Injected pyisolate editable dependency for %s",
                LOG_PREFIX,
                extension_name,
            )
    else:
        logger.error(
            "%s[Loader] Missing pyisolate source at %s; isolated node %s cannot mirror host runtime",
            LOG_PREFIX,
            PYISOLATE_EDITABLE_PATH,
            extension_name,
        )
    return deps


def _build_stub_class(node_name: str, info: Dict[str, object], extension: ComfyNodeExtension) -> type:
    function_name = "_pyisolate_execute"

    async def _execute(self, **inputs):
        result = await extension.execute_node(node_name, **inputs)
        return result

    def _input_types(cls):
        return info.get("input_types", {})

    attributes: Dict[str, object] = {
        "FUNCTION": function_name,
        "CATEGORY": info.get("category", ""),
        "OUTPUT_NODE": info.get("output_node", False),
        "RETURN_TYPES": tuple(info.get("return_types", ()) or ()),
        "RETURN_NAMES": info.get("return_names"),
        function_name: _execute,
        "_pyisolate_extension": extension,
        "_pyisolate_node_name": node_name,
        "INPUT_TYPES": classmethod(_input_types),
    }

    display_name = info.get("display_name") or node_name
    class_name = f"PyIsolate_{node_name}".replace(" ", "_")
    stub_cls = type(class_name, (), attributes)
    stub_cls.__doc__ = f"PyIsolate proxy node for {display_name}"
    logger.info(
        "%s[Loader] Built stub class %s for node %s (display=%s)",
        LOG_PREFIX,
        class_name,
        node_name,
        display_name,
    )
    return stub_cls


__all__ = [
    "LOG_PREFIX",
    "get_isolation_logger",
    "initialize_proxies",
    "initialize_isolation_nodes",
    "IsolatedNodeSpec",
]
