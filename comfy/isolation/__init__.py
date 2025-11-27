"""PyIsolate isolation system for ComfyUI custom nodes.

Provides process isolation for custom_nodes via PyIsolate, enabling:
- Dependency conflict resolution (isolated venvs)
- Security sandboxing
- Zero-copy tensor sharing (share_torch=True)
- ProxiedSingleton for shared ComfyUI services
"""

from __future__ import annotations

import logging
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import yaml

import folder_paths

try:
    import pyisolate
    from pyisolate import ExtensionManager, ExtensionManagerConfig
except ImportError:  # pragma: no cover - pyisolate always available in target env
    pyisolate = None
    ExtensionManager = None  # type: ignore
    ExtensionManagerConfig = None  # type: ignore


def initialize_proxies():
    """
    Initialize PyIsolate ProxiedSingleton instances for ComfyUI services.
    
    This makes core ComfyUI services available to isolated nodes via RPC,
    including the new ModelSamplingRegistry for pickle-safe model sampling.
    """
    # Import here to avoid circular dependencies and ensure torch is loaded
    from server import PromptServer
    import comfy.model_management as model_management
    from comfy.utils import ProgressBar
    from comfy.isolation.model_sampling_proxy import ModelSamplingRegistry
    
    # Register all ProxiedSingleton classes
    # These will be accessible in isolated processes via RPC
    apis = [
        PromptServer,
        folder_paths,
        model_management,
        ProgressBar,
        ModelSamplingRegistry,
    ]
    
    # No explicit registration needed - ProxiedSingleton auto-registers
    # Just instantiating them makes them available
    logger.info(
        "📚 [PyIsolate][System] Isolation system available: %d classes of ProxiedSingletons registered",
        len(apis)
    )

from .extension_wrapper import ComfyNodeExtension

LOG_PREFIX = "📚 [PyIsolate]"

# Module-level list to store timing data for ComfyManager-style report
isolated_node_timings: List[tuple[float, Path]] = []


class _AnyTypeProxy(str):
    """Replacement for custom AnyType objects used by some nodes."""

    def __new__(cls, value: str = "*"):
        return super().__new__(cls, value)

    def __ne__(self, other):  # type: ignore[override]
        return False


class _FlexibleOptionalInputProxy(dict):
    """Replacement for FlexibleOptionalInputType to allow dynamic inputs."""

    def __init__(self, flex_type, data: Optional[Dict[str, object]] = None):
        super().__init__()
        self.type = flex_type
        self.data = data or {}
        for key, value in self.data.items():
            self[key] = value

    def __getitem__(self, key):  # type: ignore[override]
        if key in self.data:
            return self.data[key]
        value = self.type
        if isinstance(value, tuple):
            return value
        return (value,)

    def __contains__(self, key):  # type: ignore[override]
        return True


class _ByPassTypeTupleProxy(tuple):
    """Replacement for ByPassTypeTuple to mirror wildcard fallback behavior."""

    def __new__(cls, values):
        return super().__new__(cls, values)

    def __getitem__(self, index):  # type: ignore[override]
        if index >= len(self):
            return _AnyTypeProxy("*")
        return super().__getitem__(index)


def _restore_special_value(value):
    if isinstance(value, dict):
        if value.get("__pyisolate_any_type__"):
            return _AnyTypeProxy(value.get("value", "*"))
        if value.get("__pyisolate_flexible_optional__"):
            flex_type = _restore_special_value(value.get("type"))
            data_raw = value.get("data")
            data = (
                {k: _restore_special_value(v) for k, v in data_raw.items()}
                if isinstance(data_raw, dict)
                else {}
            )
            return _FlexibleOptionalInputProxy(flex_type, data)
        if value.get("__pyisolate_tuple__") is not None:
            return tuple(_restore_special_value(v) for v in value["__pyisolate_tuple__"])
        if value.get("__pyisolate_bypass_tuple__") is not None:
            return _ByPassTypeTupleProxy(
                tuple(_restore_special_value(v) for v in value["__pyisolate_bypass_tuple__"])
            )
        return {k: _restore_special_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_restore_special_value(v) for v in value]
    return value


def _restore_input_types(raw: Dict[str, object]) -> Dict[str, object]:
    if not isinstance(raw, dict):
        return raw
    restored: Dict[str, object] = {}
    for section, entries in raw.items():
        if isinstance(entries, dict):
            restored[section] = {k: _restore_special_value(v) for k, v in entries.items()}
        else:
            restored[section] = _restore_special_value(entries)
    return restored

BLACKLISTED_NODES: dict[str, str] = {
    # https://github.com/pollockjj/ComfyUI-MultiGPU relies on direct CUDA context reuse + custom scheduler
    "ComfyUI-MultiGPU": (
        "This extension requires direct GPU process management and hooks into ComfyUI's main event loop. "
        "Running it inside PyIsolate would break GPU synchronization and create undefined behavior."
    ),
    # Crystools pulls in jetson-stats which demands a privileged host install and cannot be vendored
    "ComfyUI-Crystools": (
        "Depends on jetson-stats, which refuses to install inside sandboxed environments. "
        "Please remove the jetson-deps dependency or provide an alternate implementation before enabling isolation."
    ),
}


def _get_user_pyisolate_path() -> Path:
    target = Path(folder_paths.base_path) / "user" / "pyisolate"
    if not target.exists():
        raise RuntimeError(
            f"PyIsolate source missing at {target}; clone or move pyisolate into ComfyUI/user/pyisolate"
        )

    logging.getLogger(__name__).debug("%s[System] Using pyisolate source at %s", LOG_PREFIX, target)
    return target


PYISOLATE_EDITABLE_PATH = _get_user_pyisolate_path()
PYISOLATE_VENV_ROOT = Path(folder_paths.base_path) / ".pyisolate_venvs"
PYISOLATE_VENV_ROOT.mkdir(parents=True, exist_ok=True)


def get_isolation_logger(name: str) -> logging.Logger:
    """Get logger with PyIsolate prefix for consistent log formatting."""

    return logging.getLogger(name)


logger = get_isolation_logger(__name__)


def initialize_proxies() -> None:
    """Initialize ProxiedSingletons for isolated nodes.
    
    Registers all proxy classes so they're available to isolated nodes via RPC.
    Actual RPC binding happens when extensions load.
    """
    import os
    
    from .proxies.folder_paths_proxy import FolderPathsProxy
    from .proxies.model_management_proxy import ModelManagementProxy
    from .proxies.nodes_proxy import NodesProxy
    from .proxies.utils_proxy import UtilsProxy
    from .proxies.prompt_server_proxy import PromptServerProxy
    
    # Instantiate singletons to register them (host side only)
    is_child = os.environ.get("PYISOLATE_CHILD") == "1"
    
    if not is_child:
        FolderPathsProxy()
        ModelManagementProxy()
        NodesProxy()
        UtilsProxy()
    # In child processes, these will be injected as proxies via use_remote()
    PromptServerProxy()


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
    manifest_entries = _filter_blacklisted_entries(manifest_entries)
    if not manifest_entries:
        logger.info(f"{LOG_PREFIX}[Loader] No pyisolate manifests detected under custom_nodes")
        return []
    
    # Set flag to enable ModelSampling proxy (only when isolated nodes exist)
    import os
    os.environ["PYISOLATE_ISOLATION_ACTIVE"] = "1"
    logger.debug(f"{LOG_PREFIX}[Loader] Isolation active: ModelSampling proxy enabled")

    specs: List[IsolatedNodeSpec] = []
    for node_dir, manifest in manifest_entries:
        try:
            load_start = time.perf_counter()
            spec_list = await _load_isolated_node(node_dir, manifest)
            load_time = time.perf_counter() - load_start
            if spec_list:
                isolated_node_timings.append((load_time, node_dir))
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
                logger.debug("%s[Loader] Found pyisolate manifest: %s", LOG_PREFIX, manifest)
                manifest_dirs.append((entry, manifest))
    return manifest_dirs


def _filter_blacklisted_entries(entries: List[tuple[Path, Path]]):
    filtered: List[tuple[Path, Path]] = []
    for node_dir, manifest in entries:
        reason = BLACKLISTED_NODES.get(node_dir.name)
        if reason:
            message = (
                f"{LOG_PREFIX}[Loader] Blocking {node_dir} from isolation: {reason}"
                " Node authors must remove the manifest or ship a compliant version before retrying."
            )
            logger.error(message)
            raise RuntimeError(message)
        filtered.append((node_dir, manifest))
    return filtered


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

    manager_config = ExtensionManagerConfig(venv_root_path=str(PYISOLATE_VENV_ROOT))
    logger.debug(
        "%s[Loader] Using manager config: venv_root=%s",
        LOG_PREFIX,
        manager_config["venv_root_path"],
    )
    manager: ExtensionManager = pyisolate.ExtensionManager(ComfyNodeExtension, manager_config)
    _EXTENSION_MANAGERS.append(manager)

    # Import server only when needed, not at module level
    # This prevents spawn context from importing server before path unification
    import server
    
    extension_config = {
        "name": extension_name,
        "module_path": str(node_dir),
        "isolated": True,
        "dependencies": dependencies,
        "apis": [server.PromptServer],
        "share_torch": share_torch,
    }

    logger.debug(
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
    
    # === ROUTE INJECTION START ===
    # Check for route_manifest.json and inject routes if present
    manifest_path = node_dir / "route_manifest.json"
    if manifest_path.exists():
        logger.info(
            "%s[Loader] Found route_manifest.json for %s, injecting routes...",
            LOG_PREFIX,
            extension_name,
        )
        try:
            from .route_injector import inject_routes
            import server
            num_routes = inject_routes(
                prompt_server=server.PromptServer.instance,
                extension=extension,
                manifest_path=manifest_path,
            )
            logger.info(
                "%s[Loader] ✅ Injected %d routes for %s",
                LOG_PREFIX,
                num_routes,
                extension_name,
            )
        except Exception as e:
            logger.error(
                "%s[Loader] ❌ Route injection failed for %s: %s",
                LOG_PREFIX,
                extension_name,
                e,
            )
    else:
        logger.debug(
            "%s[Loader] No route_manifest.json for %s",
            LOG_PREFIX,
            extension_name,
        )
    # === ROUTE INJECTION END ===
    
    # Register a dummy module in sys.modules so pickle can find classes from the isolated module
    # The normalized extension name is used as the module name in the isolated process
    normalized_name = extension_name.replace("-", "_").replace(".", "_")
    if normalized_name not in sys.modules:
        # Create a minimal dummy module that will satisfy pickle's import requirements
        dummy_module = types.ModuleType(normalized_name)
        dummy_module.__file__ = str(node_dir / "__init__.py")
        dummy_module.__path__ = [str(node_dir)]  # Make it a package so submodules can be imported
        dummy_module.__package__ = normalized_name
        sys.modules[normalized_name] = dummy_module
        logger.debug(
            "%s[Loader] Registered dummy module %s in sys.modules for pickle compatibility",
            LOG_PREFIX,
            normalized_name,
        )
    
    logger.debug(
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
    logger.debug(
        "%s[Loader] %s reported %d node(s)",
        LOG_PREFIX,
        extension_name,
        len(remote_nodes),
    )
    logger.debug(
        "%s[Loader] %s nodes: %s",
        LOG_PREFIX,
        extension_name,
        list(remote_nodes.keys()),
    )
    
    for node_name, display_name in remote_nodes.items():
        # Get full node details - the isolated process will serialize everything to JSON
        details = await extension.get_node_details(node_name)
        details["display_name"] = display_name
        
        stub_cls = _build_stub_class(node_name, details, extension)
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
            logger.debug(
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
        raise RuntimeError(
            f"PyIsolate source missing at {PYISOLATE_EDITABLE_PATH}; cannot load isolated node {extension_name}"
        )
    return deps


def _build_stub_class(node_name: str, info: Dict[str, object], extension: ComfyNodeExtension) -> type:
    function_name = "_pyisolate_execute"
    restored_input_types = _restore_input_types(info.get("input_types", {}))
    logger.debug(
        "%s[Loader] Restored INPUT_TYPES for %s: %s",
        LOG_PREFIX,
        node_name,
        restored_input_types,
    )

    async def _execute(self, **inputs):
        result = await extension.execute_node(node_name, **inputs)
        return result

    def _input_types(cls):
        return restored_input_types

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
    return stub_cls


__all__ = [
    "LOG_PREFIX",
    "get_isolation_logger",
    "initialize_proxies",
    "initialize_isolation_nodes",
    "IsolatedNodeSpec",
]
