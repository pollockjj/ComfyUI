from __future__ import annotations
import logging
import sys
import time
import types
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Set
import yaml
import folder_paths

import pyisolate
from pyisolate import ExtensionManager, ExtensionManagerConfig
from .extension_wrapper import ComfyNodeExtension

# Conditional imports for experimental features (gated behind PYISOLATE_DEV=1)
import os
IS_DEV = os.environ.get("PYISOLATE_DEV") == "1"

if IS_DEV:
    from .development.clip_proxy import CLIPRegistry, CLIPProxy, maybe_wrap_clip_for_isolation
    from .development.model_patcher_proxy import (
        ModelPatcherRegistry,
        ModelPatcherProxy,
        maybe_wrap_model_for_isolation,
    )
else:
    # Stub exports for V1.0 (no advanced serialization)
    CLIPRegistry = None  # type: ignore
    CLIPProxy = None  # type: ignore
    maybe_wrap_clip_for_isolation = lambda x: x  # type: ignore
    ModelPatcherRegistry = None  # type: ignore
    ModelPatcherProxy = None  # type: ignore
    maybe_wrap_model_for_isolation = lambda x: x  # type: ignore

LOG_PREFIX = "]["

# Module-level list to store timing data for ComfyManager-style report
isolated_node_timings: List[tuple[float, Path]] = []


class _AnyTypeProxy(str):
    """Replacement for custom AnyType objects used by some nodes."""

    def __new__(cls, value: str = "*"):
        return super().__new__(cls, value)

    def __ne__(self, other):  # type: ignore[override]
        return False


class _FlexibleOptionalInputProxy(dict):
    """Replacement for FlexibleOptionalInputType to allow dynamic inputs.
    
    This mirrors the behavior of FlexibleOptionalInputType from rgthree/ComfyUI-Lora-Manager:
    - __contains__ always returns True (accept any input key)
    - __getitem__ always returns (self.type,) - a tuple with the type
    """

    def __init__(self, flex_type, data: Optional[Dict[str, object]] = None):
        super().__init__()
        self.type = flex_type

    def __getitem__(self, key):  # type: ignore[override]
        # Always return a tuple with the type, matching original behavior
        return (self.type,)

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
        # First check if the entire section is a special value (like FlexibleOptionalInputType)
        if isinstance(entries, dict) and entries.get("__pyisolate_flexible_optional__"):
            # The whole 'optional' section is a FlexibleOptionalInputType
            restored[section] = _restore_special_value(entries)
        elif isinstance(entries, dict):
            # Normal dict of entries
            restored[section] = {k: _restore_special_value(v) for k, v in entries.items()}
        else:
            restored[section] = _restore_special_value(entries)
    return restored


def _load_blacklisted_nodes() -> dict[str, str]:
    """Load blacklisted nodes from external JSON file.
    
    Blacklisted nodes cannot be isolated due to architectural constraints
    (e.g., direct GPU management, privileged system access).
    """
    blacklist_path = Path(__file__).parent / "blacklisted_nodes.json"
    if not blacklist_path.exists():
        return {}
    
    try:
        with open(blacklist_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("nodes", {})
    except Exception as e:
        logging.getLogger(__name__).warning(
            "%s[Config] Failed to load blacklist from %s: %s",
            LOG_PREFIX, blacklist_path, e
        )
        return {}


# Load blacklist from external JSON (editable without code changes)
BLACKLISTED_NODES: dict[str, str] = _load_blacklisted_nodes()


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
    
    Detects host vs child process and runs appropriate initialization.
    All child-specific logic is in child_hooks.py, not in core files.
    """
    from .child_hooks import is_child_process
    
    if is_child_process():
        from .child_hooks import initialize_child_process
        initialize_child_process()
    else:
        from .host_hooks import initialize_host_process
        initialize_host_process()


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
_RUNNING_EXTENSIONS: Dict[str, "Extension"] = {}  # Track running extensions for eviction

# Phase 2: Early orchestration - background task for parallel startup
_ISOLATION_BACKGROUND_TASK: Optional["asyncio.Task[List[IsolatedNodeSpec]]"] = None
_EARLY_START_TIME: Optional[float] = None


def start_isolation_loading_early(loop: "asyncio.AbstractEventLoop") -> None:
    """Start isolated node loading in the background BEFORE nodes.py needs them.
    
    Called from main.py immediately after PromptServer creation to maximize
    parallelism with builtin node loading.
    """
    global _ISOLATION_BACKGROUND_TASK, _EARLY_START_TIME

    if _ISOLATION_BACKGROUND_TASK is not None:
        return

    if pyisolate is None:
        return

    _EARLY_START_TIME = time.perf_counter()

    # Create the task but don't await it yet
    _ISOLATION_BACKGROUND_TASK = loop.create_task(initialize_isolation_nodes())


async def await_isolation_loading() -> List[IsolatedNodeSpec]:
    """Await the background isolation loading task, or start loading if not already started.
    
    Returns the list of isolated node specs.
    """
    global _ISOLATION_BACKGROUND_TASK, _EARLY_START_TIME
    
    if _ISOLATION_BACKGROUND_TASK is not None:
        # Early start was triggered - await the existing task
        specs = await _ISOLATION_BACKGROUND_TASK
        
        if _EARLY_START_TIME is not None:
            overlap_time = time.perf_counter() - _EARLY_START_TIME
        
        return specs
    else:
        return await initialize_isolation_nodes()


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
        return []

    manifest_entries = _find_manifest_directories()
    manifest_entries = _filter_blacklisted_entries(manifest_entries)
    if not manifest_entries:
        logger.info(f"{LOG_PREFIX}[Loader] No pyisolate manifests detected under custom_nodes")
        return []
    
    # Set flag to enable ModelSampling proxy (only when isolated nodes exist)
    import os
    import asyncio
    os.environ["PYISOLATE_ISOLATION_ACTIVE"] = "1"

    total_nodes = len(manifest_entries)
    batch_start = time.perf_counter()
    
    # Parallel loading with throttling to prevent thundering herd
    concurrency_limit = max(1, (os.cpu_count() or 4) // 2)
    semaphore = asyncio.Semaphore(concurrency_limit)
    
    async def load_with_semaphore(node_dir: Path, manifest: Path) -> List[IsolatedNodeSpec]:
        """Load a single node with semaphore throttling."""
        async with semaphore:
            load_start = time.perf_counter()
            try:
                spec_list = await _load_isolated_node(node_dir, manifest)
                load_time = time.perf_counter() - load_start
                
                if spec_list:
                    isolated_node_timings.append((load_time, node_dir))
                return spec_list
            except Exception as exc:
                logger.error(
                    "%s[Loader] Failed to initialize isolated node at %s: %s",
                    LOG_PREFIX,
                    node_dir,
                    exc,
                )
                raise
    
    # Launch all loads in parallel (throttled by semaphore)
    tasks = [
        load_with_semaphore(node_dir, manifest)
        for node_dir, manifest in manifest_entries
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Collect specs and handle any exceptions
    specs: List[IsolatedNodeSpec] = []
    for result in results:
        if isinstance(result, Exception):
            raise result  # Re-raise first exception
        specs.extend(result)
    
    batch_time = time.perf_counter() - batch_start

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


CACHE_DIR_NAME = ".pyisolate_cache"
CACHE_FILE_NAME = "node_info.json"

def _get_cache_path(node_dir: Path) -> Path:
    return node_dir / CACHE_DIR_NAME / CACHE_FILE_NAME

def _is_cache_valid(node_dir: Path, manifest_path: Path) -> bool:
    cache_path = _get_cache_path(node_dir)
    if not cache_path.exists():
        return False
    # Cache valid if newer than manifest
    return cache_path.stat().st_mtime > manifest_path.stat().st_mtime

def _load_from_cache(node_dir: Path) -> Optional[Dict[str, Dict]]:
    cache_path = _get_cache_path(node_dir)
    try:
        return json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return None

def _save_to_cache(node_dir: Path, node_data: Dict[str, Dict]) -> None:
    cache_path = _get_cache_path(node_dir)
    cache_path.parent.mkdir(exist_ok=True)
    cache_path.write_text(json.dumps(node_data, indent=2), encoding="utf-8")


async def _load_isolated_node(node_dir: Path, manifest_path: Path) -> List[IsolatedNodeSpec]:
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle) or {}

    if not manifest.get("isolated", False):
        return []

    dependencies = list(manifest.get("dependencies", []) or [])
    share_torch = manifest.get("share_torch", True)
    extension_name = manifest.get("name", node_dir.name)
    dependencies = _augment_dependencies_with_pyisolate(dependencies, extension_name)

    manager_config = ExtensionManagerConfig(venv_root_path=str(PYISOLATE_VENV_ROOT))
    manager: ExtensionManager = pyisolate.ExtensionManager(ComfyNodeExtension, manager_config)
    _EXTENSION_MANAGERS.append(manager)

    # Build APIs list based on IS_DEV flag
    # V1.0: No advanced proxies needed for basic dependency isolation
    apis = []
    if IS_DEV:
        # Only import and register advanced proxies when IS_DEV enabled
        import server
        from comfy.isolation.development.clip_proxy import CLIPRegistry
        from comfy.isolation.development.model_patcher_proxy import ModelPatcherRegistry
        apis = [server.PromptServer, CLIPRegistry, ModelPatcherRegistry]
    
    extension_config = {
        "name": extension_name,
        "module_path": str(node_dir),
        "isolated": True,
        "dependencies": dependencies,
        "apis": apis,
        "share_torch": share_torch,
    }

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
    route_manifest_path = node_dir / "route_manifest.json"
    if route_manifest_path.exists():
        try:
            from .route_injector import inject_routes
            import server
            num_routes = inject_routes(
                prompt_server=server.PromptServer.instance,
                extension=extension,
                manifest_path=route_manifest_path,
            )
        except Exception as e:
            logger.error(
                "%s[Loader]  Route injection failed for %s: %s",
                LOG_PREFIX,
                extension_name,
                e,
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
    
    specs: List[IsolatedNodeSpec] = []
    
    # Try cache first
    if _is_cache_valid(node_dir, manifest_path):
        cached_data = _load_from_cache(node_dir)
        if cached_data:
            logger.info(f"{LOG_PREFIX}[Loader] Loaded {extension_name} from cache (lazy spawn)")
            for node_name, details in cached_data.items():
                stub_cls = _build_stub_class(node_name, details, extension)
                specs.append(
                    IsolatedNodeSpec(
                        node_name=node_name,
                        display_name=details.get("display_name", node_name),
                        stub_class=stub_cls,
                        module_path=node_dir,
                    )
                )
            return specs

    # Cache miss or invalid - full load (triggers spawn)
    remote_nodes: Dict[str, str] = await extension.list_nodes()
    
    if not remote_nodes:
        raise RuntimeError(
            f"{LOG_PREFIX}[Loader] Isolated node {extension_name} at {node_dir} reported zero NODE_CLASS_MAPPINGS"
        )
    
    cache_data_to_save = {}
    
    for node_name, display_name in remote_nodes.items():
        # Get full node details - the isolated process will serialize everything to JSON
        details = await extension.get_node_details(node_name)
        details["display_name"] = display_name
        
        cache_data_to_save[node_name] = details
        
        stub_cls = _build_stub_class(node_name, details, extension)
        specs.append(
            IsolatedNodeSpec(
                node_name=node_name,
                display_name=display_name,
                stub_class=stub_cls,
                module_path=node_dir,
            )
        )
    
    # Save to cache
    _save_to_cache(node_dir, cache_data_to_save)
    
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

    async def _execute(self, **inputs):
        # Lazy spawn: ensure process is running before RPC
        extension.ensure_process_started()
        _RUNNING_EXTENSIONS[extension.name] = extension

        # ModelPatcher/CLIP serialization now uses ProxiedSingleton registries
        # No scoped registry needed - the singletons handle lifecycle
        try:
            from pyisolate._internal.model_serialization import (
                serialize_for_isolation,
                deserialize_from_isolation,
            )
            
            # Serialize inputs (CLIP, ModelPatcher, etc. → Refs)
            inputs = serialize_for_isolation(inputs)
            
            result = await extension.execute_node(node_name, **inputs)
            
            # Deserialize result (Refs → real objects)
            # This converts ModelPatcherRef back to actual ModelPatcher on host
            result = deserialize_from_isolation(result)
            
            return result
        except ImportError as e:
            logger.warning(
                "%s[Serialization] Serialization not available: %s",
                LOG_PREFIX,
                e,
            )
            # Fallback: execute without serialization
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
    
    # Add OUTPUT_IS_LIST if present
    output_is_list = info.get("output_is_list")
    if output_is_list is not None:
        attributes["OUTPUT_IS_LIST"] = tuple(output_is_list)

    display_name = info.get("display_name") or node_name
    class_name = f"PyIsolate_{node_name}".replace(" ", "_")
    stub_cls = type(class_name, (), attributes)
    stub_cls.__doc__ = f"PyIsolate proxy node for {display_name}"
    return stub_cls


def _get_class_types_for_extension(extension_name: str) -> Set[str]:
    """Get all node class types (node names) belonging to an extension."""
    # Find the extension object to get its module path
    extension = _RUNNING_EXTENSIONS.get(extension_name)
    if not extension:
        return set()
    
    ext_path = Path(extension.module_path)
    
    class_types = set()
    for spec in _ISOLATED_NODE_SPECS:
        # Compare paths (resolve to be safe)
        if spec.module_path.resolve() == ext_path.resolve():
            class_types.add(spec.node_name)
            
    return class_types


async def notify_execution_graph(needed_class_types: Set[str]) -> None:
    """Called before execution with the set of node class_types that will run.
    
    Evicts any running isolated processes whose nodes are NOT in the graph.
    This frees resources (RAM, potential VRAM) for the current execution.
    """
    # Find running processes not needed for this execution
    for ext_name, extension in list(_RUNNING_EXTENSIONS.items()):
        ext_class_types = _get_class_types_for_extension(ext_name)
        
        # If NONE of this extension's nodes are in the execution graph → evict
        if not ext_class_types.intersection(needed_class_types):
            logger.info(
                "%s[Lifecycle]  Evicting %s (not in execution graph)",
                LOG_PREFIX,
                ext_name,
            )
            extension.stop()
            del _RUNNING_EXTENSIONS[ext_name]


__all__ = [
    "LOG_PREFIX",
    "get_isolation_logger",
    "initialize_proxies",
    "initialize_isolation_nodes",
    "start_isolation_loading_early",
    "await_isolation_loading",
    "notify_execution_graph",
    "IsolatedNodeSpec",
    "CLIPRegistry",
    "CLIPProxy",
    "maybe_wrap_clip_for_isolation",
    "ModelPatcherRegistry",
    "ModelPatcherProxy",
    "maybe_wrap_model_for_isolation",
]
