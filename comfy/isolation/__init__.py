from __future__ import annotations
import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, TYPE_CHECKING

import folder_paths

from .extension_loader import load_isolated_node
from .manifest_loader import (
    find_manifest_directories,
    filter_blacklisted_entries,
    load_blacklisted_nodes,
)
from .runtime_helpers import build_stub_class, get_class_types_for_extension

if TYPE_CHECKING:
    from pyisolate import ExtensionManager
    from .extension_wrapper import ComfyNodeExtension

LOG_PREFIX = "[I]"
isolated_node_timings: List[tuple[float, Path]] = []


# Load blacklist from external JSON (editable without code changes)
BLACKLISTED_NODES: dict[str, str] = load_blacklisted_nodes()


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
_CLAIMED_PATHS: Set[Path] = set()
_ISOLATION_SCAN_ATTEMPTED = False
_EXTENSION_MANAGERS: List["ExtensionManager"] = []  # Keep alive so subprocesses persist
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

    manifest_entries = find_manifest_directories()
    manifest_entries = filter_blacklisted_entries(manifest_entries, BLACKLISTED_NODES)
    
    global _CLAIMED_PATHS
    _CLAIMED_PATHS = {entry[0].resolve() for entry in manifest_entries}

    if not manifest_entries:
        logger.info(f"{LOG_PREFIX}[Loader] No pyisolate manifests detected under custom_nodes")
        return []
    
    # Set flag to enable ModelSampling proxy (only when isolated nodes exist)
    import os
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
                spec_list = await load_isolated_node(
                    node_dir,
                    manifest,
                    logger,
                    lambda name, info, extension: build_stub_class(
                        name,
                        info,
                        extension,
                        _RUNNING_EXTENSIONS,
                        logger,
                    ),
                    PYISOLATE_EDITABLE_PATH,
                    PYISOLATE_VENV_ROOT,
                    _EXTENSION_MANAGERS,
                )
                spec_list = [
                    IsolatedNodeSpec(
                        node_name=node_name,
                        display_name=display_name,
                        stub_class=stub_cls,
                        module_path=node_dir,
                    )
                    for node_name, display_name, stub_cls in spec_list
                ]
                load_time = time.perf_counter() - load_start
                
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


async def notify_execution_graph(needed_class_types: Set[str]) -> None:
    """Called before execution with the set of node class_types that will run.
    
    Evicts any running isolated processes whose nodes are NOT in the graph.
    This frees resources (RAM, potential VRAM) for the current execution.
    """
    # Find running processes not needed for this execution
    for ext_name, extension in list(_RUNNING_EXTENSIONS.items()):
        ext_class_types = get_class_types_for_extension(
            ext_name,
            _RUNNING_EXTENSIONS,
            _ISOLATED_NODE_SPECS,
        )
        
        # If extension has NO nodes, assume it's a service extension and keep it alive
        if not ext_class_types:
            continue

        # If NONE of this extension's nodes are in the execution graph → evict
        if not ext_class_types.intersection(needed_class_types):
            logger.info(
                "%s[Lifecycle]  Evicting %s (not in execution graph)",
                LOG_PREFIX,
                ext_name,
            )
            extension.stop()
            del _RUNNING_EXTENSIONS[ext_name]


def get_claimed_paths() -> Set[Path]:
    """Return the set of paths claimed by isolation, even if they failed to load."""
    return _CLAIMED_PATHS


__all__ = [
    "LOG_PREFIX",
    "get_isolation_logger",
    "initialize_proxies",
    "initialize_isolation_nodes",
    "start_isolation_loading_early",
    "await_isolation_loading",
    "notify_execution_graph",
    "get_claimed_paths",
    "IsolatedNodeSpec",
]
