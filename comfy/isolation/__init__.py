from __future__ import annotations
import asyncio
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, TYPE_CHECKING
import folder_paths
from .extension_loader import load_isolated_node
from .manifest_loader import find_manifest_directories
from .runtime_helpers import build_stub_class, get_class_types_for_extension

if TYPE_CHECKING:
    from pyisolate import ExtensionManager
    from .extension_wrapper import ComfyNodeExtension

LOG_PREFIX = "]["
isolated_node_timings: List[tuple[float, Path, int]] = []

PYISOLATE_VENV_ROOT = Path(folder_paths.base_path) / ".pyisolate_venvs"
PYISOLATE_VENV_ROOT.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(__name__)





def initialize_proxies() -> None:
    from .child_hooks import is_child_process
    is_child = is_child_process()


    if is_child:
        from .child_hooks import initialize_child_process
        initialize_child_process()
    else:
        from .host_hooks import initialize_host_process
        initialize_host_process()

@dataclass(frozen=True)
class IsolatedNodeSpec:
    node_name: str
    display_name: str
    stub_class: type
    module_path: Path


_ISOLATED_NODE_SPECS: List[IsolatedNodeSpec] = []
_CLAIMED_PATHS: Set[Path] = set()
_ISOLATION_SCAN_ATTEMPTED = False
_EXTENSION_MANAGERS: List["ExtensionManager"] = []
_RUNNING_EXTENSIONS: Dict[str, "ComfyNodeExtension"] = {}
_ISOLATION_BACKGROUND_TASK: Optional["asyncio.Task[List[IsolatedNodeSpec]]"] = None
_EARLY_START_TIME: Optional[float] = None


def start_isolation_loading_early(loop: "asyncio.AbstractEventLoop") -> None:
    global _ISOLATION_BACKGROUND_TASK, _EARLY_START_TIME
    if _ISOLATION_BACKGROUND_TASK is not None:
        return
    _EARLY_START_TIME = time.perf_counter()
    _ISOLATION_BACKGROUND_TASK = loop.create_task(initialize_isolation_nodes())


async def await_isolation_loading() -> List[IsolatedNodeSpec]:
    global _ISOLATION_BACKGROUND_TASK, _EARLY_START_TIME
    if _ISOLATION_BACKGROUND_TASK is not None:
        specs = await _ISOLATION_BACKGROUND_TASK
        return specs
    return await initialize_isolation_nodes()


async def initialize_isolation_nodes() -> List[IsolatedNodeSpec]:
    global _ISOLATED_NODE_SPECS, _ISOLATION_SCAN_ATTEMPTED, _CLAIMED_PATHS

    if _ISOLATED_NODE_SPECS:
        return _ISOLATED_NODE_SPECS

    if _ISOLATION_SCAN_ATTEMPTED:
        return []

    _ISOLATION_SCAN_ATTEMPTED = True
    manifest_entries = find_manifest_directories()
    _CLAIMED_PATHS = {entry[0].resolve() for entry in manifest_entries}

    if not manifest_entries:
        return []

    os.environ["PYISOLATE_ISOLATION_ACTIVE"] = "1"
    concurrency_limit = max(1, (os.cpu_count() or 4) // 2)
    semaphore = asyncio.Semaphore(concurrency_limit)

    async def load_with_semaphore(node_dir: Path, manifest: Path) -> List[IsolatedNodeSpec]:
        async with semaphore:
            load_start = time.perf_counter()
            spec_list = await load_isolated_node(
                node_dir,
                manifest,
                logger,
                lambda name, info, extension: build_stub_class(
                    name, info, extension, _RUNNING_EXTENSIONS, logger,
                ),
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
            isolated_node_timings.append((time.perf_counter() - load_start, node_dir, len(spec_list)))
            return spec_list

    tasks = [load_with_semaphore(node_dir, manifest) for node_dir, manifest in manifest_entries]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    specs: List[IsolatedNodeSpec] = []
    for result in results:
        if isinstance(result, Exception):
            raise result
        specs.extend(result)

    _ISOLATED_NODE_SPECS = specs
    return list(_ISOLATED_NODE_SPECS)


def _get_class_types_for_extension(extension_name: str) -> Set[str]:
    """Get all node class types (node names) belonging to an extension."""
    extension = _RUNNING_EXTENSIONS.get(extension_name)
    if not extension:
        return set()

    ext_path = Path(extension.module_path)
    class_types = set()
    for spec in _ISOLATED_NODE_SPECS:
        if spec.module_path.resolve() == ext_path.resolve():
            class_types.add(spec.node_name)

    return class_types


async def notify_execution_graph(needed_class_types: Set[str]) -> None:
    """Evict running extensions not needed for current execution."""
    for ext_name, extension in list(_RUNNING_EXTENSIONS.items()):
        ext_class_types = _get_class_types_for_extension(ext_name)

        # If NONE of this extension's nodes are in the execution graph → evict
        if not ext_class_types.intersection(needed_class_types):
            logger.info(
                f"][ {ext_name} isolated custom_node not in execution graph, evicting"
            )
            extension.stop()
            del _RUNNING_EXTENSIONS[ext_name]


def get_claimed_paths() -> Set[Path]:
    return _CLAIMED_PATHS


def update_rpc_event_loops(loop: "asyncio.AbstractEventLoop | None" = None) -> None:
    """Update all active RPC instances with the current event loop.

    This MUST be called at the start of each workflow execution to ensure
    RPC calls are scheduled on the correct event loop. This handles the case
    where asyncio.run() creates a new event loop for each workflow.

    Args:
        loop: The event loop to use. If None, uses asyncio.get_running_loop().
    """
    if loop is None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.get_event_loop()

    update_count = 0

    # Update RPCs from ExtensionManagers
    for manager in _EXTENSION_MANAGERS:
        if not hasattr(manager, 'extensions'):
            continue
        for name, extension in manager.extensions.items():
            if hasattr(extension, 'rpc') and extension.rpc is not None:
                if hasattr(extension.rpc, 'update_event_loop'):
                    extension.rpc.update_event_loop(loop)
                    update_count += 1
                    logger.debug(f"{LOG_PREFIX}Updated loop on extension '{name}'")

    # Also update RPCs from running extensions (they may have direct RPC refs)
    for name, extension in _RUNNING_EXTENSIONS.items():
        if hasattr(extension, 'rpc') and extension.rpc is not None:
            if hasattr(extension.rpc, 'update_event_loop'):
                extension.rpc.update_event_loop(loop)
                update_count += 1
                logger.debug(f"{LOG_PREFIX}Updated loop on running extension '{name}'")

    if update_count > 0:
        logger.debug(f"{LOG_PREFIX}Updated event loop on {update_count} RPC instances")
    else:
        logger.debug(f"{LOG_PREFIX}No RPC instances found to update (managers={len(_EXTENSION_MANAGERS)}, running={len(_RUNNING_EXTENSIONS)})")


__all__ = [
    "LOG_PREFIX",
    "initialize_proxies",
    "initialize_isolation_nodes",
    "start_isolation_loading_early",
    "await_isolation_loading",
    "notify_execution_graph",
    "get_claimed_paths",
    "update_rpc_event_loops",
    "IsolatedNodeSpec",
    "get_class_types_for_extension",
]
