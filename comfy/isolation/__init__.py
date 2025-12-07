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
isolated_node_timings: List[tuple[float, Path]] = []

PYISOLATE_VENV_ROOT = Path(folder_paths.base_path) / ".pyisolate_venvs"
PYISOLATE_VENV_ROOT.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(__name__)


def initialize_proxies() -> None:
    from .child_hooks import is_child_process
    if is_child_process():
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
            isolated_node_timings.append((time.perf_counter() - load_start, node_dir))
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


async def notify_execution_graph(needed_class_types: Set[str]) -> None:
    pass


def get_claimed_paths() -> Set[Path]:
    return _CLAIMED_PATHS


__all__ = [
    "LOG_PREFIX",
    "initialize_proxies",
    "initialize_isolation_nodes",
    "start_isolation_loading_early",
    "await_isolation_loading",
    "notify_execution_graph",
    "get_claimed_paths",
    "IsolatedNodeSpec",
]
