from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import folder_paths

LOG_PREFIX = "]["


def find_manifest_directories() -> List[Tuple[Path, Path]]:
    manifest_dirs: List[Tuple[Path, Path]] = []
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


CACHE_DIR_NAME = ".pyisolate_cache"
CACHE_FILE_NAME = "node_info.json"


def get_cache_path(node_dir: Path) -> Path:
    return node_dir / CACHE_DIR_NAME / CACHE_FILE_NAME


def is_cache_valid(node_dir: Path, manifest_path: Path) -> bool:
    # BACKOUT: Caching disabled - always return False to force fresh spawn
    return False


def load_from_cache(node_dir: Path) -> Optional[Dict[str, Dict]]:
    # BACKOUT: Caching disabled - always return None to force fresh spawn
    return None


def save_to_cache(node_dir: Path, node_data: Dict[str, Dict]) -> None:
    # BACKOUT: Caching disabled - no-op, do not write cache files
    return


__all__ = [
    "LOG_PREFIX",
    "find_manifest_directories",
    "get_cache_path",
    "is_cache_valid",
    "load_from_cache",
    "save_to_cache",
]
