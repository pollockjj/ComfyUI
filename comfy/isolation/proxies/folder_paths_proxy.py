from __future__ import annotations
from typing import Optional, List, Tuple, Set, Dict

import folder_paths
from pyisolate import ProxiedSingleton

class FolderPathsProxy(ProxiedSingleton):
    """
    Dynamic proxy for folder_paths.
    Uses __getattr__ for most lookups, with explicit handling for
    mutable collections to ensure efficient by-value transfer.
    """

    def __getattr__(self, name):
        return getattr(folder_paths, name)

    # Return copies of dicts to avoid RPC chatter for every key access
    # and to ensure we send a snapshot rather than a proxy to a dict
    @property
    def folder_names_and_paths(self) -> Dict:
        return dict(folder_paths.folder_names_and_paths)

    @property
    def extension_mimetypes_cache(self) -> Dict:
        return dict(folder_paths.extension_mimetypes_cache)

    @property
    def filename_list_cache(self) -> Dict:
        return dict(folder_paths.filename_list_cache)

