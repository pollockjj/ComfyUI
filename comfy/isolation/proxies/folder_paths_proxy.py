from __future__ import annotations
from typing import Dict

import folder_paths
from pyisolate import ProxiedSingleton

class FolderPathsProxy(ProxiedSingleton):
    """
    Proxy for folder_paths.
    Explicitly implements methods to ensure correct RPC delegation.
    """

    # -------------------------------------------------------------------------
    # Core Path Resolution
    # -------------------------------------------------------------------------

    def get_folder_paths(self, folder_name: str):
        if IS_CHILD_PROCESS:
            return self._call_rpc("get_folder_paths", folder_name)
        return folder_paths.get_folder_paths(folder_name)

    def get_filename_list(self, folder_name: str):
        if IS_CHILD_PROCESS:
            return self._call_rpc("get_filename_list", folder_name)
        return folder_paths.get_filename_list(folder_name)

    def get_save_image_path(self, filename_prefix: str, output_dir: str, image_width=0, image_height=0):
        if IS_CHILD_PROCESS:
            return self._call_rpc("get_save_image_path", filename_prefix, output_dir, image_width, image_height)
        return folder_paths.get_save_image_path(filename_prefix, output_dir, image_width, image_height)

    def get_input_subfolders(self):
        if IS_CHILD_PROCESS:
            return self._call_rpc("get_input_subfolders")
        return folder_paths.get_input_subfolders()

    def get_full_path(self, folder_name: str, filename: str):
        if IS_CHILD_PROCESS:
            return self._call_rpc("get_full_path", folder_name, filename)
        return folder_paths.get_full_path(folder_name, filename)

    def get_full_path_or_raise(self, folder_name: str, filename: str):
        if IS_CHILD_PROCESS:
            return self._call_rpc("get_full_path_or_raise", folder_name, filename)
        return folder_paths.get_full_path_or_raise(folder_name, filename)

    # -------------------------------------------------------------------------
    # Advanced Path Utils
    # -------------------------------------------------------------------------

    def add_model_folder_path(self, folder_name: str, full_folder_path: str, is_default: bool = False):
        if IS_CHILD_PROCESS:
            return self._call_rpc("add_model_folder_path", folder_name, full_folder_path, is_default)
        return folder_paths.add_model_folder_path(folder_name, full_folder_path, is_default)

    def exists_annotated_filepath(self, name):
        if IS_CHILD_PROCESS:
            return self._call_rpc("exists_annotated_filepath", name)
        return folder_paths.exists_annotated_filepath(name)

    def recursive_search(self, directory: str, excluded_dir_names: list[str] | None=None):
        if IS_CHILD_PROCESS:
            return self._call_rpc("recursive_search", directory, excluded_dir_names)
        return folder_paths.recursive_search(directory, excluded_dir_names)

    def filter_files_extensions(self, files, extensions):
        if IS_CHILD_PROCESS:
            return self._call_rpc("filter_files_extensions", files, extensions)
        return folder_paths.filter_files_extensions(files, extensions)

    # -------------------------------------------------------------------------
    # Properties (Cached Snapshots)
    # -------------------------------------------------------------------------

    @property
    def folder_names_and_paths(self) -> Dict:
        return dict(folder_paths.folder_names_and_paths)

    @property
    def extension_mimetypes_cache(self) -> Dict:
        return dict(folder_paths.extension_mimetypes_cache)

    @property
    def filename_list_cache(self) -> Dict:
        return dict(folder_paths.filename_list_cache)

