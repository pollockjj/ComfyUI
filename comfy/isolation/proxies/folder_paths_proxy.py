from __future__ import annotations

from typing import Optional, List, Tuple, Set

import folder_paths

try:
    from pyisolate import ProxiedSingleton
except ImportError:
    class ProxiedSingleton:
        pass


class FolderPathsProxy(ProxiedSingleton):
    _instance: Optional['FolderPathsProxy'] = None

    def __new__(cls) -> 'FolderPathsProxy':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_output_directory(self) -> str:
        return folder_paths.get_output_directory()

    def get_temp_directory(self) -> str:
        return folder_paths.get_temp_directory()

    def get_input_directory(self) -> str:
        return folder_paths.get_input_directory()

    def get_user_directory(self) -> str:
        return folder_paths.get_user_directory()

    def get_directory_by_type(self, type_name: str) -> Optional[str]:
        return folder_paths.get_directory_by_type(type_name)

    @property
    def output_directory(self) -> str:
        return folder_paths.output_directory

    @property
    def temp_directory(self) -> str:
        return folder_paths.temp_directory

    @property
    def input_directory(self) -> str:
        return folder_paths.input_directory

    @property
    def user_directory(self) -> str:
        return folder_paths.user_directory

    @property
    def base_path(self) -> str:
        return folder_paths.base_path

    @property
    def models_dir(self) -> str:
        return folder_paths.models_dir

    def get_folder_paths(self, folder_name: str) -> List[str]:
        return folder_paths.get_folder_paths(folder_name)

    def get_full_path(self, folder_name: str, filename: str) -> Optional[str]:
        return folder_paths.get_full_path(folder_name, filename)

    def get_full_path_or_raise(self, folder_name: str, filename: str) -> str:
        return folder_paths.get_full_path_or_raise(folder_name, filename)

    def get_filename_list(self, folder_name: str) -> List[str]:
        return folder_paths.get_filename_list(folder_name)

    def add_model_folder_path(self, folder_name: str, full_folder_path: str, is_default: bool = False) -> None:
        return folder_paths.add_model_folder_path(folder_name, full_folder_path, is_default)

    def get_annotated_filepath(self, name: str, default_dir: Optional[str] = None) -> str:
        return folder_paths.get_annotated_filepath(name, default_dir)

    def exists_annotated_filepath(self, name: str) -> bool:
        return folder_paths.exists_annotated_filepath(name)

    def get_save_image_path(
        self,
        filename_prefix: str,
        output_dir: str,
        image_width: int = 0,
        image_height: int = 0
    ) -> Tuple[str, str, int, str, str]:
        return folder_paths.get_save_image_path(filename_prefix, output_dir, image_width, image_height)

    @property
    def supported_pt_extensions(self) -> Set[str]:
        return folder_paths.supported_pt_extensions

    @property
    def folder_names_and_paths(self) -> dict:
        return dict(folder_paths.folder_names_and_paths)

    def get_input_subfolders(self) -> List[str]:
        return folder_paths.get_input_subfolders()

    def filter_files_extensions(self, files: List[str], extensions: List[str]) -> List[str]:
        return folder_paths.filter_files_extensions(files, extensions)

    def filter_files_content_types(self, files: List[str], content_types: List[str]) -> List[str]:
        return folder_paths.filter_files_content_types(files, content_types)
