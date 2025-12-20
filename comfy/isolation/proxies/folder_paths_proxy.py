from __future__ import annotations

from typing import Optional, List, Tuple, Set

import folder_paths

from pyisolate import ProxiedSingleton


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
    
    def set_output_directory(self, output_dir: str) -> None:
        """Set the output directory."""
        return folder_paths.set_output_directory(output_dir)
    
    def set_temp_directory(self, temp_dir: str) -> None:
        """Set the temp directory."""
        return folder_paths.set_temp_directory(temp_dir)
    
    def set_input_directory(self, input_dir: str) -> None:
        """Set the input directory."""
        return folder_paths.set_input_directory(input_dir)
    
    def set_user_directory(self, user_dir: str) -> None:
        """Set the user directory."""
        return folder_paths.set_user_directory(user_dir)
    
    def get_system_user_directory(self, name: str = "system") -> str:
        """Get path to a System User directory (prefixed with '__')."""
        return folder_paths.get_system_user_directory(name)
    
    def get_public_user_directory(self, user_id: str) -> Optional[str]:
        """Get path to a Public User directory for HTTP endpoint access."""
        return folder_paths.get_public_user_directory(user_id)

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
    
    # Helper functions
    def map_legacy(self, folder_name: str) -> str:
        """Map legacy folder names to current names."""
        return folder_paths.map_legacy(folder_name)
    
    def annotated_filepath(self, name: str) -> Tuple[str, Optional[str]]:
        """Parse annotated filepath (e.g., 'file.ext [output]')."""
        return folder_paths.annotated_filepath(name)
    
    def recursive_search(self, directory: str, excluded_dir_names: Optional[List[str]] = None) -> Tuple[List[str], dict]:
        """Recursively search directory for files."""
        return folder_paths.recursive_search(directory, excluded_dir_names)
    
    def get_filename_list_(self, folder_name: str) -> Tuple[List[str], dict, float]:
        """Get filename list with folder mtimes and timestamp."""
        return folder_paths.get_filename_list_(folder_name)
    
    def cached_filename_list_(self, folder_name: str) -> Optional[Tuple[List[str], dict, float]]:
        """Get cached filename list if valid."""
        return folder_paths.cached_filename_list_(folder_name)
    
    # Constants
    @property
    def SYSTEM_USER_PREFIX(self) -> str:
        """System user prefix ('__')."""
        return folder_paths.SYSTEM_USER_PREFIX
    
    @property
    def extension_mimetypes_cache(self) -> dict:
        """Extension mimetype cache (return copy to prevent mutation)."""
        return dict(folder_paths.extension_mimetypes_cache)
    
    @property
    def filename_list_cache(self) -> dict:
        """Filename list cache (return copy to prevent mutation)."""
        return dict(folder_paths.filename_list_cache)
    
    @property
    def cache_helper(self):
        """Cache helper context manager."""
        return folder_paths.cache_helper

