"""ProxiedSingleton for folder_paths module.

Rev 1.0 Implementation - Complete proxy coverage per architecture spec.
"""

from __future__ import annotations

import logging
from typing import Optional, List, Tuple, Set

import folder_paths

try:
    from pyisolate import ProxiedSingleton
except ImportError:
    # Fallback when pyisolate not installed
    class ProxiedSingleton:
        pass

LOG_PREFIX = "[I]"
logger = logging.getLogger(__name__)


class FolderPathsProxy(ProxiedSingleton):
    """Proxy for folder_paths module providing path resolution for isolated nodes.
    
    All methods forward to the real folder_paths in the host process.
    This proxy enables isolated nodes to access ComfyUI path configuration
    without needing direct access to the host's folder_paths module.
    
    Rev 1.0: Complete coverage per PYISOLATE_COMFY_INTEGRATION_ARCHITECTURE.md
    """
    
    _instance: Optional['FolderPathsProxy'] = None
    
    def __new__(cls) -> 'FolderPathsProxy':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    # =========================================================================
    # Directory Getters (Core)
    # =========================================================================
    
    def get_output_directory(self) -> str:
        """Get output directory path."""
        result = folder_paths.get_output_directory()
        logger.debug(f"{LOG_PREFIX}[FolderPathsProxy] get_output_directory() → {result}")
        return result
    
    def get_temp_directory(self) -> str:
        """Get temp directory path."""
        result = folder_paths.get_temp_directory()
        logger.debug(f"{LOG_PREFIX}[FolderPathsProxy] get_temp_directory() → {result}")
        return result
    
    def get_input_directory(self) -> str:
        """Get input directory path."""
        result = folder_paths.get_input_directory()
        logger.debug(f"{LOG_PREFIX}[FolderPathsProxy] get_input_directory() → {result}")
        return result
    
    def get_user_directory(self) -> str:
        """Get user directory path."""
        result = folder_paths.get_user_directory()
        logger.debug(f"{LOG_PREFIX}[FolderPathsProxy] get_user_directory() → {result}")
        return result
    
    def get_directory_by_type(self, type_name: str) -> Optional[str]:
        """Get directory by type name (output, temp, input)."""
        result = folder_paths.get_directory_by_type(type_name)
        logger.debug(f"{LOG_PREFIX}[FolderPathsProxy] get_directory_by_type({type_name}) → {result}")
        return result
    
    # =========================================================================
    # Directory Properties (for direct attribute access)
    # =========================================================================
    
    @property
    def output_directory(self) -> str:
        """Get output directory path (property access)."""
        result = folder_paths.output_directory
        logger.debug(f"{LOG_PREFIX}[FolderPathsProxy] output_directory → {result}")
        return result
    
    @property
    def temp_directory(self) -> str:
        """Get temp directory path (property access)."""
        result = folder_paths.temp_directory
        logger.debug(f"{LOG_PREFIX}[FolderPathsProxy] temp_directory → {result}")
        return result
    
    @property
    def input_directory(self) -> str:
        """Get input directory path (property access)."""
        result = folder_paths.input_directory
        logger.debug(f"{LOG_PREFIX}[FolderPathsProxy] input_directory → {result}")
        return result
    
    @property
    def user_directory(self) -> str:
        """Get user directory path (property access)."""
        result = folder_paths.user_directory
        logger.debug(f"{LOG_PREFIX}[FolderPathsProxy] user_directory → {result}")
        return result
    
    @property
    def base_path(self) -> str:
        """Get ComfyUI base path."""
        result = folder_paths.base_path
        logger.debug(f"{LOG_PREFIX}[FolderPathsProxy] base_path → {result}")
        return result
    
    @property
    def models_dir(self) -> str:
        """Get base models directory path."""
        result = folder_paths.models_dir
        logger.debug(f"{LOG_PREFIX}[FolderPathsProxy] models_dir → {result}")
        return result
    
    # =========================================================================
    # Model/File Path Resolution
    # =========================================================================
    
    def get_folder_paths(self, folder_name: str) -> List[str]:
        """Get all registered paths for a folder type."""
        result = folder_paths.get_folder_paths(folder_name)
        logger.debug(f"{LOG_PREFIX}[FolderPathsProxy] get_folder_paths({folder_name}) → {len(result)} paths")
        return result
    
    def get_full_path(self, folder_name: str, filename: str) -> Optional[str]:
        """Get full path of a file in a folder."""
        result = folder_paths.get_full_path(folder_name, filename)
        logger.debug(f"{LOG_PREFIX}[FolderPathsProxy] get_full_path({folder_name}, {filename}) → {result}")
        return result
    
    def get_full_path_or_raise(self, folder_name: str, filename: str) -> str:
        """Get full path of a file in a folder, raises if not found."""
        result = folder_paths.get_full_path_or_raise(folder_name, filename)
        logger.debug(f"{LOG_PREFIX}[FolderPathsProxy] get_full_path_or_raise({folder_name}, {filename}) → {result}")
        return result
    
    def get_filename_list(self, folder_name: str) -> List[str]:
        """Get list of files in a folder type."""
        result = folder_paths.get_filename_list(folder_name)
        logger.debug(f"{LOG_PREFIX}[FolderPathsProxy] get_filename_list({folder_name}) → {len(result)} files")
        return result
    
    def add_model_folder_path(self, folder_name: str, full_folder_path: str, is_default: bool = False) -> None:
        """Register a model folder path with ComfyUI."""
        logger.debug(f"{LOG_PREFIX}[FolderPathsProxy] add_model_folder_path({folder_name}, {full_folder_path}, {is_default})")
        return folder_paths.add_model_folder_path(folder_name, full_folder_path, is_default)
    
    # =========================================================================
    # Annotated Filepath Handling
    # =========================================================================
    
    def get_annotated_filepath(self, name: str, default_dir: Optional[str] = None) -> str:
        """Get annotated filepath."""
        result = folder_paths.get_annotated_filepath(name, default_dir)
        logger.debug(f"{LOG_PREFIX}[FolderPathsProxy] get_annotated_filepath({name}, {default_dir}) → {result}")
        return result
    
    def exists_annotated_filepath(self, name: str) -> bool:
        """Check if annotated filepath exists."""
        result = folder_paths.exists_annotated_filepath(name)
        logger.debug(f"{LOG_PREFIX}[FolderPathsProxy] exists_annotated_filepath({name}) → {result}")
        return result
    
    # =========================================================================
    # Image Saving (Rev 1.0 - Required for 10+ nodes)
    # =========================================================================
    
    def get_save_image_path(
        self, 
        filename_prefix: str, 
        output_dir: str, 
        image_width: int = 0, 
        image_height: int = 0
    ) -> Tuple[str, str, int, str, str]:
        """Get path for saving images with auto-incrementing counter.
        
        Returns:
            Tuple of (full_output_folder, filename, counter, subfolder, filename_prefix)
        """
        result = folder_paths.get_save_image_path(filename_prefix, output_dir, image_width, image_height)
        logger.debug(f"{LOG_PREFIX}[FolderPathsProxy] get_save_image_path({filename_prefix}, {output_dir}) → {result}")
        return result
    
    # =========================================================================
    # Constants/Configuration Access
    # =========================================================================
    
    @property
    def supported_pt_extensions(self) -> Set[str]:
        """Get set of supported PyTorch model extensions."""
        result = folder_paths.supported_pt_extensions
        logger.debug(f"{LOG_PREFIX}[FolderPathsProxy] supported_pt_extensions → {result}")
        return result
    
    @property
    def folder_names_and_paths(self) -> dict:
        """Get folder names and paths configuration.
        
        Note: This is read-only. Mutations in child process don't affect host.
        """
        result = folder_paths.folder_names_and_paths
        logger.debug(f"{LOG_PREFIX}[FolderPathsProxy] folder_names_and_paths → {len(result)} entries")
        return dict(result)  # Return copy to prevent mutation
    
    # =========================================================================
    # Input Subfolder Enumeration
    # =========================================================================
    
    def get_input_subfolders(self) -> List[str]:
        """Get list of subfolders in the input directory."""
        result = folder_paths.get_input_subfolders()
        logger.debug(f"{LOG_PREFIX}[FolderPathsProxy] get_input_subfolders() → {len(result)} folders")
        return result
    
    # =========================================================================
    # File Filtering Utilities
    # =========================================================================
    
    def filter_files_extensions(self, files: List[str], extensions: List[str]) -> List[str]:
        """Filter files by extensions."""
        result = folder_paths.filter_files_extensions(files, extensions)
        logger.debug(f"{LOG_PREFIX}[FolderPathsProxy] filter_files_extensions() → {len(result)} files")
        return result
    
    def filter_files_content_types(self, files: List[str], content_types: List[str]) -> List[str]:
        """Filter files by content types (image, video, audio, model)."""
        result = folder_paths.filter_files_content_types(files, content_types)
        logger.debug(f"{LOG_PREFIX}[FolderPathsProxy] filter_files_content_types() → {len(result)} files")
        return result
