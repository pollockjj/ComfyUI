"""ProxiedSingleton for folder_paths module (Crystools subset)."""

import logging
import folder_paths
from comfy.isolation import LOG_PREFIX

logger = logging.getLogger(__name__)

class FolderPathsProxy:
    """Proxy for folder_paths module providing path resolution for isolated nodes.
    
    This is NOT a ProxiedSingleton yet - it's a simple wrapper for testing.
    Crystools needs: get_temp_directory, get_input_directory, 
                     get_annotated_filepath, exists_annotated_filepath
    """
    
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
    
    def get_annotated_filepath(self, name: str) -> str:
        """Get annotated filepath."""
        result = folder_paths.get_annotated_filepath(name)
        logger.debug(f"{LOG_PREFIX}[FolderPathsProxy] get_annotated_filepath({name}) → {result}")
        return result
    
    def exists_annotated_filepath(self, name: str) -> bool:
        """Check if annotated filepath exists."""
        result = folder_paths.exists_annotated_filepath(name)
        logger.debug(f"{LOG_PREFIX}[FolderPathsProxy] exists_annotated_filepath({name}) → {result}")
        return result
    
    @property
    def models_dir(self) -> str:
        """Get base models directory path."""
        result = folder_paths.models_dir
        logger.debug(f"{LOG_PREFIX}[FolderPathsProxy] models_dir → {result}")
        return result
    
    def add_model_folder_path(self, folder_name: str, full_folder_path: str, is_default: bool = False) -> None:
        """Register a model folder path with ComfyUI."""
        logger.debug(f"{LOG_PREFIX}[FolderPathsProxy] add_model_folder_path({folder_name}, {full_folder_path}, {is_default})")
        return folder_paths.add_model_folder_path(folder_name, full_folder_path, is_default)
    
    def get_folder_paths(self, folder_name: str) -> list:
        """Get all registered paths for a folder type."""
        result = folder_paths.get_folder_paths(folder_name)
        logger.debug(f"{LOG_PREFIX}[FolderPathsProxy] get_folder_paths({folder_name}) → {len(result)} paths")
        return result
