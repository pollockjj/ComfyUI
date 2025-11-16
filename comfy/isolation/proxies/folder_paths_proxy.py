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


def run_tests():
    """Run self-tests for FolderPathsProxy (called explicitly after ComfyUI init)."""
    proxy = FolderPathsProxy()
    passed = 0
    failed = 0
    
    # Test 1: get_temp_directory returns string
    try:
        temp_dir = proxy.get_temp_directory()
        assert isinstance(temp_dir, str), f"Expected str, got {type(temp_dir)}"
        assert len(temp_dir) > 0, "Temp directory path is empty"
        logger.info(f"{LOG_PREFIX}[Test] ✅ FolderPathsProxy.get_temp_directory() → {temp_dir}")
        passed += 1
    except Exception as e:
        logger.error(f"{LOG_PREFIX}[Test] ❌ FolderPathsProxy.get_temp_directory() failed: {e}")
        failed += 1
    
    # Test 2: get_input_directory returns string
    try:
        input_dir = proxy.get_input_directory()
        assert isinstance(input_dir, str), f"Expected str, got {type(input_dir)}"
        assert len(input_dir) > 0, "Input directory path is empty"
        logger.info(f"{LOG_PREFIX}[Test] ✅ FolderPathsProxy.get_input_directory() → {input_dir}")
        passed += 1
    except Exception as e:
        logger.error(f"{LOG_PREFIX}[Test] ❌ FolderPathsProxy.get_input_directory() failed: {e}")
        failed += 1
    
    # Test 3: exists_annotated_filepath returns bool
    try:
        exists = proxy.exists_annotated_filepath("nonexistent_file.png")
        assert isinstance(exists, bool), f"Expected bool, got {type(exists)}"
        logger.info(f"{LOG_PREFIX}[Test] ✅ FolderPathsProxy.exists_annotated_filepath() → {exists}")
        passed += 1
    except Exception as e:
        logger.error(f"{LOG_PREFIX}[Test] ❌ FolderPathsProxy.exists_annotated_filepath() failed: {e}")
        failed += 1
    
    # Summary
    total = passed + failed
    logger.info(f"{LOG_PREFIX}[Test] FolderPathsProxy: {passed}/{total} tests passed")
    
    if failed > 0:
        raise RuntimeError(f"FolderPathsProxy self-tests failed: {failed}/{total}")
