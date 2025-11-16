"""Unit tests for PyIsolate isolation system initialization."""

import logging
import pytest


def test_log_prefix():
    """Verify LOG_PREFIX constant is correctly defined."""
    from comfy.isolation import LOG_PREFIX
    assert LOG_PREFIX == "📚 [PyIsolate]"
    assert isinstance(LOG_PREFIX, str)


def test_get_isolation_logger():
    """Verify get_isolation_logger returns valid logger."""
    from comfy.isolation import get_isolation_logger
    
    logger = get_isolation_logger("test_module")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "test_module"


def test_logger_can_log():
    """Verify logger can emit messages without errors."""
    from comfy.isolation import get_isolation_logger, LOG_PREFIX
    
    logger = get_isolation_logger(__name__)
    
    # Should not raise any exceptions
    logger.debug(f"{LOG_PREFIX}[Test] debug message")
    logger.info(f"{LOG_PREFIX}[Test] info message")
    logger.warning(f"{LOG_PREFIX}[Test] warning message")


def test_module_initialization():
    """Verify module initializes without errors."""
    import comfy.isolation
    
    # Module should have expected exports
    assert hasattr(comfy.isolation, 'LOG_PREFIX')
    assert hasattr(comfy.isolation, 'get_isolation_logger')
    assert hasattr(comfy.isolation, 'logger')


class TestInitializeProxies:
    """Test initialize_proxies() function."""

    def test_initialize_proxies_runs_without_error(self):
        """Verify initialize_proxies() executes without raising exceptions."""
        from comfy.isolation import initialize_proxies
        # Should not raise
        initialize_proxies()

    def test_initialize_proxies_registers_folder_paths_proxy(self):
        """Verify FolderPathsProxy is instantiated after init."""
        from comfy.isolation import initialize_proxies
        from comfy.isolation.proxies.folder_paths_proxy import FolderPathsProxy
        
        initialize_proxies()
        
        # Create another instance—should work since these are simple wrappers
        proxy = FolderPathsProxy()
        assert proxy is not None
        assert hasattr(proxy, "get_temp_directory")

    def test_initialize_proxies_registers_model_management_proxy(self):
        """Verify ModelManagementProxy is instantiated after init."""
        from comfy.isolation import initialize_proxies
        from comfy.isolation.proxies.model_management_proxy import ModelManagementProxy
        
        initialize_proxies()
        
        proxy = ModelManagementProxy()
        assert proxy is not None
        assert hasattr(proxy, "get_torch_device")

    def test_initialize_proxies_registers_nodes_proxy(self):
        """Verify NodesProxy is instantiated after init."""
        from comfy.isolation import initialize_proxies
        from comfy.isolation.proxies.nodes_proxy import NodesProxy
        
        initialize_proxies()
        
        proxy = NodesProxy()
        assert proxy is not None
        assert hasattr(proxy, "PreviewImage")
        assert hasattr(proxy, "SaveImage")

    def test_initialize_proxies_can_be_called_multiple_times(self):
        """Verify initialize_proxies() is idempotent."""
        from comfy.isolation import initialize_proxies
        
        # Should not raise even if called multiple times
        initialize_proxies()
        initialize_proxies()
        initialize_proxies()

    def test_all_proxies_accessible_after_init(self):
        """Verify all 4 proxy classes are accessible after initialization."""
        from comfy.isolation import initialize_proxies
        from comfy.isolation.proxies.folder_paths_proxy import FolderPathsProxy
        from comfy.isolation.proxies.model_management_proxy import ModelManagementProxy
        from comfy.isolation.proxies.nodes_proxy import NodesProxy
        from comfy.isolation.proxies.utils_proxy import UtilsProxy
        
        initialize_proxies()
        
        # All proxies should be instantiable
        folder_proxy = FolderPathsProxy()
        model_proxy = ModelManagementProxy()
        nodes_proxy = NodesProxy()
        utils_proxy = UtilsProxy()
        
        assert folder_proxy is not None
        assert model_proxy is not None
        assert nodes_proxy is not None
        assert utils_proxy is not None
        
        # Verify key methods/attributes exist
        assert hasattr(folder_proxy, "get_temp_directory")
        assert hasattr(model_proxy, "get_torch_device")
        assert hasattr(nodes_proxy, "PreviewImage")
        assert hasattr(utils_proxy, "ProgressBar")

