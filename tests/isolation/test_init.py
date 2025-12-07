"""Unit tests for PyIsolate isolation system initialization."""

import logging
import pytest


def test_log_prefix():
    """Verify LOG_PREFIX constant is correctly defined."""
    from comfy.isolation import LOG_PREFIX
    assert LOG_PREFIX == "]["
    assert isinstance(LOG_PREFIX, str)


def test_module_initialization():
    """Verify module initializes without errors."""
    import comfy.isolation
    assert hasattr(comfy.isolation, 'LOG_PREFIX')
    assert hasattr(comfy.isolation, 'initialize_proxies')


class TestInitializeProxies:
    def test_initialize_proxies_runs_without_error(self):
        from comfy.isolation import initialize_proxies
        initialize_proxies()

    def test_initialize_proxies_registers_folder_paths_proxy(self):
        from comfy.isolation import initialize_proxies
        from comfy.isolation.proxies.folder_paths_proxy import FolderPathsProxy
        initialize_proxies()
        proxy = FolderPathsProxy()
        assert proxy is not None
        assert hasattr(proxy, "get_temp_directory")

    def test_initialize_proxies_registers_model_management_proxy(self):
        from comfy.isolation import initialize_proxies
        from comfy.isolation.proxies.model_management_proxy import ModelManagementProxy
        initialize_proxies()
        proxy = ModelManagementProxy()
        assert proxy is not None
        assert hasattr(proxy, "get_torch_device")

    def test_initialize_proxies_can_be_called_multiple_times(self):
        from comfy.isolation import initialize_proxies
        initialize_proxies()
        initialize_proxies()
        initialize_proxies()

    def test_dev_proxies_accessible_when_dev_mode(self, monkeypatch):
        """Verify dev proxies load when PYISOLATE_DEV=1."""
        import os
        monkeypatch.setenv("PYISOLATE_DEV", "1")
        from comfy.isolation import initialize_proxies
        from comfy.isolation.development.proxies.nodes_proxy import NodesProxy
        from comfy.isolation.development.proxies.utils_proxy import UtilsProxy
        initialize_proxies()
        nodes_proxy = NodesProxy()
        utils_proxy = UtilsProxy()
        assert nodes_proxy is not None
        assert utils_proxy is not None
