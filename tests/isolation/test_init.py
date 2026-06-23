"""Unit tests for PyIsolate isolation system initialization."""

import sys

from tests.isolation.singleton_boundary_helpers import (
    FakeSingletonRPC,
    reset_forbidden_singleton_modules,
)


class TestInitializeProxies:
    def test_initialize_proxies_runs_without_error(self):
        from comfy.isolation import initialize_proxies
        initialize_proxies()

    def test_dev_proxies_accessible_when_dev_mode(self, monkeypatch):
        """Verify dev mode does not break core proxy initialization."""
        monkeypatch.setenv("PYISOLATE_DEV", "1")
        from comfy.isolation import initialize_proxies
        from comfy.isolation.proxies.folder_paths_proxy import FolderPathsProxy
        from comfy.isolation.proxies.utils_proxy import UtilsProxy
        initialize_proxies()
        folder_proxy = FolderPathsProxy()
        utils_proxy = UtilsProxy()
        assert folder_proxy is not None
        assert utils_proxy is not None

    def test_sealed_child_safe_initialize_proxies_avoids_real_utils_import(self, monkeypatch):
        monkeypatch.setenv("PYISOLATE_CHILD", "1")
        monkeypatch.setenv("PYISOLATE_IMPORT_TORCH", "0")
        reset_forbidden_singleton_modules()

        from pyisolate._internal import rpc_protocol
        from comfy.isolation import initialize_proxies

        fake_rpc = FakeSingletonRPC()
        monkeypatch.setattr(rpc_protocol, "get_child_rpc_instance", lambda: fake_rpc)

        initialize_proxies()

        assert "comfy.utils" not in sys.modules
        assert "folder_paths" not in sys.modules
        assert "comfy_execution.progress" not in sys.modules
