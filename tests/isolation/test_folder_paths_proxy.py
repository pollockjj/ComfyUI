"""Unit tests for FolderPathsProxy."""

import pytest

from comfy.isolation.proxies.folder_paths_proxy import FolderPathsProxy
from tests.isolation.singleton_boundary_helpers import capture_sealed_singleton_imports


class TestFolderPathsProxy:
    """Test FolderPathsProxy methods."""

    @pytest.fixture
    def proxy(self):
        """Create a FolderPathsProxy instance for testing."""
        return FolderPathsProxy()

    def test_sealed_child_safe_uses_rpc_without_importing_folder_paths(self, monkeypatch):
        monkeypatch.setenv("PYISOLATE_CHILD", "1")
        monkeypatch.setenv("PYISOLATE_IMPORT_TORCH", "0")

        payload = capture_sealed_singleton_imports()

        assert payload["temp_dir"] == "/sandbox/temp"
        assert payload["models_dir"] == "/sandbox/models"
        assert "folder_paths" not in payload["modules"]
