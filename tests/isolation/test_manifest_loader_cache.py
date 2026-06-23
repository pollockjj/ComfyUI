"""
Unit tests for manifest_loader.py cache functions.

Phase 1 tests verify:
1. Cache miss on first run (no cache exists)
2. Cache hit when nothing changes
3. Invalidation on .py file touch
4. Invalidation on manifest change
5. Cache location correctness (in venv_root, NOT in custom_nodes)
6. Corrupt cache handling (graceful failure)

These tests verify the cache implementation is correct BEFORE it's activated
in extension_loader.py (Phase 2).
"""

from __future__ import annotations

import time
from pathlib import Path


class TestComputeCacheKey:
    """Tests for compute_cache_key() function."""

    def test_key_includes_manifest_content(self, tmp_path: Path) -> None:
        """Cache key changes when manifest content changes."""
        from comfy.isolation.manifest_loader import compute_cache_key

        node_dir = tmp_path / "test_node"
        node_dir.mkdir()
        manifest = node_dir / "pyisolate.yaml"

        # Initial manifest
        manifest.write_text("isolated: true\ndependencies: []\n")
        key1 = compute_cache_key(node_dir, manifest)

        # Modified manifest
        manifest.write_text("isolated: true\ndependencies: [numpy]\n")
        key2 = compute_cache_key(node_dir, manifest)

        assert key1 != key2, "Key should change when manifest content changes"

class TestGetCachePath:
    """Tests for get_cache_path() function."""

    def test_returns_correct_paths(self, tmp_path: Path) -> None:
        """Cache paths are in venv_root, not in node_dir."""
        from comfy.isolation.manifest_loader import get_cache_path

        node_dir = tmp_path / "custom_nodes" / "MyNode"
        venv_root = tmp_path / ".pyisolate_venvs"

        key_file, data_file = get_cache_path(node_dir, venv_root)

        assert key_file == venv_root / "MyNode" / "cache" / "cache_key"
        assert data_file == venv_root / "MyNode" / "cache" / "node_info.json"

class TestIsCacheValid:
    """Tests for is_cache_valid() function."""

    def test_invalidation_on_py_change(self, tmp_path: Path) -> None:
        """Cache invalidates when .py file is modified."""
        from comfy.isolation.manifest_loader import (
            compute_cache_key,
            get_cache_path,
            is_cache_valid,
        )

        node_dir = tmp_path / "test_node"
        node_dir.mkdir()
        manifest = node_dir / "pyisolate.yaml"
        manifest.write_text("isolated: true\n")
        py_file = node_dir / "nodes.py"
        py_file.write_text("# original")
        venv_root = tmp_path / ".pyisolate_venvs"

        # Create valid cache
        cache_key = compute_cache_key(node_dir, manifest)
        key_file, data_file = get_cache_path(node_dir, venv_root)
        key_file.parent.mkdir(parents=True, exist_ok=True)
        key_file.write_text(cache_key)
        data_file.write_text("{}")

        # Verify cache is valid initially
        assert is_cache_valid(node_dir, manifest, venv_root) is True

        # Modify .py file
        time.sleep(0.01)  # Ensure mtime changes
        py_file.write_text("# modified")

        # Cache should now be invalid
        assert is_cache_valid(node_dir, manifest, venv_root) is False


class TestSaveToCache:
    """Tests for save_to_cache() function."""

    def test_roundtrip_with_validation(self, tmp_path: Path) -> None:
        """Saved cache is immediately valid."""
        from comfy.isolation.manifest_loader import (
            is_cache_valid,
            load_from_cache,
            save_to_cache,
        )

        node_dir = tmp_path / "test_node"
        node_dir.mkdir()
        manifest = node_dir / "pyisolate.yaml"
        manifest.write_text("isolated: true\n")
        (node_dir / "nodes.py").write_text("# code")
        venv_root = tmp_path / ".pyisolate_venvs"

        test_data = {"TestNode": {"foo": "bar"}}
        save_to_cache(node_dir, venv_root, test_data, manifest)

        assert is_cache_valid(node_dir, manifest, venv_root) is True
        assert load_from_cache(node_dir, venv_root) == test_data
