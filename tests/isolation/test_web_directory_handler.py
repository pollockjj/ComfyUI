"""Tests for WebDirectoryProxy host-side cache and aiohttp handler integration."""

from __future__ import annotations

import base64
from unittest.mock import MagicMock

import pytest

from comfy.isolation.proxies.web_directory_proxy import (
    ALLOWED_EXTENSIONS,
    WebDirectoryCache,
)


@pytest.fixture()
def mock_proxy() -> MagicMock:
    """Create a mock WebDirectoryProxy RPC proxy."""
    proxy = MagicMock()
    proxy.list_web_files.return_value = [
        {"relative_path": "js/app.js", "content_type": "application/javascript"},
        {"relative_path": "js/utils.js", "content_type": "application/javascript"},
        {"relative_path": "index.html", "content_type": "text/html"},
        {"relative_path": "style.css", "content_type": "text/css"},
    ]
    proxy.get_web_file.return_value = {
        "content": base64.b64encode(b"console.log('hello');").decode("ascii"),
        "content_type": "application/javascript",
    }
    return proxy


@pytest.fixture()
def cache_with_proxy(mock_proxy: MagicMock) -> WebDirectoryCache:
    """Create a WebDirectoryCache with a registered mock proxy."""
    cache = WebDirectoryCache()
    cache.register_proxy("test-extension", mock_proxy)
    return cache


class TestExtensionsListing:
    """AC-2: /extensions endpoint lists proxied JS files."""

    def test_extensions_listing_includes_js_files(
        self, cache_with_proxy: WebDirectoryCache
    ) -> None:
        files = cache_with_proxy.list_files("test-extension")
        js_files = [f for f in files if f["relative_path"].endswith(".js")]
        assert len(js_files) >= 1
        assert any(
            f["relative_path"] == "js/app.js" for f in js_files
        )

    def test_extensions_listing_url_format(
        self, cache_with_proxy: WebDirectoryCache
    ) -> None:
        files = cache_with_proxy.list_files("test-extension")
        for f in files:
            # All paths should be relative (no leading /)
            assert not f["relative_path"].startswith("/")
            # Paths should use forward slashes
            assert "\\" not in f["relative_path"]

    def test_extensions_listing_unknown_extension(
        self, cache_with_proxy: WebDirectoryCache
    ) -> None:
        files = cache_with_proxy.list_files("nonexistent")
        assert files == []


class TestCacheHit:
    """AC-3: Cache populated on first request, reused on second."""

    def test_cache_hit_single_rpc_call(
        self, cache_with_proxy: WebDirectoryCache, mock_proxy: MagicMock
    ) -> None:
        # First call — RPC
        result1 = cache_with_proxy.get_file("test-extension", "js/app.js")
        assert result1 is not None
        assert result1["content"] == b"console.log('hello');"

        # Second call — cache hit
        result2 = cache_with_proxy.get_file("test-extension", "js/app.js")
        assert result2 is not None
        assert result2["content"] == b"console.log('hello');"

        # Proxy was called exactly once
        assert mock_proxy.get_web_file.call_count == 1

    def test_cache_returns_none_for_unknown_extension(
        self, cache_with_proxy: WebDirectoryCache
    ) -> None:
        result = cache_with_proxy.get_file("nonexistent", "js/app.js")
        assert result is None


class TestForbiddenType:
    """AC-4: Disallowed file types are rejected."""

    @pytest.mark.parametrize(
        "disallowed_path",
        [
            "backdoor.py",
            "malware.exe",
            "exploit.sh",
        ],
    )
    def test_forbidden_file_type_not_in_allowlist(
        self, disallowed_path: str
    ) -> None:
        import os
        suffix = os.path.splitext(disallowed_path)[1].lower()
        assert suffix not in ALLOWED_EXTENSIONS
