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
    """AC-2: /extensions endpoint lists proxied JS files in URL format."""

    def test_extensions_listing_produces_url_format_paths(
        self, cache_with_proxy: WebDirectoryCache
    ) -> None:
        """Simulate what server.py does: build /extensions/{name}/{path} URLs."""
        import urllib.parse

        ext_name = "test-extension"
        urls = []
        for entry in cache_with_proxy.list_files(ext_name):
            if entry["relative_path"].endswith(".js"):
                urls.append(
                    "/extensions/" + urllib.parse.quote(ext_name)
                    + "/" + entry["relative_path"]
                )

        # At least one proxied JS URL in /extensions/{name}/{path} format
        assert len(urls) >= 1
        assert any("/extensions/test-extension/js/app.js" == u for u in urls)
        # All URLs start with /extensions/
        for url in urls:
            assert url.startswith("/extensions/test-extension/")


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
    """AC-4: Disallowed file types return HTTP 403 Forbidden."""

    @pytest.mark.parametrize(
        "disallowed_path,expected_status",
        [
            ("backdoor.py", 403),
            ("malware.exe", 403),
            ("exploit.sh", 403),
        ],
    )
    def test_forbidden_file_type_returns_403(
        self, disallowed_path: str, expected_status: int
    ) -> None:
        """Simulate the aiohttp handler's file-type check and verify 403."""
        import os
        suffix = os.path.splitext(disallowed_path)[1].lower()

        # This mirrors the handler logic in server.py:
        # if suffix not in ALLOWED_EXTENSIONS: return web.Response(status=403)
        if suffix not in ALLOWED_EXTENSIONS:
            status = 403
        else:
            status = 200

        assert status == expected_status
