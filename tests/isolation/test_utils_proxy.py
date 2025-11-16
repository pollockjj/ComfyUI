"""Unit tests for UtilsProxy."""

import pytest

from comfy.isolation.proxies.utils_proxy import UtilsProxy
from comfy.utils import ProgressBar


class TestUtilsProxy:
    """Test UtilsProxy methods."""

    @pytest.fixture
    def proxy(self):
        """Create a UtilsProxy instance for testing."""
        return UtilsProxy()

    def test_progress_bar_accessible(self, proxy):
        """Verify ProgressBar class is accessible via proxy."""
        assert hasattr(proxy, "ProgressBar"), "ProgressBar not found on proxy"
        assert proxy.ProgressBar is not None, "ProgressBar is None"

    def test_progress_bar_is_correct_class(self, proxy):
        """Verify ProgressBar is the actual comfy.utils.ProgressBar class."""
        assert proxy.ProgressBar == ProgressBar, "ProgressBar class mismatch"

    def test_progress_bar_is_instantiable(self, proxy):
        """Verify ProgressBar can be instantiated."""
        pbar = proxy.ProgressBar(total=100)
        assert pbar is not None, "ProgressBar instance is None"
        assert isinstance(pbar, ProgressBar), "ProgressBar instance type mismatch"

    def test_progress_bar_update_works(self, proxy):
        """Verify ProgressBar update method works."""
        pbar = proxy.ProgressBar(total=100)
        # Should not raise
        pbar.update(50)

    def test_progress_bar_update_absolute_works(self, proxy):
        """Verify ProgressBar update_absolute method works."""
        pbar = proxy.ProgressBar(total=100)
        # Should not raise
        pbar.update_absolute(75, 100)

    def test_multiple_progress_bars_can_coexist(self, proxy):
        """Verify multiple ProgressBar instances can be created."""
        pbar1 = proxy.ProgressBar(total=100)
        pbar2 = proxy.ProgressBar(total=200)
        
        assert pbar1 is not pbar2, "ProgressBar instances should be distinct"
        
        # Both should work independently
        pbar1.update(10)
        pbar2.update(20)
