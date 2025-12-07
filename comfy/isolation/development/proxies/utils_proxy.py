"""ProxiedSingleton for comfy.utils module."""

from comfy.utils import ProgressBar


class UtilsProxy:
    """Proxy for comfy.utils module."""
    
    ProgressBar = ProgressBar


def run_tests():
    """Run self-tests for UtilsProxy."""
    proxy = UtilsProxy()
    passed = 0
    failed = 0
    
    try:
        assert proxy.ProgressBar is ProgressBar
        passed += 1
    except AssertionError:
        failed += 1
    
    try:
        pb = proxy.ProgressBar(10)
        pb.update_absolute(5)
        passed += 1
    except Exception:
        failed += 1
    
    return passed, failed
