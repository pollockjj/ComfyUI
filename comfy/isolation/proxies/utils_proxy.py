"""ProxiedSingleton for comfy.utils module (Florence2 needs ProgressBar)."""

import logging
from comfy.utils import ProgressBar
from comfy.isolation import LOG_PREFIX

logger = logging.getLogger(__name__)

class UtilsProxy:
    """Proxy for comfy.utils module providing utility classes for isolated nodes.
    
    This is NOT a ProxiedSingleton yet - it's a simple wrapper for testing.
    Florence2 needs: ProgressBar
    """
    
    # Expose ProgressBar class directly (no RPC per update, just class forwarding)
    ProgressBar = ProgressBar
    
    def __init__(self):
        super().__init__()
        logger.debug(f"{LOG_PREFIX}[UtilsProxy] Initialized with ProgressBar")


def run_tests():
    """Run self-tests for UtilsProxy (called explicitly after ComfyUI init)."""
    proxy = UtilsProxy()
    passed = 0
    failed = 0
    
    # Test 1: ProgressBar is correct class
    try:
        assert proxy.ProgressBar == ProgressBar, "ProgressBar class mismatch"
        logger.info(f"{LOG_PREFIX}[Test] ✅ UtilsProxy.ProgressBar exposed correctly")
        passed += 1
    except Exception as e:
        logger.error(f"{LOG_PREFIX}[Test] ❌ UtilsProxy.ProgressBar failed: {e}")
        failed += 1
    
    # Test 2: ProgressBar can be instantiated
    try:
        pbar = proxy.ProgressBar(total=100)
        assert pbar is not None, "ProgressBar instance is None"
        logger.info(f"{LOG_PREFIX}[Test] ✅ UtilsProxy.ProgressBar instantiable")
        passed += 1
    except Exception as e:
        logger.error(f"{LOG_PREFIX}[Test] ❌ UtilsProxy.ProgressBar instantiation failed: {e}")
        failed += 1
    
    # Summary
    total = passed + failed
    logger.info(f"{LOG_PREFIX}[Test] UtilsProxy: {passed}/{total} tests passed")
    
    if failed > 0:
        raise RuntimeError(f"UtilsProxy self-tests failed: {failed}/{total}")
