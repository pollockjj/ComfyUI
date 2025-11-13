"""ProxiedSingleton for nodes module base classes.

Exposes base classes like PreviewImage and SaveImage that custom nodes inherit from.
This is necessary for Crystools, which subclasses these.
"""

import logging
import nodes
from comfy.isolation import LOG_PREFIX

logger = logging.getLogger(__name__)

class NodesProxy:
    """Proxy for nodes base classes (Crystools subset).
    
    This is NOT a ProxiedSingleton yet - it's a simple wrapper for testing.
    It just forwards class definitions, no RPC needed.
    """
    
    # Expose base classes directly
    PreviewImage = nodes.PreviewImage
    SaveImage = nodes.SaveImage
    
    def __init__(self):
        super().__init__()
        logger.debug(f"{LOG_PREFIX}[NodesProxy] Initialized with PreviewImage, SaveImage")

def run_tests():
    """Run self-tests for NodesProxy (called explicitly after ComfyUI init)."""
    proxy = NodesProxy()
    passed = 0
    failed = 0
    
    # Test 1: PreviewImage is correct class
    try:
        assert proxy.PreviewImage == nodes.PreviewImage, "PreviewImage class mismatch"
        logger.info(f"{LOG_PREFIX}[Test] ✅ NodesProxy.PreviewImage exposed correctly")
        passed += 1
    except Exception as e:
        logger.error(f"{LOG_PREFIX}[Test] ❌ NodesProxy.PreviewImage failed: {e}")
        failed += 1
        
    # Test 2: SaveImage is correct class
    try:
        assert proxy.SaveImage == nodes.SaveImage, "SaveImage class mismatch"
        logger.info(f"{LOG_PREFIX}[Test] ✅ NodesProxy.SaveImage exposed correctly")
        passed += 1
    except Exception as e:
        logger.error(f"{LOG_PREFIX}[Test] ❌ NodesProxy.SaveImage failed: {e}")
        failed += 1
        
    total = passed + failed
    logger.info(f"{LOG_PREFIX}[Test] NodesProxy: {passed}/{total} tests passed")
    
    if failed > 0:
        raise RuntimeError(f"NodesProxy self-tests failed: {failed}/{total}")

# Run tests on module import
run_tests()
