"""
PyIsolate Unit Tests - Run at import time during ComfyUI startup.

These tests validate the route extraction and injection infrastructure
without requiring a running server.
"""
import logging

logger = logging.getLogger(__name__)

def run_startup_tests():
    """Execute all startup tests. Called from comfy.isolation.__init__."""
    from . import test_route_extraction
    
    results = []
    
    # Run route extraction tests
    results.extend(test_route_extraction.run_tests())
    
    # Summary
    passed = sum(1 for r in results if r["passed"])
    failed = sum(1 for r in results if not r["passed"])
    
    if failed == 0:
        logger.info(f"📚 [PyIsolate][Test] ✅ All {passed} startup tests passed")
    else:
        logger.error(f"📚 [PyIsolate][Test] ❌ {failed}/{passed + failed} startup tests FAILED")
        for r in results:
            if not r["passed"]:
                logger.error(f"📚 [PyIsolate][Test]   - {r['name']}: {r['error']}")
    
    return results
