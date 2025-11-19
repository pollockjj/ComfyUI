"""Test registration and execution framework."""

import logging
import time
import traceback
from typing import Dict, Callable, Tuple, Any

logger = logging.getLogger(__name__)

# Global test registry
_TESTS: Dict[str, Callable] = {}


def register_test(name: str):
    """Decorator to register a test function.
    
    Test functions must have signature:
        func(fsdp_model, config: dict, rank: int, device) -> Tuple[str, str, float]
    
    Returns:
        (status, message, duration)
        status: "PASS", "FAIL", "SKIP", or "ERROR"
        message: Human-readable result description
        duration: Execution time in seconds
    
    Example:
        @register_test("test_my_feature")
        def test_my_feature(fsdp_model, config, rank, device):
            start = time.time()
            # ... test logic ...
            duration = time.time() - start
            return "PASS", "Feature works correctly", duration
    """
    def decorator(func):
        _TESTS[name] = func
        logger.debug(f"Registered test: {name}")
        return func
    return decorator


def run_tests(
    fsdp_model,
    config: dict,
    rank: int,
    device,
    selected_tests: list = None
) -> Dict[str, Dict[str, Any]]:
    """Execute registered tests and collect results.
    
    Args:
        fsdp_model: FSDP2-wrapped model with distributed attention
        config: Test configuration (backend names, degrees, etc.)
        rank: Distributed rank
        device: Torch device (cuda:0, cpu, etc.)
        selected_tests: Optional list of test names to run (runs all if None)
    
    Returns:
        Dict mapping test names to result dicts with keys:
            - status: "PASS", "FAIL", "SKIP", or "ERROR"
            - message: Result description
            - duration: Execution time in seconds
            - traceback: Stack trace (only present for ERROR status)
    """
    results = {}
    tests_to_run = _TESTS if selected_tests is None else {
        name: _TESTS[name] for name in selected_tests if name in _TESTS
    }
    
    if not tests_to_run:
        logger.warning("No tests to run!")
        return results
    
    logger.info(f"Running {len(tests_to_run)} tests on rank {rank}")
    
    for name, test_func in tests_to_run.items():
        logger.debug(f"Executing test: {name}")
        try:
            status, message, duration = test_func(fsdp_model, config, rank, device)
            results[name] = {
                "status": status,
                "message": message,
                "duration": duration
            }
            logger.info(f"Test {name}: {status} ({duration:.3f}s)")
        except Exception as e:
            results[name] = {
                "status": "ERROR",
                "message": str(e),
                "traceback": traceback.format_exc(),
                "duration": 0.0
            }
            logger.error(f"Test {name} raised exception: {e}")
    
    return results


def get_registered_tests():
    """Return list of registered test names."""
    return list(_TESTS.keys())
