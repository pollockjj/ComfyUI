"""Unit test framework for parallel attention implementations.

Provides infrastructure for testing FSDP2-wrapped models, attention backends,
and distributed operations in real worker context.
"""

from .test_registry import register_test, run_tests

__all__ = ["register_test", "run_tests"]
