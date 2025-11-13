"""Unit tests for PyIsolate isolation system initialization."""

import logging
import pytest


def test_log_prefix():
    """Verify LOG_PREFIX constant is correctly defined."""
    from comfy.isolation import LOG_PREFIX
    assert LOG_PREFIX == "🔒 [PyIsolate]"
    assert isinstance(LOG_PREFIX, str)


def test_get_isolation_logger():
    """Verify get_isolation_logger returns valid logger."""
    from comfy.isolation import get_isolation_logger
    
    logger = get_isolation_logger("test_module")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "test_module"


def test_logger_can_log():
    """Verify logger can emit messages without errors."""
    from comfy.isolation import get_isolation_logger, LOG_PREFIX
    
    logger = get_isolation_logger(__name__)
    
    # Should not raise any exceptions
    logger.debug(f"{LOG_PREFIX}[Test] debug message")
    logger.info(f"{LOG_PREFIX}[Test] info message")
    logger.warning(f"{LOG_PREFIX}[Test] warning message")


def test_module_initialization():
    """Verify module initializes without errors."""
    import comfy.isolation
    
    # Module should have expected exports
    assert hasattr(comfy.isolation, 'LOG_PREFIX')
    assert hasattr(comfy.isolation, 'get_isolation_logger')
    assert hasattr(comfy.isolation, 'logger')
