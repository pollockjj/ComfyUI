"""Deduplicating logger wrapper for parallel attention.

Prevents log spam by counting consecutive duplicate messages and emitting
a summary when a different message arrives.

Example:
    >>> from comfy.parallel_attention.dedup_logger import get_dedup_logger
    >>> logger = get_dedup_logger(__name__)
    >>> for i in range(100):
    ...     logger.info("Same message")
    # Output: "Same message" (first time)
    # Output: "Same message (repeated 99 more times)" (on next different message)
"""

from __future__ import annotations

import logging
from typing import Optional


class DedupLogger:
    """Logger wrapper that deduplicates consecutive identical messages.
    
    When the same message is logged multiple times in a row, only the first
    occurrence is emitted. When a different message arrives, a summary of
    the previous message's repetition count is emitted first.
    
    Thread-safe for single-threaded use (typical ComfyUI pattern).
    For multi-rank FSDP2, each rank has its own logger instance.
    """
    
    def __init__(self, logger: logging.Logger):
        self._logger = logger
        self._last_message: Optional[str] = None
        self._last_level: Optional[int] = None
        self._repeat_count: int = 0
    
    def _flush_if_different(self, message: str, level: int) -> None:
        """Emit summary if previous message was repeated."""
        if self._last_message is not None and (
            message != self._last_message or level != self._last_level
        ):
            if self._repeat_count > 0:
                self._logger.log(
                    self._last_level,
                    "%s (repeated %d more time%s)",
                    self._last_message,
                    self._repeat_count,
                    "s" if self._repeat_count > 1 else "",
                )
            self._last_message = None
            self._repeat_count = 0
    
    def _log(self, level: int, message: str, *args, **kwargs) -> None:
        """Internal logging with deduplication."""
        formatted_message = message % args if args else message
        
        self._flush_if_different(formatted_message, level)
        
        if formatted_message == self._last_message and level == self._last_level:
            # Same message, just increment count
            self._repeat_count += 1
        else:
            # New message, emit it
            self._logger.log(level, message, *args, **kwargs)
            self._last_message = formatted_message
            self._last_level = level
            self._repeat_count = 0
    
    def debug(self, message: str, *args, **kwargs) -> None:
        """Log a debug message with deduplication."""
        self._log(logging.DEBUG, message, *args, **kwargs)
    
    def info(self, message: str, *args, **kwargs) -> None:
        """Log an info message with deduplication."""
        self._log(logging.INFO, message, *args, **kwargs)
    
    def warning(self, message: str, *args, **kwargs) -> None:
        """Log a warning message with deduplication."""
        self._log(logging.WARNING, message, *args, **kwargs)
    
    def error(self, message: str, *args, **kwargs) -> None:
        """Log an error message with deduplication."""
        self._log(logging.ERROR, message, *args, **kwargs)
    
    def flush(self) -> None:
        """Force flush any pending repeat count summary."""
        if self._last_message is not None and self._repeat_count > 0:
            self._logger.log(
                self._last_level,
                "%s (repeated %d more time%s)",
                self._last_message,
                self._repeat_count,
                "s" if self._repeat_count > 1 else "",
            )
            self._last_message = None
            self._repeat_count = 0


# Global cache of dedup loggers (one per module)
_DEDUP_LOGGERS: dict[str, DedupLogger] = {}


def get_dedup_logger(name: str) -> DedupLogger:
    """Get or create a deduplicating logger for the given module name.
    
    Args:
        name: Logger name (typically __name__ from calling module)
    
    Returns:
        DedupLogger instance that wraps the standard Python logger
    
    Example:
        >>> logger = get_dedup_logger(__name__)
        >>> logger.info("Message that might repeat")
    """
    if name not in _DEDUP_LOGGERS:
        standard_logger = logging.getLogger(name)
        _DEDUP_LOGGERS[name] = DedupLogger(standard_logger)
    return _DEDUP_LOGGERS[name]
