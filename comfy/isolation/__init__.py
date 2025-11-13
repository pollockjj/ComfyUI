"""PyIsolate isolation system for ComfyUI custom nodes.

Provides process isolation for custom_nodes via PyIsolate, enabling:
- Dependency conflict resolution (isolated venvs)
- Security sandboxing
- Zero-copy tensor sharing (share_torch=True)
- ProxiedSingleton for shared ComfyUI services
"""

import logging

LOG_PREFIX = "🟢 [PyIsolate]"

def get_isolation_logger(name: str) -> logging.Logger:
    """Get logger with PyIsolate prefix for consistent log formatting.
    
    Args:
        name: Module name (typically __name__)
        
    Returns:
        Logger instance that can be used with standard logging methods
        
    Example:
        >>> logger = get_isolation_logger(__name__)
        >>> logger.info(f"{LOG_PREFIX}[Component] message")
    """
    return logging.getLogger(name)

# Module-level logger
logger = get_isolation_logger(__name__)

# Test different emojis for visibility
test_emojis = [
    "⚡",  # Lightning (parallel-attention uses this - BRIGHT)
    "🟢",  # Green circle
    "🔵",  # Blue circle
    "🟡",  # Yellow circle
    "🔴",  # Red circle
    "💚",  # Green heart
    "✅",  # Check mark
    "🚀",  # Rocket
    "🎯",  # Target
    "⭐",  # Star
]

logger.info("=== PyIsolate Emoji Visibility Test ===")
for i, emoji in enumerate(test_emojis, 1):
    logger.info(f"{emoji} [PyIsolate][Test] Emoji #{i} visibility check")
logger.info("=== End Emoji Test - Which one is most visible? ===")

# Announce system initialization
logger.info(f"{LOG_PREFIX}[System] Isolation system initialized")

__all__ = [
    'LOG_PREFIX',
    'get_isolation_logger',
]
