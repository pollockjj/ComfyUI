"""PyIsolate isolation system for ComfyUI custom nodes.

Provides process isolation for custom_nodes via PyIsolate, enabling:
- Dependency conflict resolution (isolated venvs)
- Security sandboxing
- Zero-copy tensor sharing (share_torch=True)
- ProxiedSingleton for shared ComfyUI services
"""

import logging

LOG_PREFIX = "� [PyIsolate]"

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

# Announce system initialization
logger.info(f"{LOG_PREFIX}[System] Isolation system initialized")

def initialize_proxies():
    """Initialize and test ProxiedSingletons (called after ComfyUI torch init)."""
    logger.info(f"{LOG_PREFIX}[System] Initializing ProxiedSingletons...")
    
    # Import and test proxies
    from comfy.isolation.proxies import folder_paths_proxy
    folder_paths_proxy.run_tests()
    
    from comfy.isolation.proxies import model_management_proxy
    model_management_proxy.run_tests()
    
    from comfy.isolation.proxies import nodes_proxy
    nodes_proxy.run_tests()
    
    logger.info(f"{LOG_PREFIX}[System] ProxiedSingletons initialized")


__all__ = [
    'LOG_PREFIX',
    'get_isolation_logger',
    'initialize_proxies',
]
