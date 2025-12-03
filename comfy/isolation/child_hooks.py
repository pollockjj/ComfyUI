"""Child process initialization hooks.

This module handles initialization when running inside an isolated child process
(PYISOLATE_CHILD=1). All child-specific logic that was previously scattered in
core files is consolidated here.
"""
import logging
import os

logger = logging.getLogger(__name__)


def is_child_process() -> bool:
    """Check if current process is a PyIsolate child."""
    return os.environ.get("PYISOLATE_CHILD") == "1"


def initialize_child_process() -> None:
    """Initialize child process environment.
    
    Called from initialize_proxies() when PYISOLATE_CHILD=1.
    Sets up proxies for accessing host services via RPC.
    """
    _setup_prompt_server_proxy()
    _setup_logging()
    logger.debug("Child process initialization complete")


def _setup_prompt_server_proxy() -> None:
    """Replace PromptServer.instance with RPC proxy in child.
    
    This allows isolated nodes to use PromptServer.instance.send_sync()
    and other server methods via RPC to the host process.
    """
    try:
        import server
        from .proxies.prompt_server_proxy import PromptServerProxy
        
        # Create proxy instance that will communicate via RPC
        proxy = PromptServerProxy()
        
        # Replace the class-level instance attribute
        # Nodes that import server will see the proxy
        server.PromptServer.instance = proxy
        
        logger.debug("PromptServer.instance replaced with proxy")
    except ImportError:
        # server module not available in child - that's OK
        pass
    except Exception as e:
        logger.debug("Could not set up PromptServer proxy: %s", e)


def _setup_logging() -> None:
    """Configure logging for child process.
    
    PyIsolate's client.py already configures basic logging.
    This is a hook for any additional child-specific logging setup.
    """
    # PyIsolate client.py handles this, nothing additional needed
    pass
