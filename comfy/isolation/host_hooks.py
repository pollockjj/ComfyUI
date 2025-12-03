"""Host process initialization hooks.

This module handles initialization when running as the main ComfyUI host process
(not a PyIsolate child). All host-specific proxy registration is consolidated here.
"""
import logging

logger = logging.getLogger(__name__)


def initialize_host_process() -> None:
    """Initialize host process proxies.
    
    Called from initialize_proxies() when not in a child process.
    Registers all ProxiedSingletons so they're available to isolated nodes via RPC.
    """
    from .proxies.folder_paths_proxy import FolderPathsProxy
    from .proxies.model_management_proxy import ModelManagementProxy
    from .proxies.nodes_proxy import NodesProxy
    from .proxies.utils_proxy import UtilsProxy
    from .proxies.prompt_server_proxy import PromptServerProxy
    from .clip_proxy import CLIPRegistry
    
    # Instantiate singletons to register them with PyIsolate's RPC system
    FolderPathsProxy()
    ModelManagementProxy()
    NodesProxy()
    UtilsProxy()
    PromptServerProxy()
    CLIPRegistry()
    
    logger.debug("Host process proxies registered")
