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
    
    V1.0: Only core proxies (folder_paths, model_management)
    Advanced proxies (CLIP, ModelPatcher, PromptServer, etc.) require PYISOLATE_DEV=1
    """
    # CRITICAL: Clear any default logging handlers that might have been created
    # by imported modules (like pyisolate) before ComfyUI's setup_logger runs.
    # This prevents duplicate logs with "INFO:root:" prefix.
    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers[:]:
            root.removeHandler(handler)
    
    # Add a NullHandler to prevent Python from automatically adding a default
    # handler (StreamHandler) if logging occurs before setup_logger is called.
    # This ensures we don't get "INFO:root:" logs if something logs early.
    root.addHandler(logging.NullHandler())
    
    import os
    IS_DEV = os.environ.get("PYISOLATE_DEV") == "1"
    
    # V1.0 Production Proxies (always enabled)
    from .proxies.folder_paths_proxy import FolderPathsProxy
    from .proxies.model_management_proxy import ModelManagementProxy
    
    FolderPathsProxy()
    ModelManagementProxy()
    
    logger.debug("Host process V1.0 proxies registered (folder_paths, model_management)")
    
    # Experimental Proxies (PYISOLATE_DEV=1 only)
    if IS_DEV:
        from .development.proxies.nodes_proxy import NodesProxy
        from .development.proxies.utils_proxy import UtilsProxy
        from .development.proxies.prompt_server_proxy import PromptServerProxy
        from .development.clip_proxy import CLIPRegistry
        from .development.model_patcher_proxy import ModelPatcherRegistry
        
        NodesProxy()
        UtilsProxy()
        PromptServerProxy()
        CLIPRegistry()
        ModelPatcherRegistry()
        
        logger.debug("Host process experimental proxies registered (IS_DEV=1)")
