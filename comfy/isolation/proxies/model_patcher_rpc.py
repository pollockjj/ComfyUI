"""
comfy.isolation.proxies.model_patcher_proxy
ProxiedSingleton for MODEL RPC operations.
"""

import logging
from typing import Any, Tuple, Dict

logger = logging.getLogger(__name__)

try:
    from pyisolate import ProxiedSingleton
except ImportError:
    # Fallback for when PyIsolate isn't available
    class ProxiedSingleton:
        pass


class ModelPatcherRPC(ProxiedSingleton):
    """RPC endpoint for ModelPatcher operations.
    
    This ProxiedSingleton allows isolated nodes to execute ModelPatcher methods
    on host-side objects via RPC.
    """
    
    def __init__(self):
        super().__init__()
        logger.info("📚 [ModelPatcherRPC] Initialized RPC endpoint")
    
    async def execute_method(
        self,
        model_id: str,
        method_name: str,
        args: Tuple,
        kwargs: Dict
    ) -> Any:
        """Execute a ModelPatcher method on behalf of isolated proxy.
        
        This is the RPC endpoint that isolated nodes call when they invoke
        methods on ModelPatcherProxy instances.
        
        Args:
            model_id: UUID referencing ModelPatcher in registry
            method_name: Method to execute
            args: Positional arguments
            kwargs: Keyword arguments
            
        Returns:
            Result of method execution
        """
        from comfy.isolation.rpc_handlers import rpc_execute_model_method
        
        # Delegate to the actual handler
        result = await rpc_execute_model_method(
            model_id=model_id,
            method_name=method_name,
            args=args,
            kwargs=kwargs
        )
        
        return result
