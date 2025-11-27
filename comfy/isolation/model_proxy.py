"""
comfy.isolation.model_proxy
Isolated-process stateless proxy for ModelPatcher objects.
"""

import logging
import asyncio
from typing import Any, Callable

logger = logging.getLogger(__name__)


class ModelPatcherProxy:
    """Stateless proxy forwarding all ModelPatcher operations to host via RPC.
    
    Design Principles:
    1. Zero State: Only stores model_id + RPC client reference
    2. Transparent: Appears identical to ModelPatcher from node's perspective
    3. Lazy: Method wrappers created on-demand via __getattr__
    4. Fail-Loud: Any RPC failure raises immediately (no silent failures)
    
    Usage (automatically created by deserializer):
        proxy = ModelPatcherProxy(model_id="abc-123", rpc_client=client)
        proxy.add_patches({"key": patches})  # RPC call to host
        result = proxy.clone()  # Returns another ModelPatcherProxy
    """
    
    def __init__(self, model_id: str, rpc_client: Any):
        """Initialize stateless proxy.
        
        Args:
            model_id: UUID referencing host-side ModelPatcher
            rpc_client: RPC client for communicating with host
        """
        # Use object.__setattr__ to bypass __setattr__ override
        object.__setattr__(self, '_model_id', model_id)
        object.__setattr__(self, '_rpc_client', rpc_client)
        logger.debug(f"📚 [ModelProxy] Created proxy for model_id {model_id}")
    
    def __getattr__(self, name: str) -> Callable:
        """Forward all attribute access to host via RPC.
        
        This magic method is called for ANY attribute not found in __dict__.
        We use it to create method wrappers that forward calls to the host.
        
        Args:
            name: Attribute/method name being accessed
            
        Returns:
            Callable that executes RPC when invoked
        """
        model_id = object.__getattribute__(self, '_model_id')
        rpc_client = object.__getattribute__(self, '_rpc_client')
        
        def sync_method_wrapper(*args, **kwargs):
            """Synchronous wrapper for node execution context."""
            logger.debug(f"📚 [ModelProxy] RPC call: {name}(*args={len(args)}, **kwargs={len(kwargs)})")
            
            try:
                # Get the ProxiedSingleton RPC endpoint
                from comfy.isolation.proxies.model_patcher_rpc import ModelPatcherRPC
                rpc_endpoint = ModelPatcherRPC()
                
                # Create the coroutine
                coro = rpc_endpoint.execute_method(
                    model_id=model_id,
                    method_name=name,
                    args=args,
                    kwargs=kwargs
                )
                
                # Try to get the running loop
                try:
                    loop = asyncio.get_running_loop()
                    # Schedule the coroutine on the running loop from this sync context
                    future = asyncio.run_coroutine_threadsafe(coro, loop)
                    # Wait for it to complete (blocks this thread)
                    result = future.result()
                except RuntimeError:
                    # No running loop, create and run synchronously
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        result = loop.run_until_complete(coro)
                    finally:
                        loop.close()
                
                logger.debug(f"📚 [ModelProxy] RPC {name} completed successfully")
                return result
            
            except Exception as e:
                logger.error(f"📚 [ModelProxy] RPC {name} failed: {e}")
                raise
        
        return sync_method_wrapper
    
    def __setattr__(self, name: str, value: Any) -> None:
        """Prevent attribute assignment (proxy is stateless)."""
        if name.startswith('_'):
            object.__setattr__(self, name, value)
        else:
            raise AttributeError(
                f"Cannot set attribute '{name}' on ModelPatcherProxy. "
                f"This is a stateless proxy forwarding to host."
            )
    
    def __repr__(self) -> str:
        model_id = object.__getattribute__(self, '_model_id')
        return f"<ModelPatcherProxy model_id={model_id}>"
