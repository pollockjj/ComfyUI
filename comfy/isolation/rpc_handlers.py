"""
comfy.isolation.rpc_handlers
RPC endpoint handlers for ModelPatcher operations.

This module provides the host-side RPC handler that executes ModelPatcher
methods on behalf of isolated child processes.
"""

import logging
from typing import Any, Tuple, Dict

logger = logging.getLogger(__name__)

# Security: Whitelist of allowed methods (prevent arbitrary execution)
# Generated from ModelPatcher public API analysis (comfy/model_patcher.py)
# Last updated: 2025-11-27
ALLOWED_METHODS = {
    # Core patching
    'add_patches',
    'get_key_patches',
    'model_patches_to',
    'patch_model',
    'unpatch_model',
    
    # Cloning
    'clone',
    'is_clone',
    
    # Memory management
    'model_size',
    'loaded_size',
    'lowvram_patch_counter',
    
    # Metadata
    'model_dtype',
    'get_dtype',
    
    # Object patches
    'add_object_patch',
    'get_model_object',
    
    # Weight manipulation
    'patch_weight_to_device',
    'calculate_weight',
    
    # Options
    'set_model_sampler_cfg_function',
    'set_model_sampler_post_cfg_function',
    'set_model_unet_function_wrapper',
    'set_model_patch_replace',
    'set_model_attn1_patch',
    'set_model_attn2_patch',
    'set_model_attn1_replace',
    'set_model_attn2_replace',
    
    # NOTE: If a node fails with "Method 'X' not in whitelist",
    # audit ModelPatcher source, verify method is safe, then add here.
    # NEVER add methods starting with '_' (private).
}


async def rpc_execute_model_method(
    model_id: str, 
    method_name: str,
    args: Tuple,
    kwargs: Dict
) -> Any:
    """Execute a ModelPatcher method on behalf of isolated proxy.
    
    This is the core RPC endpoint that:
    1. Retrieves ModelPatcher from registry
    2. Validates method is whitelisted
    3. Moves tensor arguments to correct device
    4. Executes method
    5. Serializes result (including new ModelPatcher instances)
    
    Args:
        model_id: UUID referencing ModelPatcher in registry
        method_name: Method to execute (must be in ALLOWED_METHODS)
        args: Positional arguments (may contain tensors)
        kwargs: Keyword arguments (may contain tensors)
        
    Returns:
        Result of method execution (may contain ModelPatcherRef)
        
    Raises:
        ValueError: If model_id not found or method not whitelisted
        AttributeError: If method doesn't exist on ModelPatcher
    """
    # Import here to avoid circular deps and get registry from context
    from comfy.isolation.model_registry import get_current_registry
    from pyisolate._internal.model_serialization import (
        move_tensors_to_device,
        serialize_for_isolation,
    )
    
    # Get the registry for current execution scope
    try:
        registry = get_current_registry()
    except RuntimeError as e:
        logger.error(f"📚 [RPC] No registry in context: {e}")
        raise ValueError(f"RPC called outside execution scope") from e
    
    # Retrieve the ModelPatcher
    patcher = registry.get(model_id)
    if patcher is None:
        logger.error(f"📚 [RPC] ModelPatcher {model_id} not found in registry")
        raise ValueError(f"ModelPatcher {model_id} not found or execution scope expired")
    
    # Security: Validate method is whitelisted
    if method_name.startswith("_"):
        logger.error(f"📚 [RPC] Attempted access to private method: {method_name}")
        raise AttributeError(f"Access denied for private method: {method_name}")
    
    if method_name not in ALLOWED_METHODS:
        logger.error(f"📚 [RPC] Method not in whitelist: {method_name}")
        raise AttributeError(
            f"Method '{method_name}' not in whitelist. "
            f"If this is a legitimate ModelPatcher method, add it to ALLOWED_METHODS in rpc_handlers.py"
        )
    
    # Get the method
    try:
        method = getattr(patcher, method_name)
    except AttributeError as e:
        logger.error(f"📚 [RPC] Method {method_name} not found on ModelPatcher")
        raise AttributeError(f"Method '{method_name}' does not exist on ModelPatcher") from e
    
    # Argument Preparation: Move tensors from CPU shared memory to correct device
    # ModelPatcher methods typically expect tensors on load_device
    target_device = getattr(patcher, 'load_device', None)
    if target_device is not None:
        args = move_tensors_to_device(args, target_device)
        kwargs = move_tensors_to_device(kwargs, target_device)
        logger.debug(f"📚 [RPC] Moved arguments to device: {target_device}")
    
    # Execute the method
    logger.info(f"📚 [RPC] Executing {method_name}(*{len(args)} args, **{len(kwargs)} kwargs) on model_id={model_id}")
    try:
        result = method(*args, **kwargs)
        logger.debug(f"📚 [RPC] {method_name} completed successfully")
    except Exception as e:
        logger.error(f"📚 [RPC] {method_name} raised exception: {e}")
        raise
    
    # Result Handling: Serialize the result
    # If it's a ModelPatcher (e.g., from clone()), it gets registered and an ID is returned
    serialized_result = serialize_for_isolation(result)
    
    return serialized_result


def rpc_execute_model_method_sync(model_id: str, method_name: str, args: Tuple, kwargs: Dict) -> Any:
    """Synchronous wrapper for RPC calls from within ModelPatcher methods.
    
    This allows ModelPatcher.clone() (and other methods) to call RPC synchronously
    even though the underlying RPC mechanism is async.
    
    Args:
        model_id: UUID referencing ModelPatcher in host registry
        method_name: Method to execute
        args: Positional arguments
        kwargs: Keyword arguments
        
    Returns:
        Result of method execution (already deserialized)
    """
    import asyncio
    from comfy.isolation.proxies.model_patcher_rpc import ModelPatcherRPC
    
    logger.debug(f"📚 [RPC][Sync] Calling {method_name} on model {model_id}")
    
    # Get the RPC endpoint
    rpc_endpoint = ModelPatcherRPC()
    
    # Create coroutine
    coro = rpc_endpoint.execute_method(
        model_id=model_id,
        method_name=method_name,
        args=args,
        kwargs=kwargs
    )
    
    # Execute - we're in node context which is sync, but called from async extension wrapper
    # The extension wrapper is async, so there's a running loop
    try:
        loop = asyncio.get_running_loop()
        # Schedule on the running loop and wait
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        result = future.result(timeout=60)  # 60 second timeout for large operations
        logger.debug(f"📚 [RPC][Sync] {method_name} completed")
        return result
    except RuntimeError:
        # No running loop (shouldn't happen in isolated node context, but handle it)
        logger.warning(f"📚 [RPC][Sync] No running loop, creating new one for {method_name}")
        result = asyncio.run(coro)
        return result


# Export for registration with PyIsolate
RPC_HANDLERS = {
    "rpc_execute_model_method": rpc_execute_model_method,
}
