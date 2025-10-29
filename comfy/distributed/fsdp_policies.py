"""FSDP wrapping policies for model-specific sharding strategies.

Provides registry pattern for defining how FSDP should wrap different
model architectures. Each model (Flux, SD3, Wan, etc.) has specific
block structures that need custom wrapping logic.

Design:
- Registry pattern: Decorator-based policy registration
- Model-agnostic API: get_policy(model_name) returns callable
- Extensible: Easy to add new models without modifying registry
- Type-safe: Policies return standard PyTorch auto_wrap_policy functions

Based on Raylight patterns, adapted to ComfyUI model structure.

Usage:
    # Register a policy
    @FSDPPolicyRegistry.register("flux")
    def flux_fsdp_policy():
        return partial(transformer_auto_wrap_policy, ...)
    
    # Get a policy
    policy_fn = FSDPPolicyRegistry.get_policy("flux")
    policy = policy_fn()
    
    # Use with FSDP
    fsdp_model = FSDP(model, auto_wrap_policy=policy, ...)
"""

from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from functools import partial
from typing import Callable, Dict, Set
import logging

LOG_PREFIX = "⚡ [Parallel-Attention]"


class FSDPPolicyRegistry:
    """Registry for model-specific FSDP wrapping policies.
    
    Provides decorator pattern for registering policies and retrieving
    them by model name. Ensures clean separation between policy definitions
    and FSDP loader logic.
    
    Example:
        @FSDPPolicyRegistry.register("my_model")
        def my_model_policy():
            return partial(transformer_auto_wrap_policy, ...)
        
        # Later, in loader
        policy_fn = FSDPPolicyRegistry.get_policy("my_model")
        policy = policy_fn()
    """
    
    _policies: Dict[str, Callable] = {}
    
    @classmethod
    def register(cls, model_name: str):
        """Decorator to register an FSDP policy function.
        
        Args:
            model_name: Unique identifier for model (e.g., "flux", "sd3")
            
        Returns:
            Decorator function
            
        Example:
            @FSDPPolicyRegistry.register("flux")
            def flux_fsdp_policy():
                return partial(...)
        """
        def decorator(policy_fn: Callable) -> Callable:
            if model_name in cls._policies:
                logging.warning(
                    f"{LOG_PREFIX} [FSDPPolicy] Overwriting existing policy for '{model_name}'"
                )
            
            cls._policies[model_name] = policy_fn
            logging.info(f"{LOG_PREFIX} [FSDPPolicy] Registered policy: {model_name}")
            return policy_fn
        
        return decorator
    
    @classmethod
    def get_policy(cls, model_name: str) -> Callable:
        """Retrieve a registered FSDP policy function.
        
        Args:
            model_name: Model identifier used during registration
            
        Returns:
            Policy function that returns auto_wrap_policy
            
        Raises:
            ValueError: If model_name not registered
            
        Example:
            policy_fn = FSDPPolicyRegistry.get_policy("flux")
            policy = policy_fn()  # Call to get actual policy
        """
        if model_name not in cls._policies:
            available = ", ".join(cls._policies.keys()) if cls._policies else "(none)"
            raise ValueError(
                f"No FSDP policy registered for '{model_name}'. "
                f"Available policies: {available}"
            )
        
        logging.debug(f"{LOG_PREFIX} [FSDPPolicy] Retrieved policy: {model_name}")
        return cls._policies[model_name]
    
    @classmethod
    def list_registered(cls) -> list:
        """List all registered model names.
        
        Returns:
            List of model name strings
        """
        return list(cls._policies.keys())
    
    @classmethod
    def is_registered(cls, model_name: str) -> bool:
        """Check if a model has a registered policy.
        
        Args:
            model_name: Model identifier to check
            
        Returns:
            True if registered, False otherwise
        """
        return model_name in cls._policies


@FSDPPolicyRegistry.register("flux")
def flux_fsdp_policy() -> Callable:
    """Return FSDP auto-wrap policy for Flux-Dev model.
    
    Wraps DoubleStreamBlock and SingleStreamBlock as independent FSDP units.
    Each block will be sharded across GPUs, with parameters all-gathered
    during forward pass.
    
    Flux Architecture:
        - 19 DoubleStreamBlocks (img + txt dual-stream attention)
        - 38 SingleStreamBlocks (img-only attention)
        - Each DoubleStreamBlock: ~350MB parameters
        - Each SingleStreamBlock: ~250MB parameters
        - Total: 57 FSDP units, ~22GB parameters
    
    FSDP Behavior:
        - With 2 GPUs: ~11GB parameters per GPU
        - Parameters all-gathered on-demand during forward
        - Parameters freed after forward (if using FULL_SHARD)
        - Gradients not needed (inference-only)
    
    Returns:
        Callable: transformer_auto_wrap_policy configured for Flux
        
    Notes:
        - Based on Raylight implementation
        - Adapted to ComfyUI's ldm.flux.model module structure
        - CPU offload further reduces VRAM (Phase 2.2)
        
    Example:
        policy = flux_fsdp_policy()
        fsdp_model = FSDP(model, auto_wrap_policy=policy, ...)
    """
    from comfy.ldm.flux.model import DoubleStreamBlock, SingleStreamBlock
    
    # Log layer classes being wrapped
    layer_cls_names = [DoubleStreamBlock.__name__, SingleStreamBlock.__name__]
    logging.info(
        f"{LOG_PREFIX} [FSDPPolicy] Flux wrapping policy: "
        f"layers={layer_cls_names}"
    )
    
    # Return transformer auto-wrap policy
    # This will wrap any module that is an instance of these classes
    return partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={DoubleStreamBlock, SingleStreamBlock}
    )


@FSDPPolicyRegistry.register("qwen_image")
def qwen_image_fsdp_policy() -> Callable:
    """Return FSDP auto-wrap policy for Qwen VL Image model.
    
    Wraps QwenImageTransformerBlock as independent FSDP units.
    Each block contains dual-stream attention (img + txt) with modulation.
    
    Qwen Image Architecture:
        - 60 QwenImageTransformerBlocks (default)
        - Each block: dual attention + 2x MLPs
        - ~3072 dim, 24 attention heads
    
    FSDP Behavior:
        - 60 FSDP sharding units
        - Parameters all-gathered on-demand during forward
        - Parameters freed after forward (if using FULL_SHARD)
    
    Returns:
        Callable: transformer_auto_wrap_policy configured for Qwen Image
        
    Notes:
        - Based on Raylight implementation patterns
        - Similar dual-stream architecture to Flux
        
    Example:
        policy = qwen_image_fsdp_policy()
        fsdp_model = FSDP(model, auto_wrap_policy=policy, ...)
    """
    from comfy.ldm.qwen_image.model import QwenImageTransformerBlock
    
    # Log layer classes being wrapped
    logging.info(
        f"{LOG_PREFIX} [FSDPPolicy] Qwen Image wrapping policy: "
        f"layers=[{QwenImageTransformerBlock.__name__}]"
    )
    
    # Return transformer auto-wrap policy
    return partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={QwenImageTransformerBlock}
    )


@FSDPPolicyRegistry.register("wan")
def wan_fsdp_policy() -> Callable:
    """Return FSDP auto-wrap policy for Wan2.2 video model.
    
    Wraps WanAttentionBlock as independent FSDP units.
    Each block contains self-attention, cross-attention, and FFN layers.
    
    Wan Architecture:
        - 32 WanAttentionBlocks (default, configurable)
        - Each block: self-attn + cross-attn + FFN
        - ~2048 dim, ~8192 FFN dim
    
    FSDP Behavior:
        - 32 FSDP sharding units (default)
        - Parameters all-gathered on-demand during forward
        - Parameters freed after forward (if using FULL_SHARD)
    
    Returns:
        Callable: transformer_auto_wrap_policy configured for Wan
        
    Notes:
        - Based on FastVideo/Raylight implementation patterns
        - Video model with temporal attention
        
    Example:
        policy = wan_fsdp_policy()
        fsdp_model = FSDP(model, auto_wrap_policy=policy, ...)
    """
    from comfy.ldm.wan.model import WanAttentionBlock
    
    # Log layer classes being wrapped
    logging.info(
        f"{LOG_PREFIX} [FSDPPolicy] Wan wrapping policy: "
        f"layers=[{WanAttentionBlock.__name__}]"
    )
    
    # Return transformer auto-wrap policy
    return partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={WanAttentionBlock}
    )
