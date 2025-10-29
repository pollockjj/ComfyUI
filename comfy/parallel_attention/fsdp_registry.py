"""FSDP registry for model-specific wrapping strategies.

Provides centralized registry for detecting model types and retrieving
their corresponding FSDP wrapping policies.

Design:
- Registry pattern: Policies registered in fsdp_policies.py
- Detection logic: Maps state_dict keys to model types
- Extensible: Add new models by registering policy + detection logic

Usage:
    # Auto-detect model type and get policy
    policy_fn = get_fsdp_strategy(state_dict=sd)
    policy = policy_fn()
    
    # Or specify model type explicitly
    policy_fn = get_fsdp_strategy(model_type="flux")
    policy = policy_fn()
"""

import logging
from typing import Callable, Optional, Dict
from .fsdp_policies import FSDPPolicyRegistry

LOG_PREFIX = "⚡ [Parallel-Attention]"


def detect_model_type(state_dict: dict, model_config=None) -> Optional[str]:
    """Detect model type from state dict keys.
    
    Maps state dict structure to model architecture. Uses ComfyUI's
    model_config.unet_config["image_model"] or state dict key patterns
    to identify Flux, Qwen Image, Wan, etc.
    
    Args:
        state_dict: Model state dictionary
        model_config: Optional ComfyUI model config object
    
    Returns:
        str: Model type identifier (e.g., "flux", "qwen_image", "wan")
        None: If model type cannot be detected
    
    Detection Logic:
        1. Check model_config.unet_config["image_model"] (ComfyUI's detection)
        2. Check model_config.unet_config["audio_model"] (ACE, stable audio)
        3. Check model_config.unet_config for structure patterns
        4. Fallback to state dict key detection
        
    Example:
        model_type = detect_model_type(state_dict, model_config)
        if model_type == "flux":
            print("Detected Flux model")
    """
    # Method 1: Check model_config.unet_config["image_model"] - ComfyUI's built-in detection
    if model_config is not None and hasattr(model_config, 'unet_config'):
        unet_config = model_config.unet_config
        if isinstance(unet_config, dict):
            # ComfyUI sets image_model for DiT models
            image_model = unet_config.get("image_model")
            if image_model == "flux":
                logging.info(f"{LOG_PREFIX} [FSDPRegistry] Detected Flux from unet_config[image_model]")
                return "flux"
            elif image_model == "qwen_image":
                logging.info(f"{LOG_PREFIX} [FSDPRegistry] Detected Qwen Image from unet_config[image_model]")
                return "qwen_image"
            elif image_model in ["wan2.1"]:
                logging.info(f"{LOG_PREFIX} [FSDPRegistry] Detected Wan from unet_config[image_model]={image_model}")
                return "wan"
            
            # Check for other model indicators in unet_config
            if 'depth' in unet_config and 'depth_single_blocks' in unet_config:
                logging.info(f"{LOG_PREFIX} [FSDPRegistry] Detected Flux from unet_config structure (depth + depth_single_blocks)")
                return "flux"
    
    # Method 2: Fallback to state dict key detection
    keys = list(state_dict.keys())
    
    # Flux detection: Look for double_blocks and single_blocks
    has_double_blocks = any('double_blocks' in k for k in keys)
    has_single_blocks = any('single_blocks' in k for k in keys)
    
    if has_double_blocks or has_single_blocks:
        logging.info(f"{LOG_PREFIX} [FSDPRegistry] Detected Flux model from state dict keys (double_blocks or single_blocks)")
        return "flux"
    
    # Qwen Image detection: Look for transformer_blocks with qwen structure
    # Qwen has: transformer_blocks.N.img_attn, transformer_blocks.N.txt_attn
    has_img_attn = any('img_attn' in k for k in keys)
    has_txt_attn = any('txt_attn' in k for k in keys)
    has_transformer_blocks = any('transformer_blocks' in k for k in keys)
    
    if has_transformer_blocks and has_img_attn and has_txt_attn:
        logging.info(f"{LOG_PREFIX} [FSDPRegistry] Detected Qwen Image model (transformer_blocks with dual attn)")
        return "qwen_image"
    
    # Wan detection: Look for transformer_blocks with different structure
    # Wan has: transformer_blocks.N.attn1, transformer_blocks.N.attn2
    has_attn1 = any('attn1' in k and 'transformer_blocks' in k for k in keys)
    has_attn2 = any('attn2' in k and 'transformer_blocks' in k for k in keys)
    
    if has_transformer_blocks and has_attn1 and has_attn2:
        logging.info(f"{LOG_PREFIX} [FSDPRegistry] Detected Wan model (transformer_blocks with attn1/attn2)")
        return "wan"
    
    # Could not detect
    logging.warning(
        f"{LOG_PREFIX} [FSDPRegistry] Could not detect model type. "
        f"Available policies: {FSDPPolicyRegistry.list_registered()}"
    )
    return None


def get_fsdp_strategy(
    state_dict: Optional[dict] = None,
    model_type: Optional[str] = None,
    model_config = None
) -> Callable:
    """Get FSDP wrapping strategy for a model.
    
    Retrieves the appropriate FSDP auto_wrap_policy function for a given
    model. Can auto-detect model type from state dict, or use explicit type.
    
    Args:
        state_dict: Model state dictionary (for auto-detection)
        model_type: Explicit model type (e.g., "flux", "qwen_image", "wan")
        model_config: Optional ComfyUI model config object
    
    Returns:
        Callable: Policy function that returns auto_wrap_policy when called
        
    Raises:
        ValueError: If model type cannot be determined or no policy registered
        
    Example:
        # Auto-detect
        policy_fn = get_fsdp_strategy(state_dict=sd)
        policy = policy_fn()
        
        # Explicit
        policy_fn = get_fsdp_strategy(model_type="flux")
        policy = policy_fn()
        
        # Use with FSDP
        fsdp_model = FSDP(model, auto_wrap_policy=policy, ...)
    """
    # Determine model type
    if model_type is None:
        if state_dict is None:
            raise ValueError(
                "Must provide either 'model_type' or 'state_dict' for model detection"
            )
        model_type = detect_model_type(state_dict, model_config=model_config)
        
        if model_type is None:
            available = FSDPPolicyRegistry.list_registered()
            raise ValueError(
                f"Could not detect model type from state dict. "
                f"Available policies: {available}. "
                f"Please specify model_type explicitly."
            )
    
    # Get policy from registry
    logging.info(f"{LOG_PREFIX} [FSDPRegistry] Retrieving FSDP strategy for: {model_type}")
    
    try:
        policy_fn = FSDPPolicyRegistry.get_policy(model_type)
        return policy_fn
    except ValueError as e:
        # Re-raise with more context
        available = FSDPPolicyRegistry.list_registered()
        raise ValueError(
            f"No FSDP policy for model type '{model_type}'. "
            f"Available: {available}"
        ) from e


def list_available_strategies() -> list:
    """List all registered FSDP strategies.
    
    Returns:
        list: List of model type strings with registered strategies
        
    Example:
        strategies = list_available_strategies()
        print(f"Available: {strategies}")
        # Output: ["flux", "qwen_image", "wan"]
    """
    return FSDPPolicyRegistry.list_registered()


def is_strategy_available(model_type: str) -> bool:
    """Check if FSDP strategy is available for model type.
    
    Args:
        model_type: Model type identifier
        
    Returns:
        bool: True if strategy registered, False otherwise
        
    Example:
        if is_strategy_available("flux"):
            policy_fn = get_fsdp_strategy(model_type="flux")
    """
    return FSDPPolicyRegistry.is_registered(model_type)
