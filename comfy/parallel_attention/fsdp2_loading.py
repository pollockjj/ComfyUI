"""FSDP2 model loading hook called by comfy/sd.py.

This module provides the entry point for FSDP2-enabled model loading.
Called when model_options['fsdp']['enabled'] is True.
"""

import torch
import logging

LOG_PREFIX = "⚡ [Parallel-Attention][FSDP2Loading]"


def detect_model_type_from_config(model_config):
    """Detect model type from model_config.
    
    Args:
        model_config: ComfyUI model config object
        
    Returns:
        str: Model type identifier ("flux", "wan", "qwen_image", etc.)
    """
    model_class_name = type(model_config).__name__.lower()
    
    if "flux" in model_class_name:
        return "flux"
    elif "wan" in model_class_name:
        return "wan"
    elif "qwen" in model_class_name:
        return "qwen_image"
    else:
        # Default fallback
        logging.warning(f"{LOG_PREFIX} Unknown model type from config: {model_class_name}, defaulting to flux")
        return "flux"


def fsdp2_load_diffusion_model_state_dict(sd, model_options, device_mesh, model_type, cpu_offload):
    """Load model with FSDP2 preparation for distributed inference.
    
    Called by comfy/sd.py when model_options['fsdp']['enabled'] is True.
    
    This function:
    1. Detects model type if not provided
    2. Gets FSDP2 policy from registry
    3. Creates model on CPU (standard ComfyUI)
    4. Adds inner model metadata for worker access
    5. Creates ModelPatcher with parallel_attention metadata
    6. Returns ModelPatcher ready for worker spawning
    
    Args:
        sd: State dict from checkpoint
        model_options: Model options dict
        device_mesh: Optional DeviceMesh for topology-aware sharding
        model_type: Optional explicit model type (auto-detected if None)
        cpu_offload: Whether to offload FSDP parameters to CPU
        
    Returns:
        ModelPatcher with parallel_attention["phase"] = "ready_for_sharding"
    """
    # Detect model type if not provided
    if model_type is None:
        import comfy.model_detection
        model_config = comfy.model_detection.model_config_from_unet(sd, "")
        if model_config is None:
            raise RuntimeError(f"{LOG_PREFIX} Could not detect model type from checkpoint")
        model_type = detect_model_type_from_config(model_config)
    
    logging.info(f"{LOG_PREFIX} Loading model with FSDP2: {model_type}")
    
    # Get policy for model type
    from comfy.parallel_attention import FSDP2PolicyRegistry
    
    if not FSDP2PolicyRegistry.is_registered(model_type):
        available = FSDP2PolicyRegistry.list_registered()
        raise RuntimeError(
            f"{LOG_PREFIX} No FSDP2 policy registered for '{model_type}'. "
            f"Available: {available}"
        )
    
    policy = FSDP2PolicyRegistry.get_policy(model_type)
    logging.info(f"{LOG_PREFIX} Using policy: {policy.model_name}")
    
    # Create model on CPU (standard ComfyUI path)
    import comfy.model_detection
    import comfy.model_management
    
    model_config = comfy.model_detection.model_config_from_unet(sd, "")
    
    # Get model on CPU
    load_device = torch.device("cpu")
    model = model_config.get_model(sd, "", device=load_device)
    model.load_model_weights(sd, "")
    
    logging.info(f"{LOG_PREFIX} Model created on CPU")
    
    # Add inner model metadata if not already set by UNETLoader
    # Workers will access this via model.model._parallel_attention
    if not hasattr(model, '_parallel_attention'):
        model._parallel_attention = {
            "checkpoint_path": None,  # Set by UNETLoader
            "model_type": model_type,
            "policy": policy,
            "vram_per_gpu": 0,
            "sharded_params": 0,
            "replicated_params": 0,
            "vram_freed": 0,
            "vram_after_cleanup": 0,
            "phase": "initialized",
        }
    
    # Update with FSDP2 metadata
    model._parallel_attention["model_type"] = model_type
    model._parallel_attention["policy"] = policy
    model._parallel_attention["device_mesh"] = device_mesh
    model._parallel_attention["cpu_offload"] = cpu_offload
    model._parallel_attention["phase"] = "ready_for_sharding"
    
    logging.debug(f"{LOG_PREFIX} Added _parallel_attention metadata to inner model")
    
    # Create ModelPatcher
    from comfy.model_patcher import ModelPatcher
    
    model_patcher = ModelPatcher(
        model,
        load_device=load_device,
        offload_device=comfy.model_management.unet_offload_device()
    )
    
    # Mark ModelPatcher as FSDP2-ready
    # The parallel_attention dict already exists on ModelPatcher (we are comfy core)
    model_patcher.parallel_attention["enabled"] = True
    model_patcher.parallel_attention["model_type"] = model_type
    model_patcher.parallel_attention["policy"] = policy
    model_patcher.parallel_attention["phase"] = "ready_for_sharding"
    model_patcher.parallel_attention["cpu_offload"] = cpu_offload
    
    logging.info(f"{LOG_PREFIX} ModelPatcher created with FSDP2 metadata")
    logging.info(f"{LOG_PREFIX} Ready for worker spawning (run ParallelAttentionConfig node)")
    
    return model_patcher
