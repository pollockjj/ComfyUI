"""FSDP loading functions using ComfyUI model detection.

Provides distributed model loading that:
1. Uses ComfyUI's model_detection to identify model type
2. Retrieves appropriate FSDP wrapping policy from registry
3. Loads state dict using FSDP's distributed loading
4. Returns FSDPModelPatcher instead of ModelPatcher

Design: Minimal, surgical integration
- Wraps ComfyUI's load_diffusion_model_state_dict() logic
- Reuses all ComfyUI detection and configuration
- Returns FSDP-aware patcher for transparent integration

Usage:
    # Instead of load_diffusion_model_state_dict()
    patcher = fsdp_load_diffusion_model_state_dict(
        sd=state_dict,
        model_options={'fsdp': {'enabled': True}},
        device_mesh=mesh
    )
"""

import logging
from typing import Optional
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, MixedPrecision
from torch.distributed.fsdp.api import FullStateDictConfig, StateDictType

import comfy.utils
import comfy.model_management
import comfy.model_detection

from .fsdp_model_patcher import FSDPModelPatcher
from .fsdp_registry import get_fsdp_strategy, detect_model_type

LOG_PREFIX = "⚡ [Parallel-Attention]"


def fsdp_load_diffusion_model_state_dict(
    sd: dict,
    model_options: dict = {},
    device_mesh = None,
    model_type: Optional[str] = None,
    cpu_offload: bool = False
):
    """Load diffusion model with FSDP sharding from state dict.
    
    Drop-in replacement for comfy.sd.load_diffusion_model_state_dict()
    that returns FSDPModelPatcher instead of ModelPatcher.
    
    Args:
        sd: Model state dictionary
        model_options: ComfyUI model options dict (dtype, custom_operations, etc.)
        device_mesh: Optional DeviceMesh for topology-aware sharding
        model_type: Optional explicit model type (auto-detected if None)
        cpu_offload: Whether to offload FSDP parameters to CPU
    
    Returns:
        FSDPModelPatcher: FSDP-wrapped model patcher
        None: If model config cannot be detected
        
    Process:
        1. Use ComfyUI's detection logic (reuse existing code)
        2. Detect model type (Flux, Qwen, Wan, etc.)
        3. Get FSDP wrapping policy from registry
        4. Create model instance (no weights loaded yet)
        5. Wrap with FSDP (using policy)
        6. Load state dict using FSDP's distributed loading
        7. Return FSDPModelPatcher
        
    Example:
        # In comfy/sd.py hook:
        if model_options.get('fsdp', {}).get('enabled'):
            return fsdp_load_diffusion_model_state_dict(
                sd=sd,
                model_options=model_options,
                device_mesh=mesh
            )
    """
    if not dist.is_initialized():
        raise RuntimeError(
            f"{LOG_PREFIX} [FSDPLoading] torch.distributed not initialized. "
            "Cannot load FSDP model without distributed environment."
        )
    
    logging.info(f"{LOG_PREFIX} [FSDPLoading] Starting FSDP model loading...")
    
    dtype = model_options.get("dtype", None)
    
    # Step 1: Detect model format (regular, diffusers, mmdit)
    # This is copied from comfy.sd.load_diffusion_model_state_dict()
    diffusion_model_prefix = comfy.model_detection.unet_prefix_from_state_dict(sd)
    temp_sd = comfy.utils.state_dict_prefix_replace(sd, {diffusion_model_prefix: ""}, filter_keys=True)
    if len(temp_sd) > 0:
        sd = temp_sd
    
    parameters = comfy.utils.calculate_parameters(sd)
    weight_dtype = comfy.utils.weight_dtype(sd)
    
    load_device = comfy.model_management.get_torch_device()
    
    # Step 2: Get model config using ComfyUI's detection
    model_config = comfy.model_detection.model_config_from_unet(sd, "")
    
    if model_config is not None:
        new_sd = sd
    else:
        # Try diffusers mmdit format
        new_sd = comfy.model_detection.convert_diffusers_mmdit(sd, "")
        if new_sd is not None:
            model_config = comfy.model_detection.model_config_from_unet(new_sd, "")
            if model_config is None:
                logging.error(f"{LOG_PREFIX} [FSDPLoading] Could not detect model config (mmdit)")
                return None
        else:
            # Try diffusers unet format
            model_config = comfy.model_detection.model_config_from_diffusers_unet(sd)
            if model_config is None:
                logging.error(f"{LOG_PREFIX} [FSDPLoading] Could not detect model config (diffusers)")
                return None
            
            diffusers_keys = comfy.utils.unet_to_diffusers(model_config.unet_config)
            new_sd = {}
            for k in diffusers_keys:
                if k in sd:
                    new_sd[diffusers_keys[k]] = sd.pop(k)
                else:
                    logging.warning(f"{LOG_PREFIX} [FSDPLoading] Missing key: {diffusers_keys[k]} -> {k}")
    
    offload_device = comfy.model_management.unet_offload_device()
    unet_weight_dtype = list(model_config.supported_inference_dtypes)
    
    if model_config.scaled_fp8 is not None:
        weight_dtype = None
    
    # Determine dtype
    if dtype is None:
        unet_dtype = comfy.model_management.unet_dtype(
            model_params=parameters,
            supported_dtypes=unet_weight_dtype,
            weight_dtype=weight_dtype
        )
    else:
        unet_dtype = dtype
    
    manual_cast_dtype = comfy.model_management.unet_manual_cast(
        unet_dtype, load_device, model_config.supported_inference_dtypes
    )
    
    model_config.set_inference_dtype(unet_dtype, manual_cast_dtype)
    model_config.custom_operations = model_options.get("custom_operations", model_config.custom_operations)
    
    if model_options.get("fp8_optimizations", False):
        model_config.optimizations["fp8"] = True
    
    logging.info(
        f"{LOG_PREFIX} [FSDPLoading] Model config detected: "
        f"dtype={unet_dtype}, params={parameters/1e9:.2f}B"
    )
    
    # Step 3: Detect model type and get FSDP policy
    if model_type is None:
        model_type = detect_model_type(new_sd, model_config=model_config)
    
    if model_type is None:
        logging.error(
            f"{LOG_PREFIX} [FSDPLoading] Could not detect model type for FSDP. "
            f"FSDP currently supports: flux, qwen_image, wan"
        )
        return None
    
    logging.info(f"{LOG_PREFIX} [FSDPLoading] Detected model type: {model_type}")
    
    try:
        policy_fn = get_fsdp_strategy(model_type=model_type)
        auto_wrap_policy = policy_fn()
        logging.info(f"{LOG_PREFIX} [FSDPLoading] Retrieved FSDP policy for {model_type}")
    except ValueError as e:
        logging.error(
            f"{LOG_PREFIX} [FSDPLoading] No FSDP policy registered for model type '{model_type}'. "
            f"Supported models: flux, qwen_image, wan. Error: {e}"
        )
        return None
    
    # Step 4: Create model structure (EXACTLY like ComfyUI, but DON'T load weights yet)
    # From comfy/sd.py load_state_dict_guess_config():
    #   inital_load_device = model_management.unet_inital_load_device(parameters, unet_dtype)
    #   model = model_config.get_model(sd, diffusion_model_prefix, device=inital_load_device)
    # NOTE: We skip model.load_model_weights() - FSDP will load shards distributed
    inital_load_device = comfy.model_management.unet_inital_load_device(parameters, unet_dtype)
    model = model_config.get_model(new_sd, "", device=inital_load_device)
    
    logging.info(f"{LOG_PREFIX} [FSDPLoading] Model structure created on device={inital_load_device} (weights NOT loaded yet)")
    
    # Step 5: Create FSDPModelPatcher with FSDP config and state dict for distributed loading
    fsdp_config = {
        'auto_wrap_policy': auto_wrap_policy,
        'cpu_offload': cpu_offload,
        'sharding_strategy': ShardingStrategy.FULL_SHARD,
        'device_mesh': device_mesh,
        'state_dict': new_sd,  # Pass state dict for FSDP distributed loading
    }
    
    # Optional: Configure mixed precision
    # For bf16 compute with fp32 parameters
    if unet_dtype == torch.bfloat16:
        fsdp_config['mixed_precision'] = MixedPrecision(
            param_dtype=torch.float32,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32,
        )
    
    # Create FSDPModelPatcher (will wrap with FSDP and load shards on first load())
    patcher = FSDPModelPatcher(
        model=model,
        load_device=load_device,
        offload_device=offload_device,
        fsdp_config=fsdp_config
    )
    
    logging.info(f"{LOG_PREFIX} [FSDPLoading] FSDPModelPatcher created (FSDP wrapping and shard loading deferred)")
    
    # Step 6: Check for leftover keys by comparing against loaded model parameters
    # Get actual parameter names from the wrapped FSDP model
    # FSDP adds _fsdp_wrapped_module prefixes, but we can get the original names
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    
    # Get all parameter keys that were actually loaded into the model
    # We need to check against new_sd (the stripped version we tried to load)
    loaded_keys = set()
    for name, param in patcher.model.named_parameters():
        # Strip FSDP wrapper prefixes to get original key name
        clean_name = name.replace('_fsdp_wrapped_module.', '')
        loaded_keys.add(clean_name)
    
    # Also check buffers (like normalization stats)
    for name, buffer in patcher.model.named_buffers():
        clean_name = name.replace('_fsdp_wrapped_module.', '')
        loaded_keys.add(clean_name)
    
    # Find keys from new_sd that didn't match any model parameters
    state_dict_keys = set(new_sd.keys())
    leftover_keys = state_dict_keys - loaded_keys
    
    if len(leftover_keys) > 0:
        logging.warning(
            f"{LOG_PREFIX} [FSDPLoading] {len(leftover_keys)} keys from checkpoint "
            f"not loaded into model (may be non-shardable components)"
        )
        # Show sample of leftover keys for debugging
        leftover_sample = sorted(list(leftover_keys))[:20]
        for key in leftover_sample:
            logging.warning(f"{LOG_PREFIX} [FSDPLoading]   Unused key: {key}")
    
    # Log memory stats
    if torch.cuda.is_available():
        rank = dist.get_rank()
        allocated = torch.cuda.memory_allocated(load_device) / 1024**3
        logging.info(
            f"{LOG_PREFIX} [FSDPLoading] Rank {rank} VRAM: {allocated:.2f}GB allocated"
        )
    
    logging.info(f"{LOG_PREFIX} [FSDPLoading] FSDP model loading complete!")
    
    return patcher


def fsdp_load_diffusion_model(
    unet_path: str,
    model_options: dict = {},
    device_mesh = None,
    model_type: Optional[str] = None,
    cpu_offload: bool = False
):
    """Load diffusion model with FSDP sharding from file path.
    
    Drop-in replacement for comfy.sd.load_diffusion_model() that
    returns FSDPModelPatcher instead of ModelPatcher.
    
    Args:
        unet_path: Path to model checkpoint file
        model_options: ComfyUI model options dict
        device_mesh: Optional DeviceMesh for topology-aware sharding
        model_type: Optional explicit model type (auto-detected if None)
        cpu_offload: Whether to offload FSDP parameters to CPU
    
    Returns:
        FSDPModelPatcher: FSDP-wrapped model patcher
        
    Raises:
        RuntimeError: If model type cannot be detected
        
    Example:
        patcher = fsdp_load_diffusion_model(
            unet_path="models/unet/flux_dev.safetensors",
            model_options={'fsdp': {'enabled': True}},
            device_mesh=mesh
        )
    """
    sd = comfy.utils.load_torch_file(unet_path)
    
    model = fsdp_load_diffusion_model_state_dict(
        sd=sd,
        model_options=model_options,
        device_mesh=device_mesh,
        model_type=model_type,
        cpu_offload=cpu_offload
    )
    
    if model is None:
        logging.error(f"{LOG_PREFIX} [FSDPLoading] ERROR: Could not load FSDP model from {unet_path}")
        raise RuntimeError(f"ERROR: Could not detect model type of: {unet_path}")
    
    return model
