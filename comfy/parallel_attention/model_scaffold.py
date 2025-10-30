"""Model scaffold extraction for distributed loading.

Creates model STRUCTURE (0GB - no weights loaded), deepcopies it as scaffold.
The scaffold IS a real model object with all properties, just no weights.

Design Philosophy: "Copy at Perfect Information, Don't Reconstruct from Imperfect"
- Create model structure WITHOUT loading 22GB weights
- Deepcopy the structure - that IS the scaffold
- No manual property serialization (brittle, incomplete)
- Workers load their own weights with FSDP

Based on WorkSplit deepclone pattern (ComfyUI core architect).
Reference: reference_worksplit_multigpu/comfy/model_patcher.py:351-374
ComfyUI pattern: comfy/sd.py:1351-1352 (get_model creates structure, load_model_weights loads tensors)
"""

from __future__ import annotations
import torch
import logging
import copy
from typing import Dict, Any

LOG_PREFIX = "⚡ [Parallel-Attention]"


def extract_model_scaffold(checkpoint_path: str) -> tuple[Any, Dict]:
    """Extract model scaffold WITHOUT loading 22GB weights.
    
    Creates model STRUCTURE only (0GB - architecture, no tensors),
    deepcopies it. That deepcopy IS the scaffold - has ALL properties.
    
    Args:
        checkpoint_path: Path to .safetensors checkpoint
    
    Returns:
        (scaffold_model, state_dict):
            scaffold_model: Deepcopy of model structure (0GB, all properties)
            state_dict: Model weights for FSDP loading in workers
    
    Design Philosophy:
        - ComfyUI's get_model() creates structure (small)
        - ComfyUI's load_model_weights() loads tensors (22GB)
        - We call get_model() but NOT load_model_weights()
        - Deepcopy the 0GB structure - that's the scaffold
        - No serialization, no manual property lists, not brittle
    """
    import comfy.model_detection
    import comfy.utils
    import comfy.model_management
    
    logging.info(f"{LOG_PREFIX} [Scaffold] Extracting model scaffold (0GB structure) from {checkpoint_path}")
    
    # Load state dict
    state_dict = comfy.utils.load_torch_file(checkpoint_path)
    
    # CRITICAL: Follow ComfyUI's exact pattern from sd.py:1303-1313
    # 1. Get the prefix (e.g., "model.diffusion_model.")
    diffusion_model_prefix = comfy.model_detection.unet_prefix_from_state_dict(state_dict)
    
    # 2. STRIP the prefix from state dict keys
    temp_sd = comfy.utils.state_dict_prefix_replace(state_dict, {diffusion_model_prefix: ""}, filter_keys=True)
    if len(temp_sd) > 0:
        state_dict = temp_sd
    
    # 3. Detect model config with EMPTY prefix
    load_device = comfy.model_management.get_torch_device()
    offload_device = comfy.model_management.unet_offload_device()
    model_config = comfy.model_detection.model_config_from_unet(state_dict, "")
    
    if model_config is None:
        raise RuntimeError(
            f"{LOG_PREFIX} [Scaffold] Failed to detect model config from checkpoint. "
            f"Prefix used: '{diffusion_model_prefix}', keys after strip: {len(state_dict)}"
        )
    
    # Create model STRUCTURE (0GB - no weights loaded!)
    # This is ComfyUI's pattern from sd.py:1351
    # get_model() creates nn.Module architecture without loading tensors
    model = model_config.get_model(state_dict, "")
    model = model.to(offload_device)
    
    # DO NOT call model.load_model_weights(state_dict, "") - that's 22GB!
    # We only want the STRUCTURE, not the weights
    
    # Deepcopy the structure - THIS IS THE SCAFFOLD
    # Has ALL properties: latent_format, load_device, dtype, is_adm(), etc.
    # No manual serialization required - it's a real model object
    scaffold_model = copy.deepcopy(model)
    
    # Store additional metadata on scaffold for convenience
    scaffold_model._scaffold_checkpoint_path = checkpoint_path
    scaffold_model._scaffold_model_config = model_config
    scaffold_model._scaffold_load_device = load_device
    scaffold_model._scaffold_offload_device = offload_device
    
    logging.info(
        f"{LOG_PREFIX} [Scaffold] Scaffold extracted (0GB structure): "
        f"type={model_config.unet_config.get('model_type', 'unknown')}, "
        f"dtype={scaffold_model.get_dtype()}, "
        f"latent_format={scaffold_model.latent_format.__class__.__name__}"
    )
    
    return scaffold_model, state_dict

    """Serialize model_config for reconstruction.
    
    Args:
        model_config: ComfyUI model config object
    
    Returns:
        Serialized configuration dict
    """
    return {
        "unet_config": model_config.unet_config,
        "latent_format_class": model_config.latent_format.__class__.__name__,
        "manual_cast_dtype": str(model_config.manual_cast_dtype) if model_config.manual_cast_dtype else None,
    }


def _serialize_latent_format(latent_format) -> Dict[str, Any]:
    """Serialize latent_format object (ALL properties).
    
    Captures all properties needed for exact reconstruction.
    This ensures fix_empty_latent_channels() and other ComfyUI
    functions work correctly with distributed wrapper.
    
    Args:
        latent_format: ComfyUI latent format object
    
    Returns:
        Complete serialization of latent format properties
    """
    return {
        "class_name": latent_format.__class__.__name__,
        "scale_factor": getattr(latent_format, "scale_factor", 1.0),
        "shift_factor": getattr(latent_format, "shift_factor", None),
        "latent_channels": getattr(latent_format, "latent_channels", 4),
        "latent_dimensions": getattr(latent_format, "latent_dimensions", 2),
        "latent_rgb_factors": getattr(latent_format, "latent_rgb_factors", None),
        "latent_rgb_factors_bias": getattr(latent_format, "latent_rgb_factors_bias", None),
        "taesd_decoder_name": getattr(latent_format, "taesd_decoder_name", None),
    }


def deserialize_latent_format(latent_format_dict: Dict[str, Any]):
    """Reconstruct latent_format from scaffold.
    
    Creates exact replica of latent_format object from serialized properties.
    Used by DistributedModelWrapper to provide latent_format to ComfyUI.
    
    Args:
        latent_format_dict: Serialized latent format properties
    
    Returns:
        Reconstructed latent format object
    """
    import comfy.latent_formats
    
    class_name = latent_format_dict["class_name"]
    
    # Get class and instantiate
    if hasattr(comfy.latent_formats, class_name):
        latent_format_class = getattr(comfy.latent_formats, class_name)
        latent_format = latent_format_class()
    else:
        logging.warning(
            f"{LOG_PREFIX} [Scaffold] Unknown latent format class: {class_name}, "
            f"using LatentFormat base"
        )
        latent_format = comfy.latent_formats.LatentFormat()
    
    # Override properties if needed (for custom formats or validation)
    for key in ["scale_factor", "shift_factor", "latent_channels", 
                "latent_dimensions", "latent_rgb_factors", 
                "latent_rgb_factors_bias", "taesd_decoder_name"]:
        if key in latent_format_dict and latent_format_dict[key] is not None:
            setattr(latent_format, key, latent_format_dict[key])
    
    logging.debug(
        f"{LOG_PREFIX} [Scaffold] Deserialized latent_format: "
        f"{class_name} (channels={latent_format.latent_channels}, "
        f"dims={latent_format.latent_dimensions})"
    )
    
    return latent_format


def _estimate_model_size(state_dict: Dict) -> int:
    """Estimate total model size from state dict.
    
    Calculates unsharded model size for memory reporting.
    
    Args:
        state_dict: Model state dictionary
    
    Returns:
        Model size in bytes
    """
    total_bytes = 0
    for key, tensor in state_dict.items():
        if isinstance(tensor, torch.Tensor):
            total_bytes += tensor.numel() * tensor.element_size()
    return total_bytes
