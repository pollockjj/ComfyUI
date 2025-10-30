"""Model scaffold extraction for distributed loading.

Creates complete model structure on meta device (zero memory),
extracts ALL ComfyUI properties, then sends to workers for
FSDP weight loading.

Design Philosophy: "Copy at Perfect Information, Don't Reconstruct from Imperfect"
- Extract once at perfect information point (CPU/meta load)
- Send complete scaffold to workers for exact reconstruction
- No piecemeal property discovery via RPC errors
- Closed loop: all properties captured upfront

Based on WorkSplit deepclone pattern (ComfyUI core architect).
Reference: reference_worksplit_multigpu/comfy/model_patcher.py:351-374
"""

from __future__ import annotations
import torch
import logging
from typing import Dict, Any, Optional

LOG_PREFIX = "⚡ [Parallel-Attention]"


def extract_model_scaffold(checkpoint_path: str) -> tuple[Dict[str, Any], Dict]:
    """Extract complete model scaffold from checkpoint.
    
    This is the "perfect information" extraction point. We load the
    model once on CPU/meta device and capture EVERYTHING ComfyUI needs.
    Workers will reconstruct exact replicas from this scaffold.
    
    Args:
        checkpoint_path: Path to .safetensors checkpoint
    
    Returns:
        (scaffold, state_dict):
            scaffold: Complete model metadata (latent_format, dtype, etc.)
            state_dict: Model weights for FSDP loading
    
    Design Philosophy:
        - Extract once at perfect information (CPU load)
        - Don't reconstruct from imperfect information (RPC)
        - Workers get exact scaffold, not partial properties
        - Eliminates "discover via error" anti-pattern
    """
    import comfy.model_detection
    import comfy.utils
    
    logging.info(f"{LOG_PREFIX} [Scaffold] Extracting model scaffold from {checkpoint_path}")
    
    # Load checkpoint (ComfyUI standard method - USE existing infrastructure)
    state_dict = comfy.utils.load_torch_file(checkpoint_path)
    
    # CRITICAL: Follow ComfyUI's exact pattern from sd.py:1309-1313
    # 1. Get the prefix (e.g., "model.diffusion_model.")
    diffusion_model_prefix = comfy.model_detection.unet_prefix_from_state_dict(state_dict)
    
    # 2. STRIP the prefix from state dict keys (sd.py:1309)
    temp_sd = comfy.utils.state_dict_prefix_replace(state_dict, {diffusion_model_prefix: ""}, filter_keys=True)
    if len(temp_sd) > 0:
        state_dict = temp_sd
    
    # 3. Detect with EMPTY prefix (sd.py:1313)
    model_config = comfy.model_detection.model_config_from_unet(state_dict, "")
    
    if model_config is None:
        raise RuntimeError(
            f"{LOG_PREFIX} [Scaffold] Failed to detect model config from checkpoint. "
            f"Prefix used: '{diffusion_model_prefix}', keys after strip: {len(state_dict)}"
        )
    
    # Create model on meta device (zero memory allocation)
    # This gives us the complete structure without loading weights
    with torch.device('meta'):
        model = model_config.get_model(state_dict, "")
    
    # Extract ALL model properties (closed loop - perfect information)
    scaffold = {
        # Core properties
        "model_config_class": model_config.__class__.__name__,
        "model_config_dict": _serialize_model_config(model_config),
        
        # Model metadata
        "dtype": str(model.get_dtype()).split('.')[-1],  # 'torch.bfloat16' -> 'bfloat16'
        "model_type": model_config.unet_config.get("model_type", "unknown"),
        
        # Latent format (THE MISSING PIECE that caused the error)
        "latent_format": _serialize_latent_format(model.latent_format),
        
        # Conditioning (structure only, not runtime values)
        "is_adm": hasattr(model, 'is_adm') and callable(getattr(model, 'is_adm')) and model.is_adm(),
        # Don't call extra_conds() - requires actual conditioning data (pooled_output, etc.)
        # that we don't have on meta device. ComfyUI populates this at runtime during sampling.
        "extra_conds": {},
        
        # Device config
        "load_device": str(model.load_device),
        "offload_device": str(model.offload_device) if hasattr(model, 'offload_device') else 'cpu',
        
        # Model options
        "model_options": getattr(model, 'model_options', {}),
        
        # Memory estimates
        "model_size": _estimate_model_size(state_dict),
    }
    
    logging.info(
        f"{LOG_PREFIX} [Scaffold] Extracted complete scaffold: "
        f"type={scaffold['model_type']}, dtype={scaffold['dtype']}, "
        f"latent_format={scaffold['latent_format']['class_name']}, "
        f"size={scaffold['model_size'] / (1024**3):.2f}GB"
    )
    
    return scaffold, state_dict


def _serialize_model_config(model_config) -> Dict[str, Any]:
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
