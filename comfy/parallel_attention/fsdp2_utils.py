"""
Universal FSDP2 utilities - apply to ALL models.

Design: Universal logic here, model-specific logic in policies.
"""

import torch
import logging

LOG_PREFIX = "⚡ [Parallel-Attention][FSDP2Utils]"


def detect_scalar_params(module):
    """Detect 0-dimensional scalar parameters that FSDP2 cannot shard.
    
    ComfyUI's FP8 ops create scalar parameters (e.g., scale_weight) that are
    0-dimensional tensors. FSDP2's fully_shard() cannot handle these and will
    raise ValueError. These must be excluded from sharding.
    
    
    Args:
        module: PyTorch module to scan
        
    Returns:
        set: Parameters with ndim == 0
    """
    ignored = set()
    scalar_names = []
    
    for name, param in module.named_parameters(recurse=True):
        if param.ndim == 0:
            ignored.add(param)
            scalar_names.append(name)
    
    if scalar_names:
        logging.debug(f"{LOG_PREFIX} Found {len(scalar_names)} scalar params: {scalar_names[:5]}")
    
    return ignored


def detect_dtype_mismatch(module, ref_dtype):
    """Detect parameters with dtype different from reference.
    
    Quantized layers (NF4, INT8, GGUF) and FP8 scaled layers have different
    dtypes than the main model. FSDP2 cannot shard mixed dtypes in the same
    unit. These parameters are excluded and kept replicated.
    
    
    Args:
        module: PyTorch module to scan
        ref_dtype: Reference dtype (e.g., torch.bfloat16)
        
    Returns:
        set: Parameters with dtype != ref_dtype
    """
    ignored = set()
    mismatch_names = []
    
    for name, param in module.named_parameters(recurse=True):
        if param.dtype != ref_dtype:
            ignored.add(param)
            mismatch_names.append(f"{name} ({param.dtype})")
    
    if mismatch_names:
        logging.debug(f"{LOG_PREFIX} Found {len(mismatch_names)} dtype mismatches: {mismatch_names[:5]}")
    
    return ignored


def detect_unshardable_params(module, ref_dtype=None):
    """Detect ALL parameters that cannot be sharded by FSDP2.
    
    Combines:
    1. Scalar detection (0-dim tensors from ComfyUI FP8 ops)
    2. Dtype mismatch detection (quantized/mixed-precision layers)
    
    This is a universal defensive pattern applied to ALL models.
    
    Args:
        module: PyTorch module to scan
        ref_dtype: Optional reference dtype for mismatch detection
        
    Returns:
        set: All unshardable parameters
    """
    ignored = set()
    
    # Pattern 1: Scalar parameters (ComfyUI FP8)
    scalars = detect_scalar_params(module)
    ignored.update(scalars)
    
    # Pattern 2: Dtype mismatches (quantized layers)
    if ref_dtype is not None:
        mismatches = detect_dtype_mismatch(module, ref_dtype)
        ignored.update(mismatches)
    
    if ignored:
        logging.debug(f"{LOG_PREFIX} Total unshardable params: {len(ignored)}")
    
    return ignored


def get_reference_dtype(module):
    """Get reference dtype from first 2D+ parameter in module.
    
    Skips scalars and 1D parameters (biases) to find the main weight dtype.
    
    Args:
        module: PyTorch module to scan
        
    Returns:
        torch.dtype or None: Reference dtype from first weight parameter
    """
    for name, param in module.named_parameters():
        if param.ndim >= 2:  # Skip scalars and biases
            logging.debug(f"{LOG_PREFIX} Reference dtype from {name}: {param.dtype}")
            return param.dtype
    
    logging.warning(f"{LOG_PREFIX} No reference dtype found (no 2D+ params)")
    return None
