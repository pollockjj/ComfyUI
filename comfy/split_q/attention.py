"""
Split-Q Parallel Attention Computation
Core logic for dual-GPU query splitting and parallel execution.
"""

import torch
import torch.nn.functional as F
from torch import einsum
import logging
from .state import SplitQState

_logger = logging.getLogger('comfy')


def _attention_compute_kernel(state, q, k, v, heads, mask=None, **kwargs):
    """
    Core attention computation kernel (single-device).
    Calls the original attention function to ensure correctness.
    
    This is a wrapper that delegates to state.original_attn_func,
    ensuring we use the exact same attention implementation as baseline.
    
    Args:
        state: SplitQState with original_attn_func reference
        q: Query tensor [batch, seq_len_split, dim] (ALREADY SPLIT)
        k: Key tensor [batch, seq_len_full, dim] (FULL, replicated)
        v: Value tensor [batch, seq_len_full, dim] (FULL, replicated)
        heads: Number of attention heads
        mask: Optional attention mask
        **kwargs: Pass-through to original function
        
    Returns:
        Attention output [batch, seq_len_split, heads*dim_head]
    """
    # CRITICAL: Call the ORIGINAL attention function to ensure byte-accuracy
    # Do NOT reimplement - just call what works
    return state.original_attn_func(q, k, v, heads, mask, **kwargs)


def _serial_attention_compute(state: SplitQState, q, k, v, heads, mask=None, **kwargs):
    """
    Wrapper to call original attention function for validation.
    Used in validation_mode to compare against parallel execution.
    
    Args:
        state: SplitQState instance
        q, k, v, heads, mask: Attention parameters
        **kwargs: Pass-through arguments
        
    Returns:
        Attention output from original function
    """
    return state.original_attn_func(q, k, v, heads, mask, **kwargs)


def _parallel_attention_compute(state: SplitQState, q, k, v, heads, mask=None, **kwargs):
    """
    Phase 1: Blocking parallel execution (Blueprint Section 8.1).
    
    Strategy:
    1. Split Q along sequence dimension (dim=1)
    2. Replicate K/V to cuda:1 (blocking for Phase 1)
    3. Compute attention on both GPUs (sequential for Phase 1)
    4. Gather results back to cuda:0
    5. Use torch.cuda.synchronize() for correctness
    
    Phase 2 will replace blocking operations with async streams per Blueprint Table 1.
    
    Args:
        state: SplitQState with CUDA streams/events
        q: Query [batch, seq_len, dim]
        k: Key [batch, seq_len, dim]
        v: Value [batch, seq_len, dim]
        heads: Number of attention heads
        mask: Optional attention mask
        **kwargs: Additional arguments (unused in Phase 1)
        
    Returns:
        Attention output [batch, seq_len, heads*dim_head]
    """
    
    # CHECKPOINT: Log entry
    _logger.info(f"⚡ [split-q][attention] Parallel compute entry: q.shape={q.shape}, device={q.device}")
    
    # STEP 1: Split Q along sequence dimension (Blueprint Section 5.1)
    # q.shape is [batch, 8192, dim] -> q0=[batch, 4096, dim], q1=[batch, 4096, dim]
    q0, q1 = torch.tensor_split(q, 2, dim=1)
    _logger.info(f"⚡ [split-q][attention] Q split: q0.shape={q0.shape}, q1.shape={q1.shape}")
    
    # STEP 2: Replicate K/V to cuda:1 (BLOCKING in Phase 1)
    # Blueprint Section 4.1: Always replicate FULL K/V for both self and cross-attention
    _logger.info(f"⚡ [split-q][attention] Replicating K/V to cuda:1 (blocking)")
    k_1 = k.to(state.device_1, non_blocking=False)  # Blocking in Phase 1
    v_1 = v.to(state.device_1, non_blocking=False)
    
    # Handle mask replication if present
    if mask is not None:
        mask_1 = mask.to(state.device_1, non_blocking=False)
    else:
        mask_1 = None
    
    # STEP 3: Compute attention on cuda:0 (BLOCKING in Phase 1)
    _logger.info(f"⚡ [split-q][attention] Computing attention on cuda:0")
    out_0 = _attention_compute_kernel(state, q0, k, v, heads, mask, **kwargs)
    
    # STEP 4: Compute attention on cuda:1 (BLOCKING in Phase 1)
    _logger.info(f"⚡ [split-q][attention] Computing attention on cuda:1")
    q1_dev1 = q1.to(state.device_1, non_blocking=False)  # Move q1 to cuda:1
    out_1 = _attention_compute_kernel(state, q1_dev1, k_1, v_1, heads, mask_1, **kwargs)
    
    # STEP 5: Synchronize (BLOCKING in Phase 1)
    # Blueprint Section 8.1: Use torch.cuda.synchronize() for correctness
    # Phase 2 will replace this with event-based synchronization
    torch.cuda.synchronize()
    _logger.info(f"⚡ [split-q][attention] Synchronization complete")
    
    # STEP 6: Gather results back to cuda:0
    out_1_dev0 = out_1.to(state.device_0, non_blocking=False)  # Blocking gather
    out = torch.cat([out_0, out_1_dev0], dim=1)
    
    _logger.info(f"⚡ [split-q][attention] Gather complete: out.shape={out.shape}, device={out.device}")
    
    return out
