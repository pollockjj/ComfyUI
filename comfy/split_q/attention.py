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


def _attention_compute_kernel(q, k, v, heads, mask=None):
    """
    Core attention computation kernel (single-device).
    Re-implements attention_split logic to operate on split Q tensors.
    
    This is the actual math: Q @ K^T @ V with softmax.
    Called twice per attention block (once on cuda:0, once on cuda:1).
    
    Args:
        q: Query tensor [batch, seq_len_split, dim] (ALREADY SPLIT)
        k: Key tensor [batch, seq_len_full, dim] (FULL, replicated)
        v: Value tensor [batch, seq_len_full, dim] (FULL, replicated)
        heads: Number of attention heads
        mask: Optional attention mask
        
    Returns:
        Attention output [batch, seq_len_split, dim]
    """
    batch, seq_len, dim = q.shape
    dim_head = dim // heads
    
    # Reshape to multi-head format: [batch*heads, seq_len, dim_head]
    q = q.reshape(batch * heads, seq_len, dim_head)
    k = k.reshape(batch * heads, k.shape[1], dim_head)  # k.shape[1] is FULL seq_len
    v = v.reshape(batch * heads, v.shape[1], dim_head)
    
    # Compute attention with memory-efficient slicing (from attention_split)
    # This slices Q to avoid OOM, not for parallelism
    out = torch.zeros_like(q)
    scale = dim_head ** -0.5
    slice_size = seq_len  # Use full split size (Phase 1 simplification)
    
    for i in range(0, seq_len, slice_size):
        # Q @ K^T
        s1 = einsum('b i d, b j d -> b i j', q[:, i:i+slice_size], k) * scale
        
        # Apply mask if provided
        if mask is not None:
            if len(mask.shape) == 2:
                s1 += mask[i:i+slice_size]
            else:
                if mask.shape[1] == 1:
                    s1 += mask
                else:
                    s1 += mask[:, i:i+slice_size]
        
        # Softmax + V
        s2 = F.softmax(s1, dim=-1)
        out[:, i:i+slice_size] = einsum('b i j, b j d -> b i d', s2, v)
        
        del s1, s2
    
    # Reshape back to [batch, seq_len, dim]
    return out.reshape(batch, seq_len, dim)


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
    out_0 = _attention_compute_kernel(q0, k, v, heads, mask)
    
    # STEP 4: Compute attention on cuda:1 (BLOCKING in Phase 1)
    _logger.info(f"⚡ [split-q][attention] Computing attention on cuda:1")
    q1_dev1 = q1.to(state.device_1, non_blocking=False)  # Move q1 to cuda:1
    out_1 = _attention_compute_kernel(q1_dev1, k_1, v_1, heads, mask_1)
    
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
