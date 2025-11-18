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


def _attention_compute_kernel(state, q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False, **kwargs):
    """
    Core attention computation kernel (single-device).
    Phase 1.5: Direct implementation of attention_basic math for stream compatibility.
    
    Reimplements the core QK^T @ V operations to be stream-aware. This is necessary
    because calling state.original_attn_func from within a CUDA stream context
    doesn't respect the stream (operations get scheduled on default stream).
    
    Math copied EXACTLY from comfy/ldm/modules/attention.py::attention_basic.
    
    Args:
        state: SplitQState (unused in Phase 1.5, kept for compatibility)
        q: Query tensor [batch, seq_len_split, dim] (ALREADY SPLIT)
        k: Key tensor [batch, seq_len_full, dim] (FULL, replicated)
        v: Value tensor [batch, seq_len_full, dim] (FULL, replicated)
        heads: Number of attention heads
        mask: Optional attention mask
        attn_precision: Precision override (None = use tensor dtype)
        skip_reshape: If True, tensors already in [b*h, seq, dim_head] format
        **kwargs: Ignored (for compatibility)
        
    Returns:
        Attention output [batch, seq_len_split, heads*dim_head]
    """
    from torch import einsum
    
    # CHECKPOINT 1: Input shapes
    _logger.info(f"⚡ [split-q][kernel][CHECKPOINT-1] Input: q={q.shape}, k={k.shape}, v={v.shape}, heads={heads}, device={q.device}")
    
    # Determine precision (copied from attention_basic)
    if attn_precision is None:
        attn_precision = q.dtype
    
    # Get dimensions
    if skip_reshape:
        b, _, _, dim_head = q.shape
        _logger.info(f"⚡ [split-q][kernel][CHECKPOINT-2] skip_reshape=True, b={b}, dim_head={dim_head}")
    else:
        b, _, dim_head = q.shape
        dim_head //= heads
        _logger.info(f"⚡ [split-q][kernel][CHECKPOINT-2] skip_reshape=False, b={b}, calculated dim_head={dim_head}")
    
    scale = dim_head ** -0.5
    _logger.info(f"⚡ [split-q][kernel][CHECKPOINT-3] scale={scale}")
    
    h = heads
    
    # Reshape tensors (copied EXACTLY from attention_basic)
    if skip_reshape:
        q, k, v = map(
            lambda t: t.reshape(b * heads, -1, dim_head),
            (q, k, v),
        )
    else:
        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(b, -1, heads, dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * heads, -1, dim_head)
            .contiguous(),
            (q, k, v),
        )
    
    _logger.info(f"⚡ [split-q][kernel][CHECKPOINT-4] After reshape: q={q.shape}, k={k.shape}, v={v.shape}")
    
    # CORE ATTENTION MATH (copied EXACTLY from attention_basic)
    # Step 1: QK^T * scale
    if attn_precision == torch.float32:
        sim = einsum('b i d, b j d -> b i j', q.float(), k.float()) * scale
    else:
        sim = einsum('b i d, b j d -> b i j', q, k) * scale
    
    _logger.info(f"⚡ [split-q][kernel][CHECKPOINT-5] After QK^T: sim={sim.shape}, dtype={sim.dtype}")
    
    del q, k  # Free memory
    
    # Handle mask (copied from attention_basic)
    if mask is not None:
        _logger.info(f"⚡ [split-q][kernel][CHECKPOINT-6] Applying mask: mask.shape={mask.shape}")
        if mask.dtype == torch.bool:
            from einops import rearrange, repeat
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)
        else:
            if len(mask.shape) == 2:
                bs = 1
            else:
                bs = mask.shape[0]
            mask = mask.reshape(bs, -1, mask.shape[-2], mask.shape[-1]).expand(b, heads, -1, -1).reshape(-1, mask.shape[-2], mask.shape[-1])
            sim.add_(mask)
    
    # Step 2: Softmax
    sim = sim.softmax(dim=-1)
    _logger.info(f"⚡ [split-q][kernel][CHECKPOINT-7] After softmax: sim={sim.shape}")
    
    # Step 3: @ V
    out = einsum('b i j, b j d -> b i d', sim.to(v.dtype), v)
    _logger.info(f"⚡ [split-q][kernel][CHECKPOINT-8] After @V: out={out.shape}")
    
    # Reshape output (always use standard reshape, not skip_output_reshape)
    out = (
        out.unsqueeze(0)
        .reshape(b, heads, -1, dim_head)
        .permute(0, 2, 1, 3)
        .reshape(b, -1, heads * dim_head)
    )
    
    _logger.info(f"⚡ [split-q][kernel][CHECKPOINT-9] Final output: out={out.shape}, device={out.device}")
    
    return out


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
    Phase 1: Blocking parallel execution (FINAL - async not viable).
    
    Strategy:
    1. Split Q along sequence dimension (dim=1)
    2. Replicate K/V to cuda:1 (blocking)
    3. Compute attention on both GPUs (sequential due to original_attn_func limitations)
    4. Gather results back to cuda:0
    5. Use torch.cuda.synchronize() for correctness
    
    Note: Attempted Phase 2 async streams but original_attn_func is not stream-aware.
    Calling it from within a CUDA stream context produces corrupted output because
    the wrapped attention function doesn't respect the stream context. Would require
    reimplementing attention math (rejected due to correctness risk) or deeper hooks.
    
    Args:
        state: SplitQState with CUDA streams/events
        q: Query [batch, seq_len, dim]
        k: Key [batch, seq_len, dim]
        v: Value [batch, seq_len, dim]
        heads: Number of attention heads
        mask: Optional attention mask
        **kwargs: Additional arguments
        
    Returns:
        Attention output [batch, seq_len, heads*dim_head]
    """
    
    # CHECKPOINT: Log entry
    _logger.info(f"⚡ [split-q][attention] Parallel compute entry: q.shape={q.shape}, device={q.device}")
    
    # STEP 1: Split Q along sequence dimension
    q0, q1 = torch.tensor_split(q, 2, dim=1)
    _logger.info(f"⚡ [split-q][attention] Q split: q0.shape={q0.shape}, q1.shape={q1.shape}")
    
    # STEP 2: Replicate K/V to cuda:1 (BLOCKING)
    _logger.info(f"⚡ [split-q][attention] Replicating K/V to cuda:1 (blocking)")
    k_1 = k.to(state.device_1, non_blocking=False)
    v_1 = v.to(state.device_1, non_blocking=False)
    
    # Handle mask replication if present
    if mask is not None:
        mask_1 = mask.to(state.device_1, non_blocking=False)
    else:
        mask_1 = None
    
    # STEP 3: Compute attention on cuda:0 (BLOCKING)
    _logger.info(f"⚡ [split-q][attention] Computing attention on cuda:0")
    out_0 = _attention_compute_kernel(state, q0, k, v, heads, mask, **kwargs)
    
    # STEP 4: Compute attention on cuda:1 (BLOCKING)
    _logger.info(f"⚡ [split-q][attention] Computing attention on cuda:1")
    q1_dev1 = q1.to(state.device_1, non_blocking=False)
    out_1 = _attention_compute_kernel(state, q1_dev1, k_1, v_1, heads, mask_1, **kwargs)
    
    # STEP 5: Synchronize (BLOCKING)
    torch.cuda.synchronize()
    _logger.info(f"⚡ [split-q][attention] Synchronization complete")
    
    # STEP 6: Gather results back to cuda:0
    out_1_dev0 = out_1.to(state.device_0, non_blocking=False)
    out = torch.cat([out_0, out_1_dev0], dim=1)
    
    _logger.info(f"⚡ [split-q][attention] Gather complete: out.shape={out.shape}, device={out.device}")
    
    return out
