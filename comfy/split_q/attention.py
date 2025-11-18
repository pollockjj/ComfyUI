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
    
    # Determine precision (copied from attention_basic)
    if attn_precision is None:
        attn_precision = q.dtype
    
    # Get dimensions
    if skip_reshape:
        b, _, _, dim_head = q.shape
    else:
        b, _, dim_head = q.shape
        dim_head //= heads
    
    scale = dim_head ** -0.5
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
    
    # CORE ATTENTION MATH (copied EXACTLY from attention_basic)
    # Step 1: QK^T * scale
    if attn_precision == torch.float32:
        sim = einsum('b i d, b j d -> b i j', q.float(), k.float()) * scale
    else:
        sim = einsum('b i d, b j d -> b i j', q, k) * scale
    
    del q, k  # Free memory
    
    # Handle mask (copied from attention_basic)
    if mask is not None:
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
    
    # Step 3: @ V
    out = einsum('b i j, b j d -> b i d', sim.to(v.dtype), v)
    
    # Reshape output (always use standard reshape, not skip_output_reshape)
    out = (
        out.unsqueeze(0)
        .reshape(b, heads, -1, dim_head)
        .permute(0, 2, 1, 3)
        .reshape(b, -1, heads * dim_head)
    )
    
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
    Phase 2: Async parallel execution with CUDA streams.
    
    Binary search mode: Set SPLIT_Q_NO_OVERLAP=1 env var to run streams sequentially (no overlap)
    for byte-exact comparison with blocking mode.
    
    Pipeline:
    - Stream 0: cuda:0 compute starts immediately
    - Stream 1: Replicate K/V → cuda:1 compute → gather (overlapped)
    
    Uses CUDA events for fine-grained synchronization.
    Falls back to blocking if async disabled via disable_async_mode().
    
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
    import os
    from .state import is_async_enabled
    
    # Check if async mode is still enabled (may be disabled by fallback)
    if not is_async_enabled():
        _logger.info(f"⚡ [split-q][attention] Async disabled, using blocking mode")
        return _parallel_attention_compute_blocking(state, q, k, v, heads, mask, **kwargs)
    
    # Binary search mode: no overlap (sequential streams for exact comparison)
    no_overlap = os.environ.get('SPLIT_Q_NO_OVERLAP', '0') == '1'
    if no_overlap:
        _logger.info(f"⚡ [split-q][attention][NO-OVERLAP] Sequential stream execution for binary search")
    
    # STEP 1: Split Q (runs on DEFAULT stream)
    # Detect sequence dimension: dim=1 for 3D [b, s, d], dim=2 for 4D [b, h, s, d]
    split_dim = 2 if q.ndim == 4 else 1
    q0, q1 = torch.tensor_split(q, 2, dim=split_dim)
    
    # *** PRIMARY FIX: Get default stream (producer of q, k, v) ***
    default_stream = torch.cuda.current_stream(state.device_0)
    
    # STEP 2: Launch Stream 1 replication (async, overlapped with Stream 0)
    with torch.cuda.stream(state.stream_1):
        # *** PRIMARY FIX: Wait for default stream to finish producing inputs ***
        state.stream_1.wait_stream(default_stream)
        k_1 = k.to(state.device_1, non_blocking=True)
        v_1 = v.to(state.device_1, non_blocking=True)
        q1_dev1 = q1.to(state.device_1, non_blocking=True)
        
        if mask is not None:
            mask_1 = mask.to(state.device_1, non_blocking=True)
        else:
            mask_1 = None
        
        # Record completion of all transfers
        state.event_k_replicated.record(state.stream_1)
        state.event_v_replicated.record(state.stream_1)
    
    # STEP 3: Launch Stream 0 compute (parallel with Stream 1 replication)
    with torch.cuda.stream(state.stream_0):
        # *** PRIMARY FIX: Wait for default stream to finish producing inputs ***
        state.stream_0.wait_stream(default_stream)
        
        out_0 = _attention_compute_kernel(state, q0, k, v, heads, mask, **kwargs)
        
        # NOTE: out_0 is already on stream_0, record_stream() unnecessary
        # The wait_event below ensures default stream waits before using out_0
        
        state.event_attn_0_done.record(state.stream_0)
    
    # NO OVERLAP MODE: Wait for stream 0 to finish before starting stream 1
    if no_overlap:
        state.stream_0.synchronize()
        _logger.info(f"⚡ [split-q][attention][NO-OVERLAP] Stream 0 synchronized before stream 1")
    
    # STEP 4: Stream 1 compute (waits for K/V replication to finish)
    with torch.cuda.stream(state.stream_1):
        state.stream_1.wait_event(state.event_k_replicated)
        state.stream_1.wait_event(state.event_v_replicated)
        
        out_1 = _attention_compute_kernel(state, q1_dev1, k_1, v_1, heads, mask_1, **kwargs)
        
        out_1_dev0 = out_1.to(state.device_0, non_blocking=True)
        
        # NOTE: Async transfer handled by stream_1, wait_event below ensures safety
        # record_stream() unnecessary and causes warnings
        
        state.event_attn_1_done.record(state.stream_1)
    
    # STEP 5: Synchronize for final concatenation
    # *** TERTIARY FIX: Efficient GPU-side sync (doesn't block CPU) ***
    # Make default stream (which torch.cat runs on) wait for compute streams
    default_stream.wait_event(state.event_attn_0_done)
    default_stream.wait_event(state.event_attn_1_done)
    
    # STEP 6: Concatenate results (runs on default stream, now safe)
    out = torch.cat([out_0, out_1_dev0], dim=1)
    
    # CRITICAL: Explicitly delete intermediate tensors to free VRAM
    # Without this, Python keeps references and VRAM leaks ~2GB per step
    del out_0, out_1_dev0, q0, q1, k_1, v_1, q1_dev1, out_1
    if mask is not None:
        del mask_1
    
    return out


def _parallel_attention_compute_blocking(state: SplitQState, q, k, v, heads, mask=None, **kwargs):
    """
    Phase 1.5 blocking fallback (used if async disabled).
    Identical to Phase 1.5 implementation.
    """
    _logger.info(f"⚡ [split-q][attention][BLOCKING] Entry: q.shape={q.shape}")
    
    q0, q1 = torch.tensor_split(q, 2, dim=1)
    
    k_1 = k.to(state.device_1, non_blocking=False)
    v_1 = v.to(state.device_1, non_blocking=False)
    
    if mask is not None:
        mask_1 = mask.to(state.device_1, non_blocking=False)
    else:
        mask_1 = None
    
    out_0 = _attention_compute_kernel(state, q0, k, v, heads, mask, **kwargs)
    
    q1_dev1 = q1.to(state.device_1, non_blocking=False)
    out_1 = _attention_compute_kernel(state, q1_dev1, k_1, v_1, heads, mask_1, **kwargs)
    
    torch.cuda.synchronize()
    
    out_1_dev0 = out_1.to(state.device_0, non_blocking=False)
    out = torch.cat([out_0, out_1_dev0], dim=1)
    
    _logger.info(f"⚡ [split-q][attention][BLOCKING] Complete: out.shape={out.shape}")
    
    return out


def _parallel_attention_compute_blocking(state: SplitQState, q, k, v, heads, mask=None, **kwargs):
    """
    Blocking fallback version (Phase 1.5 implementation).
    Used when async mode is disabled due to divergence detection.
    """
    _logger.info(f"⚡ [split-q][attention][BLOCKING] Entry: q.shape={q.shape}")
    
    q0, q1 = torch.tensor_split(q, 2, dim=1)
    
    k_1 = k.to(state.device_1, non_blocking=False)
    v_1 = v.to(state.device_1, non_blocking=False)
    q1_dev1 = q1.to(state.device_1, non_blocking=False)
    
    if mask is not None:
        mask_1 = mask.to(state.device_1, non_blocking=False)
    else:
        mask_1 = None
    
    out_0 = _attention_compute_kernel(state, q0, k, v, heads, mask, **kwargs)
    out_1 = _attention_compute_kernel(state, q1_dev1, k_1, v_1, heads, mask_1, **kwargs)
    
    torch.cuda.synchronize()
    
    out_1_dev0 = out_1.to(state.device_0, non_blocking=False)
    out = torch.cat([out_0, out_1_dev0], dim=1)
    
    _logger.info(f"⚡ [split-q][attention][BLOCKING] Complete: out.shape={out.shape}")
    return out
