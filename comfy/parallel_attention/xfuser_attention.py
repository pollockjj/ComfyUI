"""xFuser attention backend wrapper (Raylight pattern).

This module provides a global singleton xFuser attention instance
that handles YunChang ring-attention kernels for USP inference.
"""

import torch
from typing import Optional

# Global xFuser attention instance (created at initialization)
_xfuser_attn_instance: Optional[object] = None

def initialize_xfuser_attention(attn_type: str = "flash_attn"):
    """Initialize global xFuser attention backend.
    
    Args:
        attn_type: Attention backend type ("flash_attn", "sage_attn", "torch")
    """
    global _xfuser_attn_instance
    
    from xfuser.core.long_ctx_attention import xFuserLongContextAttention
    from yunchang.kernels import AttnType
    
    attn_map = {
        "flash_attn": AttnType.FA,
        "sage_attn": AttnType.SAGE_AUTO,
        "torch": AttnType.TORCH,
    }
    
    if attn_type not in attn_map:
        raise ValueError(f"Unknown attention type: {attn_type}. Valid: {list(attn_map.keys())}")
    
    _xfuser_attn_instance = xFuserLongContextAttention(attn_type=attn_map[attn_type])
    print(f"⚡ [xFuser] Initialized {attn_type} attention backend")


def xfuser_optimized_attention(
    q: torch.Tensor,
    k: torch.Tensor, 
    v: torch.Tensor,
    heads: int,
    skip_reshape: bool = True,
    skip_output_reshape: bool = False
) -> torch.Tensor:
    """Optimized attention using xFuser backend (Raylight exact pattern).
    
    Args:
        q: Query tensor [B, H, S, D] if skip_reshape=True, else [B, S, H*D]
        k: Key tensor [B, H, S, D] if skip_reshape=True, else [B, S, H*D]
        v: Value tensor [B, H, S, D] if skip_reshape=True, else [B, S, H*D]
        heads: Number of attention heads
        skip_reshape: If True, input already in [B, H, S, D] format
        skip_output_reshape: If True, keep output in [B, H, S, D] format
        
    Returns:
        Attention output tensor
    """
    if _xfuser_attn_instance is None:
        raise RuntimeError("xFuser attention not initialized. Call initialize_xfuser_attention() first.")
    
    # Get dimensions based on input format
    if skip_reshape:
        b, _, _, dim_head = q.shape  # [B, H, S, D]
    else:
        b, _, dim_head = q.shape  # [B, S, H*D]
        dim_head //= heads
        # Reshape: [B, S, H*D] → [B, H, S, D]
        q = q.view(b, -1, heads, dim_head).transpose(1, 2)
        k = k.view(b, -1, heads, dim_head).transpose(1, 2)
        v = v.view(b, -1, heads, dim_head).transpose(1, 2)
    
    # Debug: Log tensor shapes going INTO xfuser
    print(f"⚡ [xFuser] INPUT: q.shape={q.shape}, k.shape={k.shape}, v.shape={v.shape}")
    print(f"⚡ [xFuser] Transposing to [B,S,H,D]: q.transpose(1,2).shape={q.transpose(1,2).shape}")
    
    # Debug: Log shapes before xfuser call
    import logging
    logging.info(f"⚡ [xFuser] Before transpose: q.shape={q.shape}, k.shape={k.shape}, v.shape={v.shape}")
    q_transposed = q.transpose(1, 2)
    k_transposed = k.transpose(1, 2)
    v_transposed = v.transpose(1, 2)
    logging.info(f"⚡ [xFuser] After transpose: q.shape={q_transposed.shape}, k.shape={k_transposed.shape}, v.shape={v_transposed.shape}")
    
    # xFuser expects [B, S, H, D] format (sequence-first)
    # Transpose: [B, H, S, D] → [B, S, H, D]
    out = _xfuser_attn_instance(
        None,  # attn_mask (unused by xFuser for Flux)
        q_transposed,
        k_transposed,
        v_transposed,
    ).transpose(1, 2)  # [B, S, H, D] → [B, H, S, D]
    
    logging.info(f"⚡ [xFuser] Output shape: {out.shape}")
    
    print(f"⚡ [xFuser] OUTPUT: out.shape={out.shape}")
    
    # Reshape output if requested (Raylight exact pattern)
    if not skip_output_reshape:
        # out is currently [B, H, S, D]
        # Need [B, S, H*D] for output
        out = out.transpose(1, 2).reshape(b, -1, heads * dim_head)
    
    return out
