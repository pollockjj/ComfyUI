"""USP-aware single block forward method (Raylight pattern).

This module provides a replacement forward method for Flux SingleStreamBlock
that uses xFuser attention backend for distributed inference.
"""

import torch
from torch import Tensor
from comfy.ldm.flux.layers import apply_mod
from comfy.ldm.flux.math import apply_rope
from comfy.parallel_attention.xfuser_attention import xfuser_optimized_attention


def usp_single_stream_forward(
    self,
    x: Tensor,
    vec: Tensor,
    pe: Tensor,
    attn_mask=None,
    modulation_dims=None,
    **kwargs
) -> Tensor:
    """USP-aware single block forward (Raylight exact pattern).
    
    This function replaces the standard SingleStreamBlock.forward() when
    USP (Ulysses/Ring) is enabled. It uses xFuser's attention backend
    instead of standard PyTorch attention.
    
    Args:
        self: SingleStreamBlock instance
        x: Input tensor [B, S, D]
        vec: Vector embedding [B, D]
        pe: Positional encoding
        attn_mask: Attention mask (unused)
        modulation_dims: Modulation dimensions
        
    Returns:
        Output tensor [B, S, D]
    """
    # Modulation
    mod, _ = self.modulation(vec)
    
    # Linear projection with modulation, then split into QKV and MLP
    qkv, mlp = torch.split(
        self.linear1(
            apply_mod(
                self.pre_norm(x),
                (1 + mod.scale),
                mod.shift,
                modulation_dims
            )
        ),
        [3 * self.hidden_size, self.mlp_hidden_dim],
        dim=-1
    )
    
    # Reshape to multi-head: [B, S, 3*H*D] → [3, B, H, S, D]
    q, k, v = qkv.view(
        qkv.shape[0],
        qkv.shape[1],
        3,
        self.num_heads,
        -1
    ).permute(2, 0, 3, 1, 4)
    
    # QK normalization
    q, k = self.norm(q, k, v)
    
    # Apply RoPE positional encoding
    if pe is not None:
        q, k = apply_rope(q, k, pe)
    
    # xFuser attention (expects [B, H, S, D] format)
    attn = xfuser_optimized_attention(
        q, k, v,
        self.num_heads,
        skip_reshape=True,  # Input already in [B, H, S, D]
        skip_output_reshape=False  # Output as [B, S, H*D] for concat
    )
    
    # Debug: Log shapes to understand the dimension mismatch
    import logging
    logging.debug(f"⚡ [USP SingleBlock] attn.shape={attn.shape}, mlp.shape={mlp.shape}")
    
    # MLP stream
    mlp_out = self.mlp_act(mlp)
    
    # Combine attention and MLP outputs
    output = self.linear2(torch.cat((attn, mlp_out), 2))
    
    # Apply gate modulation and residual connection
    x += apply_mod(output, mod.gate, None, modulation_dims)
    
    # FP16 safety
    if x.dtype == torch.float16:
        x = torch.nan_to_num(x, nan=0.0, posinf=65504, neginf=-65504)
    
    return x
