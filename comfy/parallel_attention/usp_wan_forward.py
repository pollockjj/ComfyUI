"""Unified Sequence Parallel (USP) forward helpers for Wan video models.

Ported from Raylight's xdit_context_parallel.py to work with our xFuser USP stack.
Wan uses a single block type (unlike Flux's double/single split) with distinct RoPE format.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Optional

import torch
import torch.distributed as dist
from torch import Tensor

from xfuser.core.distributed import (
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
    get_sp_group,
)

from .usp_attention import usp_attention
from .dedup_logger import get_dedup_logger

LOG_PREFIX = "⚡ [Parallel-Attention][USP][Wan]"
LOGGER = logging.getLogger(__name__)
dedup_logger = get_dedup_logger(__name__)


def log_checkpoint_wan(step_name, rank, **tensors):
    """Log tensor checksums for byte-exact comparison.
    
    Args:
        step_name: Checkpoint identifier (e.g., "STEP7")
        rank: GPU rank (0 or 1)
        **tensors: Named tensors to checkpoint (e.g., q=q_tensor, k=k_tensor)
    """
    for name, tensor in tensors.items():
        if tensor is None:
            continue
        # Normalize: detach, cpu, fp32, numpy, sha256
        tensor_bytes = tensor.detach().cpu().to(torch.float32).numpy().tobytes()
        tensor_hash = hashlib.sha256(tensor_bytes).hexdigest()[:16]
        logging.info(
            f"⚡ [WAN-{step_name}-VERIFY][Rank {rank}] {name} - "
            f"hash: {tensor_hash}, shape: {tuple(tensor.shape)}, "
            f"dtype: {tensor.dtype}, device: {tensor.device}"
        )


def pad_if_odd(t: Tensor, dim: int = 1) -> Tensor:
    """Pad tensor along dimension if it has odd size (for even splitting)."""
    if t.size(dim) % 2 != 0:
        pad_shape = list(t.shape)
        pad_shape[dim] = 1
        pad_tensor = torch.zeros(pad_shape, dtype=t.dtype, device=t.device)
        t = torch.cat([t, pad_tensor], dim=dim)
    return t


def sinusoidal_embedding_1d(dim: int, position: Tensor) -> Tensor:
    """Generate 1D sinusoidal positional embeddings."""
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)
    
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half))
    )
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


def pad_freqs(original_tensor: Tensor, target_len: int) -> Tensor:
    """Pad frequency tensor to target length.
    
    Args:
        original_tensor: [B, L_global, 1, D/2, 2, 2] — full freq tensor
        target_len: Target sequence length
        
    Returns:
        Padded tensor [B, target_len, 1, D/2, 2, 2]
    """
    b, seq_len, z, dim, a, c = original_tensor.shape
    pad_size = target_len - seq_len
    if pad_size <= 0:
        return original_tensor
    
    padding_tensor = torch.ones(
        b, pad_size, z, dim, a, c,
        dtype=original_tensor.dtype,
        device=original_tensor.device,
    )
    padded_tensor = torch.cat([original_tensor, padding_tensor], dim=1)
    return padded_tensor


def apply_rope_sp(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    """Apply sequence-parallel RoPE to Q/K tensors.
    
    Wan-specific RoPE that slices frequency tensor by sequence-parallel rank.
    Different from Flux's apply_rope() - uses 6D frequency format.
    
    Args:
        xq, xk: [B, L_local, 1, D] — local sequence chunks
        freqs_cis: [B, L_global, 1, D/2, 2, 2] — full frequency tensor
        
    Returns:
        RoPE-applied (xq, xk) with same shapes as input
    """
    sp_rank = get_sequence_parallel_rank()
    sp_size = get_sequence_parallel_world_size()
    
    B, L_local, _, D = xq.shape
    L_global = L_local * sp_size
    
    # Ensure freqs_cis has length L_global (pad if needed)
    freqs_cis = pad_freqs(freqs_cis, L_global)
    
    # Slice the correct frequency chunk for this rank
    start = sp_rank * L_local
    end = start + L_local
    freqs_local = freqs_cis[:, start:end]  # [B, L_local, 1, D/2, 2, 2]
    
    # Prepare xq/xk for RoPE (split real/imag components)
    xq_ = xq.to(dtype=freqs_local.dtype).reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.to(dtype=freqs_local.dtype).reshape(*xk.shape[:-1], -1, 1, 2)
    
    # Apply RoPE using local frequencies
    xq_out = freqs_local[..., 0] * xq_[..., 0] + freqs_local[..., 1] * xq_[..., 1]
    xk_out = freqs_local[..., 0] * xk_[..., 0] + freqs_local[..., 1] * xk_[..., 1]
    
    return xq_out.reshape_as(xq).type_as(xq), xk_out.reshape_as(xk).type_as(xk)


def usp_self_attn_forward(
    self,
    x: Tensor,
    freqs: Optional[Tensor] = None,
    **kwargs,
) -> Tensor:
    """Wan self-attention forward with sequence parallelism.
    
    Args:
        x: [B, L_local, C] — local sequence chunk
        freqs: [B, L_global, 1, D/2, 2, 2] — RoPE frequencies
        
    Returns:
        [B, L_local, C] — attention output
    """
    b, s, _ = x.shape
    n, d = self.num_heads, self.head_dim
    
    # Query, key, value projections
    q = self.norm_q(self.q(x)).view(b, s, n, d)
    k = self.norm_k(self.k(x)).view(b, s, n, d)
    v = self.v(x).view(b, s, n * d)
    
    # Apply RoPE
    if freqs is not None:
        q, k = apply_rope_sp(q, k, freqs)
    
    # CHECKPOINT STEP7: Before attention (after RoPE)
    rank = dist.get_rank() if dist.is_initialized() else 0
    log_checkpoint_wan("STEP7", rank, q=q, k=k, v=v, freqs=freqs)
    
    # Distributed attention via xfuser
    q_flat = q.view(b, s, n * d)
    k_flat = k.view(b, s, n * d)
    
    from .usp_attention import _ATTENTION_HANDLE
    if _ATTENTION_HANDLE is None:
        raise RuntimeError("USP attention not initialized. Call initialize_usp_attention() first.")
    
    x = _ATTENTION_HANDLE(
        q_flat,
        k_flat,
        v,
        heads=self.num_heads,
    )
    x = x.flatten(2)
    x = self.o(x)
    return x


def usp_cross_attn_forward(
    self,
    x: Tensor,
    context: Tensor,
    context_img_len: Optional[int] = None,
    **kwargs,
) -> Tensor:
    """Wan cross-attention forward (text/image → video).
    
    Handles two modes:
    - T2V: Only text context
    - I2V: Text + image context (concatenated, separated by context_img_len)
    
    Args:
        x: [B, L_video, C] — video tokens (local chunk)
        context: [B, L_context, C] — text (+ optional image) tokens
        context_img_len: If not None, first N tokens are image features
        
    Returns:
        [B, L_video, C] — cross-attention output
    """
    from .usp_attention import _ATTENTION_HANDLE
    if _ATTENTION_HANDLE is None:
        raise RuntimeError("USP attention not initialized.")
    
    # I2V mode: separate image and text contexts
    if context_img_len is not None:
        context_img = context[:, :context_img_len]
        context_txt = context[:, context_img_len:]
        
        # Query projection (shared)
        q = self.norm_q(self.q(x))
        
        # Text branch
        k_txt = self.norm_k(self.k(context_txt))
        v_txt = self.v(context_txt)
        
        # Image branch
        k_img = self.norm_k_img(self.k_img(context_img))
        v_img = self.v_img(context_img)
        
        # Compute both attentions
        x_txt = _ATTENTION_HANDLE(q, k_txt, v_txt, heads=self.num_heads)
        x_img = _ATTENTION_HANDLE(q, k_img, v_img, heads=self.num_heads)
        
        x = x_txt + x_img
    else:
        # T2V mode: only text context
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(context))
        v = self.v(context)
        
        x = _ATTENTION_HANDLE(q, k, v, heads=self.num_heads)
    
    x = x.flatten(2)
    x = self.o(x)
    return x


def usp_block_forward(
    self,
    x: Tensor,
    e: Tensor,
    freqs: Optional[Tensor],
    context: Tensor,
    context_img_len: Optional[int] = None,
    **kwargs,
) -> Tensor:
    """Single Wan transformer block forward with USP.
    
    Block structure:
    1. Modulation (adaln)
    2. Self-attention (with RoPE)
    3. Cross-attention (text/image)
    4. FFN
    
    Args:
        x: [B, L_local, C] — video tokens (local sequence chunk)
        e: [B, 6, C] — time embedding (modulation parameters)
        freqs: [B, L_global, 1, D/2, 2, 2] — RoPE frequencies
        context: [B, L_context, C] — text (+ optional image) tokens
        context_img_len: If not None, enables I2V mode
        
    Returns:
        [B, L_local, C] — block output
    """
    # Modulation parameters
    scale_qkv, shift_qkv, gate_self_attn, scale_ff, shift_ff, gate_ff = e.chunk(6, dim=1)
    
    # Self-attention with modulation
    h = self.norm1(x) * (1 + scale_qkv) + shift_qkv
    h = self.self_attn(h, freqs=freqs)
    x = x + gate_self_attn * h
    
    # Cross-attention
    h = self.norm3(x)
    h = self.cross_attn(h, context=context, context_img_len=context_img_len)
    x = x + h
    
    # FFN with modulation
    h = self.norm2(x) * (1 + scale_ff) + shift_ff
    h = self.ffn(h)
    x = x + gate_ff * h
    
    return x


def usp_dit_forward(
    self,
    x: Tensor,
    t: Tensor,
    context: Tensor,
    clip_fea: Optional[Tensor] = None,
    freqs: Optional[Tensor] = None,
    transformer_options: dict = {},
    **kwargs,
) -> Tensor:
    """Wan DiT root forward with sequence parallelism.
    
    Args:
        x: [B, C_in, F, H, W] — input video latents
        t: [B] — timesteps
        context: [B, L_txt, C] — text embeddings
        clip_fea: [B, L_img, C] — optional image features (I2V mode)
        freqs: [B, F*H*W, 1, D/2, 2, 2] — RoPE frequencies
        transformer_options: Dict with patches_replace hooks
        
    Returns:
        [B, C_out, F, H, W] — denoised video
    """
    dedup_logger.info(
        "%s DiT forward called: x.shape=%s, t=%s",
        LOG_PREFIX, x.shape, t.shape
    )
    
    # Patch embedding: [B, C, F, H, W] → [B, C, F*H*W] → [B, F*H*W, C]
    x = self.patch_embedding(x.float()).to(x.dtype)
    grid_sizes = x.shape[2:]  # (F, H, W)
    x = x.flatten(2).transpose(1, 2)
    
    # Time embedding
    e = self.time_embedding(
        sinusoidal_embedding_1d(self.freq_dim, t).to(dtype=x.dtype)
    )
    e0 = self.time_projection(e).unflatten(1, (6, self.dim))
    
    # Text embedding
    context = self.text_embedding(context)
    
    # I2V mode: prepend image features
    context_img_len = None
    if clip_fea is not None:
        if self.img_emb is not None:
            context_clip = self.img_emb(clip_fea)  # [B, 257, C]
            context = torch.concat([context_clip, context], dim=1)
        context_img_len = clip_fea.shape[-2]
    
    # Sequence-parallel split: chunk along sequence dimension
    x = pad_if_odd(x, dim=1)
    x = torch.chunk(x, get_sequence_parallel_world_size(), dim=1)[get_sequence_parallel_rank()]
    
    # Process blocks (with optional patch hooks)
    patches_replace = transformer_options.get("patches_replace", {})
    blocks_replace = patches_replace.get("dit", {})
    
    for i, block in enumerate(self.blocks):
        if ("double_block", i) in blocks_replace:
            # Hook support for custom block replacements
            def block_wrap(args):
                out = {}
                out["img"] = block(
                    args["img"],
                    context=args["txt"],
                    e=args["vec"],
                    freqs=args["pe"],
                    context_img_len=context_img_len,
                )
                return out
            
            out = blocks_replace[("double_block", i)](
                {"img": x, "txt": context, "vec": e0, "pe": freqs},
                {"original_block": block_wrap},
            )
            x = out["img"]
        else:
            x = block(
                x,
                e=e0,
                freqs=freqs,
                context=context,
                context_img_len=context_img_len,
            )
    
    # Final projection
    x = self.head(x, e)
    
    # Gather sequence from all ranks
    x = get_sp_group().all_gather(x, dim=1)
    
    # Unpatchify: [B, F*H*W, C] → [B, C, F, H, W]
    x = self.unpatchify(x, grid_sizes)
    
    dedup_logger.info(
        "%s DiT forward complete: output.shape=%s",
        LOG_PREFIX, x.shape
    )
    
    return x


__all__ = [
    "apply_rope_sp",
    "usp_self_attn_forward",
    "usp_cross_attn_forward",
    "usp_block_forward",
    "usp_dit_forward",
]
