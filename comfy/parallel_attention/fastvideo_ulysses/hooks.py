"""FastVideo-style Ulysses forward hooks for Flux DiT model.

Simplified from USP hooks - no ring logic, pure Ulysses only.
"""

import logging
import torch
from torch import Tensor
from typing import Optional

from comfy.ldm.flux.layers import timestep_embedding
from comfy.parallel_attention.fastvideo_ulysses.communicator import (
    get_sp_rank,
    get_sp_world_size,
)
from comfy.parallel_attention.fastvideo_ulysses.attention import UlyssesAttention

LOG_PREFIX = "⚡ [FastVideo-Ulysses][Hooks]"
logger = logging.getLogger(__name__)


# Global attention instance (created on first use)
_ULYSSES_ATTENTION: Optional[UlyssesAttention] = None


def _get_or_create_attention(num_heads: int, head_dim: int) -> UlyssesAttention:
    """Lazy-init global attention instance."""
    global _ULYSSES_ATTENTION
    if _ULYSSES_ATTENTION is None:
        _ULYSSES_ATTENTION = UlyssesAttention(num_heads=num_heads, head_dim=head_dim)
    return _ULYSSES_ATTENTION


def pad_if_odd(t: Tensor, dim: int = 1) -> Tensor:
    """Pad tensor if dimension is odd (required for chunking)."""
    if t.size(dim) % 2 != 0:
        pad_shape = list(t.shape)
        pad_shape[dim] = 1
        pad_tensor = torch.zeros(pad_shape, dtype=t.dtype, device=t.device)
        t = torch.cat([t, pad_tensor], dim=dim)
    return t


def ulysses_dit_forward(
    self,
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    timesteps: Tensor,
    y: Tensor,
    guidance: Optional[Tensor] = None,
    control=None,
    transformer_options: Optional[dict] = None,
    attn_mask: Optional[Tensor] = None,
) -> Tensor:
    """DiT forward with pure Ulysses parallelism.
    
    Key difference from USP: Only image tokens are sequence-parallel, text tokens are replicated.
    """
    transformer_options = transformer_options or {}
    patches_replace = transformer_options.get("patches_replace", {})
    
    # Pad sequences if needed
    img = pad_if_odd(img, dim=1)
    if img_ids is not None:
        img_ids = pad_if_odd(img_ids, dim=1)
    txt = pad_if_odd(txt, dim=1)
    if txt_ids is not None:
        txt_ids = pad_if_odd(txt_ids, dim=1)
    
    # Prepare y vector
    if y is None:
        y = torch.zeros((img.shape[0], self.params.vec_in_dim), device=img.device, dtype=img.dtype)
    
    # Input projections
    img = self.img_in(img)
    vec = self.time_in(timestep_embedding(timesteps, 256).to(img.dtype))
    if self.params.guidance_embed and guidance is not None:
        vec = vec + self.guidance_in(timestep_embedding(guidance, 256).to(img.dtype))
    vec = vec + self.vector_in(y[:, : self.params.vec_in_dim])
    txt = self.txt_in(txt)
    
    # Positional embeddings
    if img_ids is not None:
        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)
    else:
        pe = None
    
    # Chunk image tokens for sequence parallelism
    # Text tokens stay replicated (full copy on each rank)
    sp_rank = get_sp_rank()
    sp_size = get_sp_world_size()
    
    img = torch.chunk(img, sp_size, dim=1)[sp_rank]
    txt = txt  # Replicated, no chunking
    
    if pe is not None:
        # Split PE for both image and text
        img_pe_len = img_ids.shape[1]
        txt_pe_len = txt_ids.shape[1]
        pe_txt = pe[:, :txt_pe_len]
        pe_img = pe[:, txt_pe_len:]
        pe_img = torch.chunk(pe_img, sp_size, dim=1)[sp_rank]
        pe = torch.cat([pe_txt, pe_img], dim=1)
    
    # Process through double blocks
    blocks_replace = patches_replace.get("dit", {})
    for i, block in enumerate(self.double_blocks):
        block_key = ("double_block", i)
        
        if block_key in blocks_replace:
            def block_wrap(args):
                out = {}
                out["img"], out["txt"] = block(
                    img=args["img"],
                    txt=args["txt"],
                    vec=args["vec"],
                    pe=args["pe"]
                )
                return out
            
            out = patches_replace[block_key](
                {"img": img, "txt": txt, "vec": vec, "pe": pe},
                {"original_block": block_wrap}
            )
            txt = out["txt"]
            img = out["img"]
        else:
            img, txt = block(img=img, txt=txt, vec=vec, pe=pe)
    
    # Concatenate for single blocks
    img = torch.cat((txt, img), 1)
    
    # Process through single blocks
    for i, block in enumerate(self.single_blocks):
        block_key = ("single_block", i)
        
        if block_key in blocks_replace:
            def block_wrap(args):
                out = {}
                out["img"] = block(args["img"], vec=args["vec"], pe=args["pe"])
                return out
            
            out = patches_replace[block_key](
                {"img": img, "vec": vec, "pe": pe},
                {"original_block": block_wrap}
            )
            img = out["img"]
        else:
            img = block(img, vec=vec, pe=pe)
    
    # Separate text and image
    txt_len = txt.shape[1]
    img = img[:, txt_len:]
    
    # Final layer norm and projection
    img = self.final_layer(img, vec)
    
    # All-gather image tokens back to full sequence
    from comfy.parallel_attention.fastvideo_ulysses.communicator import all_gather_nd
    img = all_gather_nd(img.contiguous(), dim=1)
    
    return img


def ulysses_double_block_forward(
    self,
    img: Tensor,
    txt: Tensor,
    vec: Tensor,
    pe: Tensor,
    attn_mask: Optional[Tensor] = None,
):
    """Double block forward with Ulysses attention.
    
    Image tokens: sequence-parallel
    Text tokens: replicated
    """
    img_mod1, img_mod2 = self.img_mod(vec)
    txt_mod1, txt_mod2 = self.txt_mod(vec)
    
    # Modulate
    img_modulated = self.img_norm1(img)
    img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
    img_q = self.img_attn.qkv(img_modulated)
    
    txt_modulated = self.txt_norm1(txt)
    txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
    txt_q = self.txt_attn.qkv(txt_modulated)
    
    # Reshape for attention [batch, seq, heads, dim]
    num_heads = self.img_attn.num_heads
    head_dim = self.img_attn.head_dim
    
    # Split QKV
    img_qkv = img_q.reshape(img.shape[0], img.shape[1], 3, num_heads, head_dim)
    img_q, img_k, img_v = img_qkv.permute(2, 0, 1, 3, 4).unbind(0)
    
    txt_qkv = txt_q.reshape(txt.shape[0], txt.shape[1], 3, num_heads, head_dim)
    txt_q, txt_k, txt_v = txt_qkv.permute(2, 0, 1, 3, 4).unbind(0)
    
    # Apply RoPE to image tokens
    if pe is not None:
        img_q, img_k = self.img_attn.norm(img_q, img_k, img_v, pe)
    
    # Ulysses attention: image is distributed, text is replicated
    attn = _get_or_create_attention(num_heads, head_dim)
    img_attn_out, txt_attn_out = attn.forward(
        q=img_q, k=img_k, v=img_v,
        replicated_q=txt_q, replicated_k=txt_k, replicated_v=txt_v
    )
    
    # Reshape back
    img_attn_out = img_attn_out.reshape(img.shape[0], img.shape[1], -1)
    txt_attn_out = txt_attn_out.reshape(txt.shape[0], txt.shape[1], -1)
    
    # Project and modulate
    img = img + img_mod1.gate * self.img_attn.proj(img_attn_out)
    txt = txt + txt_mod1.gate * self.txt_attn.proj(txt_attn_out)
    
    # FFN
    img = img + img_mod2.gate * self.img_mlp((1 + img_mod2.scale) * self.img_norm2(img) + img_mod2.shift)
    txt = txt + txt_mod2.gate * self.txt_mlp((1 + txt_mod2.scale) * self.txt_norm2(txt) + txt_mod2.shift)
    
    return img, txt


def ulysses_single_block_forward(
    self,
    x: Tensor,
    vec: Tensor,
    pe: Tensor,
    attn_mask: Optional[Tensor] = None,
):
    """Single block forward with Ulysses attention.
    
    All tokens are treated as sequence-parallel (no text/image separation at this stage).
    """
    mod, _ = self.modulation(vec)
    x_mod = (1 + mod.scale) * self.pre_norm(x) + mod.shift
    qkv = self.linear1(x_mod)
    
    # Reshape for attention
    num_heads = self.num_heads
    head_dim = self.head_dim
    
    qkv = qkv.reshape(x.shape[0], x.shape[1], 3, num_heads, head_dim)
    q, k, v = qkv.permute(2, 0, 1, 3, 4).unbind(0)
    
    # Apply RoPE
    if pe is not None:
        q, k = self.norm(q, k, v, pe)
    
    # Ulysses attention (all distributed, no replicated)
    attn = _get_or_create_attention(num_heads, head_dim)
    attn_out, _ = attn.forward(q=q, k=k, v=v)
    
    # Reshape and project
    attn_out = attn_out.reshape(x.shape[0], x.shape[1], -1)
    x = x + mod.gate * self.linear2(attn_out)
    
    return x
