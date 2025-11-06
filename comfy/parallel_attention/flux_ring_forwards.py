"""Ring-Attention forward methods for Flux (worker installation).

Based on Raylight's xdit_context_parallel.py approach.
These functions replace the forward() methods on DoubleStreamBlock and SingleStreamBlock.
"""

import torch
from torch import Tensor

from comfy.parallel_attention.raylight_attention import attention as raylight_attention


def pad_if_odd(t: torch.Tensor, dim: int = 1):
    """Pad tensor if dimension is odd (for sequence splitting)."""
    if t.size(dim) % 2 != 0:
        pad_shape = list(t.shape)
        pad_shape[dim] = 1
        pad_tensor = torch.zeros(pad_shape, dtype=t.dtype, device=t.device)
        t = torch.cat([t, pad_tensor], dim=dim)
    return t


def ring_double_stream_forward(
    self,
    img: Tensor,
    txt: Tensor,
    vec: Tensor,
    pe: Tensor,
    attn_mask=None,
    modulation_dims_img=None,
    modulation_dims_txt=None,
    transformer_options={},
):
    """Ring-Attention forward for DoubleStreamBlock.
    
    This function is installed via types.MethodType() in workers.
    Replaces the standard Flux DoubleStreamBlock.forward() with USP-enabled version.
    
    Pattern: Same Q/K/V extraction as original, but calls xfuser attention.
    """
    
    # Import xfuser attention
    try:
        from xfuser.core.distributed import (
            get_sequence_parallel_rank,
            get_sequence_parallel_world_size,
            get_sp_group,
        )
    except ImportError:
        return self._original_forward(
            img,
            txt,
            vec,
            pe,
            attn_mask,
            modulation_dims_img,
            modulation_dims_txt,
            transformer_options,
        )

    from comfy.ldm.flux.layers import apply_mod
    from comfy.ldm.flux.math import apply_rope
    
    # Pad sequences if odd length (Raylight pattern)
    img = pad_if_odd(img, dim=1)
    txt = pad_if_odd(txt, dim=1)

    img_mod1, img_mod2 = self.img_mod(vec)
    txt_mod1, txt_mod2 = self.txt_mod(vec)

    sp_rank = get_sequence_parallel_rank()
    sp_size = get_sequence_parallel_world_size()

    original_img_len = img.shape[1]
    original_txt_len = txt.shape[1]

    img = torch.chunk(img, sp_size, dim=1)[sp_rank].contiguous()
    txt = torch.chunk(txt, sp_size, dim=1)[sp_rank].contiguous()

    chunk_img_len = img.shape[1]
    chunk_txt_len = txt.shape[1]

    pe_img_chunk = None
    if pe is not None:
        txt_pe = pe[:, :, :original_txt_len, ...]
        img_pe = pe[:, :, original_txt_len:original_txt_len + original_img_len, ...]
        txt_pe_chunk = torch.chunk(txt_pe, sp_size, dim=2)[sp_rank].contiguous()
        img_pe_chunk = torch.chunk(img_pe, sp_size, dim=2)[sp_rank].contiguous()
        pe_img_chunk = img_pe_chunk

    # prepare image for attention
    img_modulated = self.img_norm1(img)
    img_modulated = apply_mod(img_modulated, (1 + img_mod1.scale), img_mod1.shift, modulation_dims_img)
    img_qkv = self.img_attn.qkv(img_modulated)
    img_q, img_k, img_v = img_qkv.view(img_qkv.shape[0], img_qkv.shape[1], 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

    # prepare txt for attention
    txt_modulated = self.txt_norm1(txt)
    txt_modulated = apply_mod(txt_modulated, (1 + txt_mod1.scale), txt_mod1.shift, modulation_dims_txt)
    txt_qkv = self.txt_attn.qkv(txt_modulated)
    txt_q, txt_k, txt_v = txt_qkv.view(txt_qkv.shape[0], txt_qkv.shape[1], 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

    img_q, img_k = apply_rope(img_q, img_k, pe_img_chunk) if pe_img_chunk is not None else (img_q, img_k)

    if self.flipped_img_txt:
        attn = raylight_attention(
            torch.cat((img_q, txt_q), dim=2),
            torch.cat((img_k, txt_k), dim=2),
            torch.cat((img_v, txt_v), dim=2),
            pe=None,
            mask=attn_mask,
        )
        img_attn = attn[:, :chunk_img_len]
        txt_attn = attn[:, chunk_img_len:]
    else:
        attn = raylight_attention(
            torch.cat((txt_q, img_q), dim=2),
            torch.cat((txt_k, img_k), dim=2),
            torch.cat((txt_v, img_v), dim=2),
            pe=None,
            mask=attn_mask,
        )
        txt_attn = attn[:, :chunk_txt_len]
        img_attn = attn[:, chunk_txt_len:]

    # calculate the img bloks
    img = img + apply_mod(self.img_attn.proj(img_attn), img_mod1.gate, None, modulation_dims_img)
    img = img + apply_mod(self.img_mlp(apply_mod(self.img_norm2(img), (1 + img_mod2.scale), img_mod2.shift, modulation_dims_img)), img_mod2.gate, None, modulation_dims_img)

    # calculate the txt bloks
    txt += apply_mod(self.txt_attn.proj(txt_attn), txt_mod1.gate, None, modulation_dims_txt)
    txt += apply_mod(self.txt_mlp(apply_mod(self.txt_norm2(txt), (1 + txt_mod2.scale), txt_mod2.shift, modulation_dims_txt)), txt_mod2.gate, None, modulation_dims_txt)

    if txt.dtype == torch.float16:
        txt = torch.nan_to_num(txt, nan=0.0, posinf=65504, neginf=-65504)

    # Gather sequences from all GPUs (Raylight pattern)
    img = get_sp_group().all_gather(img.contiguous(), dim=1)
    txt = get_sp_group().all_gather(txt.contiguous(), dim=1)

    return img, txt


def ring_single_stream_forward(self, x: Tensor, vec: Tensor, pe: Tensor, attn_mask=None, modulation_dims=None, **kwargs) -> Tensor:
    """Ring-Attention forward for SingleStreamBlock.
    
    This function is installed via types.MethodType() in workers.
    Replaces the standard Flux SingleStreamBlock.forward() with USP-enabled version.
    """
    from comfy.ldm.flux.layers import apply_mod

    mod, _ = self.modulation(vec)
    qkv, mlp = torch.split(self.linear1(apply_mod(self.pre_norm(x), (1 + mod.scale), mod.shift, modulation_dims)), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)

    q, k, v = qkv.view(qkv.shape[0], qkv.shape[1], 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    q, k = self.norm(q, k, v)

    attn = raylight_attention(q, k, v, pe=pe, mask=attn_mask)
    
    output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
    x += apply_mod(output, mod.gate, None, modulation_dims)
    
    if x.dtype == torch.float16:
        x = torch.nan_to_num(x, nan=0.0, posinf=65504, neginf=-65504)
    
    return x
