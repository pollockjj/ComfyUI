"""Ring-Attention forward methods for Flux (worker installation).

Based on Raylight's xdit_context_parallel.py approach.
These functions replace the forward() methods on DoubleStreamBlock and SingleStreamBlock.
"""

import torch
from torch import Tensor


def pad_if_odd(t: torch.Tensor, dim: int = 1):
    """Pad tensor if dimension is odd (for sequence splitting)."""
    if t.size(dim) % 2 != 0:
        pad_shape = list(t.shape)
        pad_shape[dim] = 1
        pad_tensor = torch.zeros(pad_shape, dtype=t.dtype, device=t.device)
        t = torch.cat([t, pad_tensor], dim=dim)
    return t


def ring_double_stream_forward(self, img: Tensor, txt: Tensor, vec: Tensor, pe: Tensor, 
                                attn_mask=None, modulation_dims_img=None, modulation_dims_txt=None, 
                                transformer_options={}):
    """Ring-Attention forward for DoubleStreamBlock.
    
    This function is installed via types.MethodType() in workers.
    Replaces the standard Flux DoubleStreamBlock.forward() with USP-enabled version.
    
    Pattern: Same Q/K/V extraction as original, but calls xfuser attention.
    """
    
    # Import xfuser attention
    try:
        from xfuser.model_executor.layers.usp import USP
        from xfuser.core.distributed import get_sequence_parallel_world_size
        sp_size = get_sequence_parallel_world_size()
    except ImportError as e:
        # Fall back to original forward if xfuser not available
        return self._original_forward(img, txt, vec, pe, attn_mask, modulation_dims_img, modulation_dims_txt, transformer_options)
    
    from comfy.ldm.flux.layers import apply_mod
    from comfy.ldm.flux.math import attention, apply_rope
    from xfuser.core.distributed import get_sequence_parallel_rank, get_sequence_parallel_world_size, get_sp_group
    
    # Pad sequences if odd length (Raylight pattern)
    img = pad_if_odd(img, dim=1)
    txt = pad_if_odd(txt, dim=1)
    
    img_mod1, img_mod2 = self.img_mod(vec)
    txt_mod1, txt_mod2 = self.txt_mod(vec)
    
    # Split sequences across GPUs (Raylight pattern)
    sp_rank = get_sequence_parallel_rank()
    sp_size = get_sequence_parallel_world_size()
    
    # pe contains embeddings for txt + img concatenated
    # We need to slice it to match img only (for apply_rope)
    # Original: pe[:, :, :txt.shape[1]+img.shape[1], ...]
    # After split we only need the img portion
    original_txt_len = txt.shape[1]  # 256
    
    img = torch.chunk(img, sp_size, dim=1)[sp_rank]
    txt = torch.chunk(txt, sp_size, dim=1)[sp_rank]
    
    # Now split pe for just the img portion
    if pe is not None:
        # pe has txt embeddings first, then img embeddings
        # Extract just img portion: pe[:, :, txt_len:, ...]
        pe_img = pe[:, :, original_txt_len:, ...]
        # Split the img portion across GPUs
        pe_img = torch.chunk(pe_img, sp_size, dim=2)[sp_rank]
        pe = pe_img  # Use the split img-only pe for apply_rope

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

    # USE XFUSER USP ATTENTION
    if self.flipped_img_txt:
        img_q, img_k = apply_rope(img_q, img_k, pe)
        
        attn = USP(
            query=torch.cat((img_q, txt_q), dim=2),
            key=torch.cat((img_k, txt_k), dim=2),
            value=torch.cat((img_v, txt_v), dim=2),
            dropout_p=0.0,
            is_causal=False
        )
        
        img_attn, txt_attn = attn[:, :, : img.shape[1]], attn[:, :, img.shape[1]:]
    else:
        img_q, img_k = apply_rope(img_q, img_k, pe)
        
        attn = USP(
            query=torch.cat((txt_q, img_q), dim=2),
            key=torch.cat((txt_k, img_k), dim=2),
            value=torch.cat((txt_v, img_v), dim=2),
            dropout_p=0.0,
            is_causal=False
        )
        
        txt_attn, img_attn = attn[:, :, : txt.shape[1]], attn[:, :, txt.shape[1]:]

    # Reshape attention output from [B, H, S, D] to [B, S, H*D] for projection
    # img_attn: [1, 24, 2048, 128] -> [1, 2048, 3072]
    img_attn = img_attn.transpose(1, 2).reshape(img_attn.shape[0], img_attn.shape[2], -1)
    txt_attn = txt_attn.transpose(1, 2).reshape(txt_attn.shape[0], txt_attn.shape[2], -1)

    # calculate the img bloks
    img = img + apply_mod(self.img_attn.proj(img_attn), img_mod1.gate, None, modulation_dims_img)
    img = img + apply_mod(self.img_mlp(apply_mod(self.img_norm2(img), (1 + img_mod2.scale), img_mod2.shift, modulation_dims_img)), img_mod2.gate, None, modulation_dims_img)

    # calculate the txt bloks
    txt += apply_mod(self.txt_attn.proj(txt_attn), txt_mod1.gate, None, modulation_dims_txt)
    txt += apply_mod(self.txt_mlp(apply_mod(self.txt_norm2(txt), (1 + txt_mod2.scale), txt_mod2.shift, modulation_dims_txt)), txt_mod2.gate, None, modulation_dims_txt)

    if txt.dtype == torch.float16:
        txt = torch.nan_to_num(txt, nan=0.0, posinf=65504, neginf=-65504)

    # Gather sequences from all GPUs (Raylight pattern)
    img = get_sp_group().all_gather(img, dim=1)
    txt = get_sp_group().all_gather(txt, dim=1)

    return img, txt


def ring_single_stream_forward(self, x: Tensor, vec: Tensor, pe: Tensor, attn_mask=None, modulation_dims=None, **kwargs) -> Tensor:
    """Ring-Attention forward for SingleStreamBlock.
    
    This function is installed via types.MethodType() in workers.
    Replaces the standard Flux SingleStreamBlock.forward() with USP-enabled version.
    """
    from comfy.ldm.flux.layers import apply_mod
    from comfy.ldm.flux.math import attention
    from xfuser.core.distributed import get_sequence_parallel_world_size
    
    sp_size = get_sequence_parallel_world_size()
    
    mod, _ = self.modulation(vec)
    qkv, mlp = torch.split(self.linear1(apply_mod(self.pre_norm(x), (1 + mod.scale), mod.shift, modulation_dims)), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)

    q, k, v = qkv.view(qkv.shape[0], qkv.shape[1], 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    q, k = self.norm(q, k, v)

    # Use xfuser USP attention if sp_size > 1 (Ulysses or Ring enabled)
    if sp_size > 1:
        from xfuser.model_executor.layers.usp import USP
        attn = USP(query=q, key=k, value=v, dropout_p=0.0, is_causal=False)
    else:
        attn = attention(q, k, v, pe=pe, mask=attn_mask)
    
    output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
    x += apply_mod(output, mod.gate, None, modulation_dims)
    
    if x.dtype == torch.float16:
        x = torch.nan_to_num(x, nan=0.0, posinf=65504, neginf=-65504)
    
    return x
