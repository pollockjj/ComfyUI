"""Unified Sequence Parallel double-stream forward helper."""

from __future__ import annotations

import logging
from typing import Optional

import torch
from torch import Tensor

from .usp_attention import apply_rope, usp_attention
from .usp_single_forward import apply_mod
from .dedup_logger import get_dedup_logger

LOG_PREFIX = "⚡ [Parallel-Attention][USP][Double]"
LOGGER = logging.getLogger(__name__)
dedup_logger = get_dedup_logger(__name__)


def usp_double_forward(
    self,
    *,
    img: Tensor,
    txt: Tensor,
    vec: Tensor,
    pe: Optional[Tensor],
    attn_mask: Optional[Tensor] = None,
    modulation_dims_img=None,
    modulation_dims_txt=None,
    **kwargs,
) -> tuple[Tensor, Tensor]:
    # CHECKPOINT STEP3: Inputs to double block (FSDP2)
    if self.rank == 0:
        import hashlib
        for tensor, name in [(img, "img"), (txt, "txt"), (vec, "vec")]:
            tensor_bytes = tensor.detach().cpu().to(torch.float32).numpy().tobytes()
            tensor_hash = hashlib.sha256(tensor_bytes).hexdigest()[:16]
            LOGGER.info(
                "⚡ [STEP3-VERIFY] %s - hash: %s, shape: %s, first5: %s",
                name,
                tensor_hash,
                tuple(tensor.shape),
                tensor.flatten()[:5].tolist()
            )
    """Sequence-parallel double-stream forward pass."""
    
    dedup_logger.info(
        "⚡⚡⚡ XFUSER USP DOUBLE FORWARD CALLED: img.shape=%s, txt.shape=%s, vec=%s, block_id=%s",
        img.shape, txt.shape, 'None' if vec is None else vec.shape, id(self)
    )
    
    # CHECKPOINT: STEP9 - Block inputs (logging removed)

    img_local = img
    txt_local = txt
    pe_local = pe

    img_mod1, img_mod2 = self.img_mod(vec)
    txt_mod1, txt_mod2 = self.txt_mod(vec)

    img_modulated = apply_mod(self.img_norm1(img_local), (1 + img_mod1.scale), img_mod1.shift, modulation_dims_img)
    txt_modulated = apply_mod(self.txt_norm1(txt_local), (1 + txt_mod1.scale), txt_mod1.shift, modulation_dims_txt)

    img_qkv = self.img_attn.qkv(img_modulated)
    txt_qkv = self.txt_attn.qkv(txt_modulated)
    
    # CHECKPOINT: STEP10 - QKV projection output (logging removed)

    img_q, img_k, img_v = img_qkv.view(img_qkv.shape[0], img_qkv.shape[1], 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    
    # CHECKPOINT: STEP11 - After view+permute split (logging removed)
    txt_q, txt_k, txt_v = txt_qkv.view(txt_qkv.shape[0], txt_qkv.shape[1], 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)

    # CHECKPOINT: NON-BPS QKV split (logging removed)

    img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)
    txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)
    
    # CHECKPOINT: STEP3 - After norm (before RoPE) (logging removed)

    if pe_local is not None:
        # CHECKPOINT: STEP11.4/11.5/12 - RoPE handling for img stream (logging removed)
        img_q, img_k = apply_rope(img_q, img_k, pe_local)

    if self.flipped_img_txt:
        attn = usp_attention(
            torch.cat((img_q, txt_q), dim=2),
            torch.cat((img_k, txt_k), dim=2),
            torch.cat((img_v, txt_v), dim=2),
            freqs_cis=None,
            mask=attn_mask,
        )
        img_attn, txt_attn = attn[:, : img_local.shape[1]], attn[:, img_local.shape[1]:]
    else:
        # CHECKPOINT STEP7: Before attention (FSDP2)
        import hashlib
        q_cat = torch.cat((txt_q, img_q), dim=2)
        k_cat = torch.cat((txt_k, img_k), dim=2)
        v_cat = torch.cat((txt_v, img_v), dim=2)
        
        if self.rank == 0:
            for tensor, name in [(q_cat, "q_cat"), (k_cat, "k_cat"), (v_cat, "v_cat")]:
                tensor_bytes = tensor.detach().cpu().to(torch.float32).numpy().tobytes()
                tensor_hash = hashlib.sha256(tensor_bytes).hexdigest()[:16]
                LOGGER.info(
                    "⚡ [STEP7-VERIFY] %s - hash: %s, shape: %s, first5: %s",
                    name,
                    tensor_hash,
                    tuple(tensor.shape),
                    tensor.flatten()[:5].tolist()
                )
        
        attn = usp_attention(
            q_cat,
            k_cat,
            v_cat,
            freqs_cis=None,
            mask=attn_mask,
        )
        
        txt_attn, img_attn = attn[:, : txt_local.shape[1]], attn[:, txt_local.shape[1]:]

    img_attn = img_attn.contiguous()
    txt_attn = txt_attn.contiguous()

    img_updated = img_local + apply_mod(
        self.img_attn.proj(img_attn), img_mod1.gate, None, modulation_dims_img
    )
    
    img_updated = img_updated + apply_mod(
        self.img_mlp(apply_mod(self.img_norm2(img_updated), (1 + img_mod2.scale), img_mod2.shift, modulation_dims_img)),
        img_mod2.gate,
        None,
        modulation_dims_img,
    )

    txt_updated = txt_local + apply_mod(
        self.txt_attn.proj(txt_attn), txt_mod1.gate, None, modulation_dims_txt
    )
    txt_updated = txt_updated + apply_mod(
        self.txt_mlp(apply_mod(self.txt_norm2(txt_updated), (1 + txt_mod2.scale), txt_mod2.shift, modulation_dims_txt)),
        txt_mod2.gate,
        None,
        modulation_dims_txt,
    )

    if txt_updated.dtype == torch.float16:
        txt_updated = torch.nan_to_num(txt_updated, nan=0.0, posinf=65504, neginf=-65504)

    # CHECKPOINT: STEP16 - End of double block forward

    return img_updated, txt_updated
