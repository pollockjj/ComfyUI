"""Raylight's exact attention implementation (copied verbatim).

This mirrors `raylight.distributed_modules.attention` to ensure identical
behavior for both double and single stream Flux blocks.
"""

from typing import Optional

import torch
from torch import Tensor

from comfy.ldm.flux.math import apply_rope

_xfuser_optimized_attention = None


def _normalise_mask(mask: Optional[Tensor]) -> Optional[Tensor]:
    if mask is None:
        return None
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    return mask


def initialize_raylight_attention(attn_type: str = "flash_attn") -> None:
    """Initialise the xfuser attention wrapper exactly like Raylight."""
    global _xfuser_optimized_attention

    from xfuser.core.long_ctx_attention import xFuserLongContextAttention
    from yunchang.kernels import AttnType

    attn_map = {
        "flash_attn": AttnType.FA,
        "sage_attn": AttnType.SAGE_AUTO,
        "torch": AttnType.TORCH,
    }

    key = attn_type.lower()
    if key not in attn_map:
        valid = ", ".join(attn_map.keys())
        raise ValueError(f"Unknown attention type '{attn_type}'. Valid options: {valid}")

    xfuser_attn = xFuserLongContextAttention(attn_type=attn_map[key])
    print(f"⚡ [Raylight Attention] Initialized {key} backend")

    def _attention_xfuser_unmask(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        heads: int,
        join_q: Optional[Tensor] = None,
        join_k: Optional[Tensor] = None,
        join_v: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        attn_precision: Optional[torch.dtype] = None,
        skip_reshape: bool = False,
        skip_output_reshape: bool = False,
    ) -> Tensor:
        if skip_reshape:
            b, _, _, dim_head = q.shape  # [B, H, S, D]
        else:
            b, seq, dim = q.shape  # [B, S, H*D]
            dim_head = dim // heads

            def _reshape(t: Tensor) -> Tensor:
                return t.view(b, seq, heads, dim_head).transpose(1, 2)

            q = _reshape(q)
            k = _reshape(k)
            v = _reshape(v)

            if join_q is not None:
                join_q = _reshape(join_q)
                join_k = _reshape(join_k)
                join_v = _reshape(join_v)

        mask = _normalise_mask(mask)

        if join_q is not None:
            out = xfuser_attn(
                mask,
                q.transpose(1, 2),  # [B, S, H, D]
                k.transpose(1, 2),
                v.transpose(1, 2),
                joint_strategy="rear",
                joint_tensor_query=join_q.transpose(1, 2),
                joint_tensor_key=join_k.transpose(1, 2),
                joint_tensor_value=join_v.transpose(1, 2),
                attn_precision=attn_precision,
            ).transpose(1, 2)
        else:
            out = xfuser_attn(
                mask,
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                attn_precision=attn_precision,
            ).transpose(1, 2)

        if not skip_output_reshape:
            out = out.transpose(1, 2).reshape(b, -1, heads * dim_head)

        return out

    _xfuser_optimized_attention = _attention_xfuser_unmask


def attention(q: Tensor, k: Tensor, v: Tensor, pe: Optional[Tensor], mask=None) -> Tensor:
    if _xfuser_optimized_attention is None:
        raise RuntimeError("Raylight attention not initialized")

    if pe is not None:
        q, k = apply_rope(q, k, pe)

    heads = q.shape[1]
    return _xfuser_optimized_attention(
        q,
        k,
        v,
        heads,
        mask=mask,
        skip_reshape=True,
    )
