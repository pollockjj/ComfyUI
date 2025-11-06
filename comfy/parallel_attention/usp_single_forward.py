"""Unified Sequence Parallel single-stream forward helper."""

from __future__ import annotations

import logging
from typing import Optional

import torch
from torch import Tensor

from .usp_attention import usp_attention

LOG_PREFIX = "⚡ [Parallel-Attention][USP][Single]"
LOGGER = logging.getLogger(__name__)


def apply_mod(tensor: Tensor, mult: Tensor, add: Optional[Tensor] = None, modulation_dims=None) -> Tensor:
    """Apply modulation gates to `tensor` in-place and return the result."""
    if modulation_dims is None:
        if add is not None:
            return torch.addcmul(add, tensor, mult)
        return tensor * mult

    for d_start, d_end, m_idx in modulation_dims:
        tensor[:, d_start:d_end] *= mult[:, m_idx]
        if add is not None:
            tensor[:, d_start:d_end] += add[:, m_idx]
    return tensor


def usp_single_forward(
    self,
    x: Tensor,
    *,
    vec: Tensor,
    pe: Optional[Tensor],
    attn_mask: Optional[Tensor] = None,
    modulation_dims=None,
    **kwargs,
) -> Tensor:
    """Sequence-parallel single-stream forward pass."""

    local_tokens = x
    pe_local = pe

    if LOGGER.isEnabledFor(logging.INFO):
        LOGGER.info("%s rank_chunk=%d", LOG_PREFIX, local_tokens.shape[1])

    mod, _ = self.modulation(vec)
    pre_norm = self.pre_norm(local_tokens)
    modulated = apply_mod(pre_norm, (1 + mod.scale), mod.shift, modulation_dims)

    qkv_mlp = self.linear1(modulated)
    qkv, mlp = torch.split(qkv_mlp, [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)

    q, k, v = qkv.view(qkv.shape[0], qkv.shape[1], 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    q, k = self.norm(q, k, v)

    attn = usp_attention(q, k, v, freqs_cis=pe_local, mask=attn_mask)
    output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), dim=-1))

    updated = local_tokens + apply_mod(output, mod.gate, None, modulation_dims)
    if updated.dtype == torch.float16:
        updated = torch.nan_to_num(updated, nan=0.0, posinf=65504, neginf=-65504)

    return updated
