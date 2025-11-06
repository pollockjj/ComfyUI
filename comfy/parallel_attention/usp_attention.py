"""Unified Sequence Parallel (USP) attention helpers.

Provides low level building blocks to execute sequence-parallel attention in
the ComfyUI core runtime. Interfaces mirror the flux context-parallel forward
helpers so higher-level patches can be ported with minimal changes.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, Optional

import torch
from torch import Tensor

try:
    from xfuser.core.distributed import (
        get_sequence_parallel_rank,
        get_sequence_parallel_world_size,
        get_sp_group,
    )
    from xfuser.core.long_ctx_attention import xFuserLongContextAttention
    from yunchang.kernels import AttnType
except ImportError as exc:  # pragma: no cover - environment validation
    raise ImportError(
        "USP attention requires xfuser and yunchang packages to be installed"
    ) from exc

LOGGER = logging.getLogger(__name__)

_ATTENTION_HANDLE: Optional[Callable[..., Tensor]] = None
_ATTENTION_CONFIG: Dict[str, int | str] = {}

_ATTENTION_TYPE_MAP = {
    "FLASH_ATTN": AttnType.FA,
    "FLASH_ATTN_3": AttnType.FA3,
    "SAGE_AUTO_DETECT": AttnType.SAGE_AUTO,
    "SAGE_FP16_TRITON": AttnType.SAGE_FP16_TRITON,
    "SAGE_FP16_CUDA": AttnType.SAGE_FP16,
    "SAGE_FP8_CUDA": AttnType.SAGE_FP8,
    "SAGE_FP8_SM90": AttnType.SAGE_FP8_SM90,
}

LOG_PREFIX = "⚡ [Parallel-Attention][USP][Attention]"


def _resolve_attn_type(attn_type: str) -> AttnType:
    key = attn_type.upper()
    if key not in _ATTENTION_TYPE_MAP:
        raise ValueError(f"Unsupported USP attention type: {attn_type}")
    return _ATTENTION_TYPE_MAP[key]


def _build_xfuser_attention(attn_enum: AttnType) -> Callable[..., Tensor]:
    """Create a callable wrapper around `xFuserLongContextAttention`.

    Mirrors the original `make_xfuser_attention` helper so ported forward
    methods can invoke it without modification.
    """

    xfuser_attn = xFuserLongContextAttention(attn_type=attn_enum)

    def _attention(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        heads: int,
        join_q: Optional[Tensor] = None,
        join_k: Optional[Tensor] = None,
        join_v: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        attn_precision: Optional[str] = None,
        skip_reshape: bool = False,
        skip_output_reshape: bool = False,
    ) -> Tensor:
        if skip_reshape:
            batch, _, _, dim_head = q.shape
        else:
            batch, _, dim = q.shape
            dim_head = dim // heads
            q, k, v = map(
                lambda t: t.view(batch, -1, heads, dim_head).transpose(1, 2),
                (q, k, v),
            )
            if join_q is not None:
                join_batch, _, join_dim = join_q.shape
                join_dim_head = join_dim // heads
                join_q, join_k, join_v = map(
                    lambda t: t.view(join_batch, -1, heads, join_dim_head).transpose(1, 2),
                    (join_q, join_k, join_v),
                )

        if mask is not None:
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)

        if join_q is not None:
            out = xfuser_attn(
                None,
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                joint_strategy="rear",
                joint_tensor_query=join_q.transpose(1, 2),
                joint_tensor_key=join_k.transpose(1, 2),
                joint_tensor_value=join_v.transpose(1, 2),
            ).transpose(1, 2)
        else:
            out = xfuser_attn(
                None,
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
            ).transpose(1, 2)

        if not skip_output_reshape:
            out = out.transpose(1, 2).reshape(batch, -1, heads * dim_head)
        return out

    return _attention


def initialize_usp_attention(
    ulysses_degree: int,
    ring_degree: int,
    *,
    attn_type: str = "FLASH_ATTN",
) -> None:
    """Initialise the global USP attention backend.

    The xfuser attention kernel is expensive to build; we cache a callable that
    mirrors the Raylight helper so our ported forwards can reuse it. Calling
    this function multiple times with the same configuration is a no-op.
    """

    global _ATTENTION_HANDLE, _ATTENTION_CONFIG

    config = {
        "ulysses_degree": int(ulysses_degree),
        "ring_degree": int(ring_degree),
        "attn_type": attn_type.upper(),
    }

    if _ATTENTION_HANDLE is not None and config == _ATTENTION_CONFIG:
        return

    attn_enum = _resolve_attn_type(attn_type)
    _ATTENTION_HANDLE = _build_xfuser_attention(attn_enum)
    _ATTENTION_CONFIG = config

    LOGGER.info(
        "%s Initialized (ulysses=%d, ring=%d, backend=%s)",
        LOG_PREFIX,
        config["ulysses_degree"],
        config["ring_degree"],
        config["attn_type"],
    )


def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    """Apply rotary positional embeddings to Q and K chunks."""

    xq_ = xq.to(dtype=freqs_cis.dtype).reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.to(dtype=freqs_cis.dtype).reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)


def usp_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    freqs_cis: Optional[Tensor] = None,
    mask: Optional[Tensor] = None,
) -> Tensor:
    """Execute distributed attention on local sequence chunks.

    Args:
        q/k/v: Tensors shaped `[batch, heads, tokens_local, head_dim]`.
        freqs_cis: Optional RoPE coefficients shaped `[tokens_local, head_dim, 2]`.
        mask: Optional attention mask broadcastable to the local chunk.
    Returns:
        Tensor shaped `[batch, tokens_local, heads * head_dim]`.
    """

    if _ATTENTION_HANDLE is None:
        raise RuntimeError("USP attention has not been initialised. Call initialize_usp_attention() first.")

    if freqs_cis is not None:
        q, k = apply_rope(q, k, freqs_cis)

    heads = q.shape[1]
    return _ATTENTION_HANDLE(
        q,
        k,
        v,
        heads,
        mask=mask,
        skip_reshape=True,
    )


def chunk_sequence_for_rank(tensor: Tensor, dim: int = 1) -> Tensor:
    """Return the shard of `tensor` that belongs to the current sequence rank."""
    world_size = get_sequence_parallel_world_size()
    if world_size == 1:
        return tensor
    rank = get_sequence_parallel_rank()
    return torch.chunk(tensor, world_size, dim=dim)[rank].contiguous()


def gather_sequence_from_ranks(tensor: Tensor, dim: int = 1) -> Tensor:
    """Collect shards from all sequence-parallel ranks and concatenate along `dim`."""
    world_size = get_sequence_parallel_world_size()
    if world_size == 1:
        return tensor
    group = get_sp_group()
    return group.all_gather(tensor.contiguous(), dim=dim)


__all__ = [
    "initialize_usp_attention",
    "usp_attention",
    "apply_rope",
    "chunk_sequence_for_rank",
    "gather_sequence_from_ranks",
]
