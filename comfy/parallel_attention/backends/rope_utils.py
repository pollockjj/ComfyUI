"""Utilities for normalising RoPE tensors across attention backends.

Canonical representation
------------------------
All helper functions operate on tensors shaped ``(batch, seq, rotary_dim, 2, 2)``:
- ``batch`` matches the per-rank batch size (broadcastable when set to ``1``).
- ``seq`` is the token count *per rank* after any sequence sharding.
- ``rotary_dim`` equals ``head_dim // 2`` for the target attention kernel.
- The trailing ``2x2`` stores ``[[cos, -sin], [sin, cos]]`` blocks as produced by
  ``comfy.ldm.flux.math.rope``.

Backends that expect alternate layouts should convert to the canonical shape
first and then reshape as required. This guarantees a single source of truth for
RoPE semantics within the Backend Plugin System.
"""

from __future__ import annotations

import logging
from typing import Iterable, Tuple

import torch
from torch import Tensor


_CANONICAL_RANK = 5
LOGGER = logging.getLogger(__name__)


def _pair_to_canonical(cos: Tensor, sin: Tensor) -> Tensor:
    if cos.shape != sin.shape:
        raise ValueError("cos and sin tensors must share the same shape")
    stacked = torch.stack((cos, -sin, sin, cos), dim=-1)
    return stacked.view(*cos.shape, 2, 2)


def canonicalize_rope(freqs_cis: Tensor | Iterable[Tensor]) -> Tensor:
    """Return a canonical RoPE tensor regardless of input layout."""

    if isinstance(freqs_cis, (tuple, list)):
        if len(freqs_cis) != 2:
            raise ValueError("RoPE iterable must contain exactly two tensors (cos, sin)")
        cos, sin = freqs_cis  # type: ignore[assignment]
        LOGGER.debug("⚡ [Parallel-Attention][BPS-RoPE] canonicalize tuple input shape=%s", tuple(torch.as_tensor(cos).shape))
        return canonicalize_rope(_pair_to_canonical(torch.as_tensor(cos), torch.as_tensor(sin)))

    if not torch.is_tensor(freqs_cis):
        raise TypeError("freqs_cis must be a Tensor or a (cos, sin) pair")

    tensor = freqs_cis

    if tensor.ndim == 5:
        if tensor.shape[-1] != 2 or tensor.shape[-2] != 2:
            raise ValueError("Last two dims must be size 2 for canonical RoPE tensors")
        LOGGER.debug("⚡ [Parallel-Attention][BPS-RoPE] canonical tensor already 5D shape=%s", tuple(tensor.shape))
        return tensor

    if tensor.ndim == 4 and tensor.shape[-1] == 2:
        LOGGER.debug("⚡ [Parallel-Attention][BPS-RoPE] expanding 4D tensor shape=%s", tuple(tensor.shape))
        return tensor.unsqueeze(-2)

    if tensor.ndim == 3 and tensor.shape[-1] == 2:
        LOGGER.debug("⚡ [Parallel-Attention][BPS-RoPE] expanding 3D tensor shape=%s", tuple(tensor.shape))
        return tensor.unsqueeze(0).unsqueeze(-2)

    if tensor.ndim == 2 and tensor.shape[-1] == 2:
        LOGGER.debug("⚡ [Parallel-Attention][BPS-RoPE] expanding 2D tensor shape=%s", tuple(tensor.shape))
        return tensor.unsqueeze(0).unsqueeze(0).unsqueeze(-2)

    raise ValueError("Unsupported RoPE tensor shape")


def match_rope_batch(freqs_cis: Tensor, batch_size: int) -> Tensor:
    """Broadcast canonical RoPE tensor to an expected batch size."""

    canonical = canonicalize_rope(freqs_cis)

    if canonical.shape[0] not in (1, batch_size):
        raise ValueError(
            f"Cannot broadcast RoPE batch dimension {canonical.shape[0]} to {batch_size}"
        )

    if canonical.shape[0] == batch_size:
        return canonical

    # Broadcast batch dimension lazily; caller can clone if mutation is required.
    LOGGER.debug(
        "⚡ [Parallel-Attention][BPS-RoPE] broadcasting batch from 1 to %s shape=%s",
        batch_size,
        tuple(canonical.shape),
    )
    return canonical.expand(batch_size, *canonical.shape[1:])


def prepare_rope_for_qkv(freqs_cis: Tensor | Iterable[Tensor], q: Tensor) -> Tensor:
    """Produce a canonical RoPE tensor aligned with a given Q tensor."""

    canonical = canonicalize_rope(freqs_cis)
    canonical = match_rope_batch(canonical, q.shape[0])
    LOGGER.debug(
        "⚡ [Parallel-Attention][BPS-RoPE] aligning tensor to device=%s dtype=%s", q.device, q.dtype
    )
    return canonical.to(device=q.device, dtype=q.dtype)


def to_cos_sin_pair(freqs_cis: Tensor | Iterable[Tensor]) -> Tuple[Tensor, Tensor]:
    """Return explicit cos/sin tensors from any supported RoPE layout."""

    canonical = canonicalize_rope(freqs_cis)
    cos = canonical[..., 0, 0]
    sin = canonical[..., 1, 0]
    return cos, sin