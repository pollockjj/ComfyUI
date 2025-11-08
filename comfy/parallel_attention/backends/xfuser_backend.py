"""xFuser USP backend adapter for the Backend Plugin System."""

from __future__ import annotations

import logging
from typing import Optional

import torch
from torch import Tensor
from torch.distributed import ProcessGroup

from ..usp_attention import initialize_usp_attention, usp_attention
from .abstract import AttentionBackend, AttentionImpl, AttentionMetadata

LOGGER = logging.getLogger(__name__)
LOG_PREFIX = "⚡ [Parallel-Attention][BPS-xFuserUSP]"


class XFuserUSPBackend(AttentionBackend):
    """Factory for the xFuser USP attention implementation."""

    @staticmethod
    def get_name() -> str:
        return "XFUSER_USP"

    @staticmethod
    def get_impl_cls() -> type[XFuserUSPImpl]:
        return XFuserUSPImpl


class XFuserUSPImpl(AttentionImpl):
    """xFuser USP attention kernel wrapper.

    RoPE handling: xFuser's `usp_attention` applies RoPE internally when
    `freqs_cis` is provided. We pass metadata through without additional
    preprocessing.
    """

    def __init__(
        self,
        *,
        num_heads: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
        process_group: Optional[ProcessGroup] = None,
        ulysses_degree: int = 1,
        ring_degree: int = 1,
        attn_type: str = "FLASH_ATTN",
    ) -> None:
        super().__init__(
            num_heads=num_heads,
            head_dim=head_dim,
            device=device,
            dtype=dtype,
            process_group=process_group,
        )
        self.ulysses_degree = ulysses_degree
        self.ring_degree = ring_degree
        self.attn_type = attn_type

        # Map common lowercase names to xFuser's uppercase enum names
        attn_type_normalized = attn_type.upper()
        if attn_type_normalized == "SDPA":
            attn_type_normalized = "FLASH_ATTN"  # Default SDPA to FLASH_ATTN

        initialize_usp_attention(
            ulysses_degree=ulysses_degree,
            ring_degree=ring_degree,
            attn_type=attn_type_normalized,
        )
        LOGGER.debug(
            "%s Initialized xFuser backend (ulysses=%d, ring=%d, attn_type=%s)",
            LOG_PREFIX,
            ulysses_degree,
            ring_degree,
            attn_type,
        )

    def preprocess_qkv(self, qkv: Tensor, metadata: AttentionMetadata) -> Tensor:
        """No-op: xFuser applies RoPE inside its kernel."""
        return qkv

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        metadata: AttentionMetadata,
        *,
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Execute xFuser USP attention."""
        return usp_attention(
            q,
            k,
            v,
            freqs_cis=metadata.freqs_cis,
            mask=attn_mask,
        )
