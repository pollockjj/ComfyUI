"""Unified DistributedAttention layer for the Backend Plugin System.

This module provides a backend-agnostic sequence-parallel attention layer that
delegates to registered backend implementations. It replaces direct calls to
`usp_attention` or `UlyssesAttention` in forward hooks.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
from torch import Tensor, nn

from .backends import AttentionBackendEnum, AttentionMetadata, get_attn_backend

LOGGER = logging.getLogger(__name__)
LOG_PREFIX = "⚡ [Parallel-Attention][DistributedAttention]"


class DistributedAttention(nn.Module):
    """Backend-agnostic distributed attention layer.

    Wraps backend-specific implementations and provides a unified interface
    for sequence-parallel attention operations.
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
        backend_name: str = "XFUSER_USP",
        **backend_kwargs,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype

        backend_enum = AttentionBackendEnum[backend_name.upper().replace("-", "_")]
        attn_backend_cls = get_attn_backend(
            head_dim=head_dim,
            dtype=dtype,
            supported_backends=(backend_enum,),
        )

        impl_cls = attn_backend_cls.get_impl_cls()
        self.attn_impl = impl_cls(
            num_heads=num_heads,
            head_dim=head_dim,
            device=device,
            dtype=dtype,
            **backend_kwargs,
        )
        self.backend_name = attn_backend_cls.get_name()

        LOGGER.debug(
            "%s Initialized with backend=%s, num_heads=%d, head_dim=%d",
            LOG_PREFIX,
            self.backend_name,
            num_heads,
            head_dim,
        )

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        freqs_cis: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Execute distributed attention using the configured backend.

        Args:
            q, k, v: Query, key, value tensors shaped [batch, heads, seq_local, head_dim].
            freqs_cis: Optional RoPE coefficients.
            attn_mask: Optional attention mask.

        Returns:
            Attention output shaped [batch, seq_local, heads * head_dim].
        """
        metadata = AttentionMetadata(freqs_cis=freqs_cis)

        # Backend-specific preprocessing (e.g., RoPE application)
        q_prep = self.attn_impl.preprocess_qkv(q, metadata)
        k_prep = self.attn_impl.preprocess_qkv(k, metadata)
        v_prep = self.attn_impl.preprocess_qkv(v, metadata)

        # Core attention computation
        output = self.attn_impl.forward(
            q_prep,
            k_prep,
            v_prep,
            metadata,
            attn_mask=attn_mask,
        )

        # Backend-specific postprocessing
        output = self.attn_impl.postprocess_output(output, metadata)

        return output
