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
        replicated_q: Optional[Tensor] = None,
        replicated_k: Optional[Tensor] = None,
        replicated_v: Optional[Tensor] = None,
        freqs_cis: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, Optional[Tensor]]:
        """Execute distributed attention using the configured backend.

        FastVideo pattern: Main q/k/v are sharded across sequence (Ulysses),
        replicated q/k/v are replicated on all GPUs (typically text conditioning).

        Args:
            q, k, v: Query, key, value tensors shaped [batch, seq, heads, dim] (will be sharded).
            replicated_q/k/v: Optional replicated tensors [batch, seq, heads, dim] (not sharded).
            freqs_cis: Optional RoPE coefficients.
            attn_mask: Optional attention mask.

        Returns:
            Tuple of (main_output [batch, seq, heads, dim], replicated_output or None).
        """
        metadata = AttentionMetadata(freqs_cis=freqs_cis)

        # Stack QKV (FastVideo pattern uses torch.cat along dim 0)
        # Input: [batch, seq, heads, dim] each → Cat to: [batch*3, seq, heads, dim]
        qkv = torch.cat([q, k, v], dim=0)

        # Backend-specific preprocessing on STACKED tensor (e.g., RoPE application)
        qkv = self.attn_impl.preprocess_qkv(qkv, metadata)

        # Split back to individual tensors
        batch_size = q.shape[0]
        q, k, v = torch.split(qkv, batch_size, dim=0)
        
        # Store original sequence length before concatenation
        main_seq_len = q.shape[1]
        
        # Handle replicated tensors if provided
        if replicated_q is not None:
            # Concatenate replicated tokens with main tokens
            assert replicated_k is not None and replicated_v is not None
            replicated_qkv = torch.cat([replicated_q, replicated_k, replicated_v], dim=0)
            replicated_qkv = self.attn_impl.preprocess_qkv(replicated_qkv, metadata)
            rep_batch_size = replicated_q.shape[0]
            rep_q, rep_k, rep_v = torch.split(replicated_qkv, rep_batch_size, dim=0)
            
            # Concatenate: [main_seq + replicated_seq, ...]
            q = torch.cat([q, rep_q], dim=1)
            k = torch.cat([k, rep_k], dim=1)
            v = torch.cat([v, rep_v], dim=1)

        # Core attention computation
        output = self.attn_impl.forward(
            q,
            k,
            v,
            metadata,
            attn_mask=attn_mask,
        )

        # Backend-specific postprocessing
        output = self.attn_impl.postprocess_output(output, metadata)
        
        # Split output if we had replicated tokens
        replicated_output = None
        if replicated_q is not None:
            main_output = output[:, :main_seq_len]
            replicated_output = output[:, main_seq_len:]
            return main_output, replicated_output

        return output, None
