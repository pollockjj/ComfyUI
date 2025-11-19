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
        self.backend_cls = attn_backend_cls  # Store backend class for communicator methods

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
        
        # Check if backend needs sequence parallel wrapper
        needs_wrapper = self.backend_cls.needs_sequence_parallel_wrapper()
        
        if not needs_wrapper:
            # Backend handles all communication internally (e.g., xFuser)
            # Just call the backend's forward directly
            LOGGER.debug("%s backend handles communication internally, skipping wrapper", LOG_PREFIX)
            output = self.attn_impl.forward(q, k, v, metadata, attn_mask=attn_mask)
            
            # INSTRUMENTATION: Attention output logging removed (to reduce noise)
            
            # Handle replicated tokens if provided (backend should support this)
            if replicated_q is not None:
                replicated_output = self.attn_impl.forward(
                    replicated_q, replicated_k, replicated_v, metadata, attn_mask=None
                )
                return output, replicated_output
            return output, None
        
        # Use backend's communicator for wrapper-based backends
        world_size = self.backend_cls.get_world_size()
        local_rank = self.backend_cls.get_rank()

        LOGGER.debug(
            "%s forward input shapes: q=%s, k=%s, v=%s",
            LOG_PREFIX,
            tuple(q.shape),
            tuple(k.shape),
            tuple(v.shape),
        )
        if replicated_q is not None:
            LOGGER.debug(
                "%s replicated input shapes: q=%s, k=%s, v=%s",
                LOG_PREFIX,
                tuple(replicated_q.shape),
                tuple(replicated_k.shape),
                tuple(replicated_v.shape),
            )

        # FastVideo pattern: cat → all_to_all → preprocess → forward → postprocess → all_to_all back
        # Input: [batch, seq, heads, dim] each → Cat to: [batch*3, seq, heads, dim]
        qkv = torch.cat([q, k, v], dim=0)
        LOGGER.debug("%s concatenated qkv shape: %s", LOG_PREFIX, tuple(qkv.shape))

        # Step 1: Redistribute heads across sequence dimension (Ulysses)
        # Before: [batch*3, seq_local, heads, dim]
        # After:  [batch*3, seq_local * world_size, heads / world_size, dim]
        qkv = self.backend_cls.all_to_all_4d(qkv, scatter_dim=2, gather_dim=1)
        LOGGER.debug("%s after all-to-all qkv shape: %s", LOG_PREFIX, tuple(qkv.shape))

        # Step 2: Backend-specific preprocessing (e.g., RoPE application)
        qkv = self.attn_impl.preprocess_qkv(qkv, metadata)

        # Step 3: Handle replicated tokens (text) if provided
        batch_size = qkv.shape[0] // 3
        q, k, v = torch.split(qkv, batch_size, dim=0)
        
        replicated_output = None
        if replicated_q is not None:
            assert replicated_k is not None and replicated_v is not None
            
            # Shard replicated tokens by heads (they haven't been through all-to-all)
            # Each rank gets heads/world_size from the ORIGINAL head count
            heads_per_rank = self.num_heads // world_size
            head_start = local_rank * heads_per_rank
            head_end = (local_rank + 1) * heads_per_rank
            
            replicated_q = replicated_q[:, :, head_start:head_end, :]
            replicated_k = replicated_k[:, :, head_start:head_end, :]
            replicated_v = replicated_v[:, :, head_start:head_end, :]
            
            # Concatenate replicated tokens with distributed tokens along sequence dim
            main_seq_len = q.shape[1]
            q = torch.cat([q, replicated_q], dim=1)
            k = torch.cat([k, replicated_k], dim=1)
            v = torch.cat([v, replicated_v], dim=1)
        else:
            main_seq_len = None

        # Step 4: Core attention computation (LOCAL attention only)
        output = self.attn_impl.forward(
            q,
            k,
            v,
            metadata,
            attn_mask=attn_mask,
        )

        # Step 5: Backend-specific postprocessing
        output = self.attn_impl.postprocess_output(output, metadata)
        
        # Step 6: Split replicated output if we had replicated tokens
        if replicated_q is not None:
            main_output = output[:, :main_seq_len]
            replicated_output = output[:, main_seq_len:]
            # Gather replicated output across heads
            LOGGER.debug(
                "%s gathering replicated output shape=%s across dim=2",
                LOG_PREFIX, tuple(replicated_output.shape)
            )
            try:
                replicated_output = self.backend_cls.all_gather_nd(replicated_output.contiguous(), dim=2)
                LOGGER.debug(
                    "%s gathered replicated output shape=%s",
                    LOG_PREFIX, tuple(replicated_output.shape)
                )
            except Exception as e:
                LOGGER.error("%s all_gather_nd failed: %s", LOG_PREFIX, e)
                raise
        else:
            main_output = output

        # Step 7: Redistribute back (sequence → heads)
        # Before: [batch, seq_local * world_size, heads / world_size, dim]
        # After:  [batch, seq_local, heads, dim]
        main_output = self.backend_cls.all_to_all_4d(main_output, scatter_dim=1, gather_dim=2)
        LOGGER.debug("%s after reverse all-to-all output shape: %s", LOG_PREFIX, tuple(main_output.shape))

        return main_output, replicated_output
