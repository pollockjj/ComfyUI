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
    
    @staticmethod
    def needs_sequence_parallel_wrapper() -> bool:
        """xFuser handles all sequence parallel communication internally."""
        return False
    
    @staticmethod
    def get_world_size() -> int:
        """Return xFuser's sequence parallel world size."""
        LOGGER.info("%s get_world_size() called", LOG_PREFIX)
        from xfuser.core.distributed import get_sequence_parallel_world_size
        result = get_sequence_parallel_world_size()
        LOGGER.info("%s get_world_size() = %d", LOG_PREFIX, result)
        return result
    
    @staticmethod
    def get_rank() -> int:
        """Return xFuser's sequence parallel rank."""
        LOGGER.info("%s get_rank() called", LOG_PREFIX)
        from xfuser.core.distributed import get_sequence_parallel_rank
        result = get_sequence_parallel_rank()
        LOGGER.info("%s get_rank() = %d", LOG_PREFIX, result)
        return result
    
    @staticmethod
    def all_to_all_4d(tensor: Tensor, scatter_dim: int, gather_dim: int) -> Tensor:
        """Use xFuser's all-to-all implementation."""
        from xfuser.core.distributed.runtime_state import get_runtime_state
        runtime_state = get_runtime_state()
        return runtime_state.comm_manager.all_to_all(tensor, scatter_dim, gather_dim)
    
    @staticmethod
    def all_gather_nd(tensor: Tensor, dim: int) -> Tensor:
        """Use xFuser's all-gather implementation."""
        from xfuser.core.distributed.runtime_state import get_runtime_state
        runtime_state = get_runtime_state()
        return runtime_state.comm_manager.all_gather(tensor, dim)


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
        """Execute xFuser USP attention.
        
        BPS passes [batch, seq, heads, dim] but xFuser expects [batch, heads, seq, dim].
        Transpose inputs to match xFuser's convention, call attention, return 3D output.
        """
        # Transpose from BPS convention [batch, seq, heads, dim] to xFuser [batch, heads, seq, dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Call xFuser with [batch, heads, seq, dim]
        output = usp_attention(
            q,
            k,
            v,
            freqs_cis=metadata.freqs_cis,
            mask=attn_mask,
        )
        
        # xFuser returns [batch, seq_combined, heads * dim] (3D, already flattened)
        return output
