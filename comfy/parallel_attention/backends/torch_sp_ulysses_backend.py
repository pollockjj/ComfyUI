"""Torch Sequence-Parallel Ulysses backend adapter for the Backend Plugin System."""

from __future__ import annotations

import logging
from typing import Optional

import torch
from torch import Tensor
from torch.distributed import ProcessGroup

from ..torch_sp_ulysses.attention import UlyssesAttention
from .abstract import AttentionBackend, AttentionImpl, AttentionMetadata
from .rope_utils import prepare_rope_for_qkv

LOGGER = logging.getLogger(__name__)
LOG_PREFIX = "⚡ [Parallel-Attention][BPS-TorchSPUlysses]"


class TorchSPUlyssesBackend(AttentionBackend):
    """Factory for the Torch SP Ulysses attention implementation."""

    @staticmethod
    def get_name() -> str:
        return "TORCH_SP_ULYSSES"

    @staticmethod
    def get_impl_cls() -> type[TorchSPUlyssesImpl]:
        return TorchSPUlyssesImpl


class TorchSPUlyssesImpl(AttentionImpl):
    """Torch SP Ulysses attention kernel wrapper.

    RoPE handling: Applies RoPE in preprocess_qkv using rope_utils to normalize
    tensor shapes before passing to the underlying UlyssesAttention kernel.
    """

    def __init__(
        self,
        *,
        num_heads: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
        process_group: Optional[ProcessGroup] = None,
        backend: str = "sdpa",
        # Ignore unused BPS kwargs
        ulysses_degree: Optional[int] = None,
        ring_degree: Optional[int] = None,
        attn_type: Optional[str] = None,
    ) -> None:
        super().__init__(
            num_heads=num_heads,
            head_dim=head_dim,
            device=device,
            dtype=dtype,
            process_group=process_group,
        )
        # Use attn_type if provided, otherwise backend
        self.backend = attn_type if attn_type else backend

        self._attn = UlyssesAttention(
            num_heads=num_heads,
            head_dim=head_dim,
            backend=self.backend,
        )

        LOGGER.debug(
            "%s Initialized Torch SP Ulysses (backend=%s)",
            LOG_PREFIX,
            self.backend,
        )

    def preprocess_qkv(self, qkv: Tensor, metadata: AttentionMetadata) -> Tensor:
        """Apply RoPE to concatenated QKV tensor (FastVideo pattern).
        
        Args:
            qkv: Concatenated tensor [batch*3, seq, heads, dim] (FastVideo uses cat not stack)
            metadata: Contains freqs_cis for RoPE
            
        Returns:
            Concatenated tensor with RoPE applied to Q and K
        """
        if metadata.freqs_cis is None:
            return qkv

        # Split to get Q, K, V (they're concatenated along batch dim)
        batch_size = qkv.shape[0] // 3
        q, k, v = torch.split(qkv, batch_size, dim=0)

        # Prepare RoPE tensor to match Q shape
        freqs_cis = prepare_rope_for_qkv(metadata.freqs_cis, q)

        # Apply RoPE to Q and K (borrowed from usp_attention.py)
        # Shape: [batch, seq, heads, dim]
        q_ = q.to(dtype=freqs_cis.dtype).reshape(*q.shape[:-1], -1, 1, 2)
        k_ = k.to(dtype=freqs_cis.dtype).reshape(*k.shape[:-1], -1, 1, 2)
        q_out = freqs_cis[..., 0] * q_[..., 0] + freqs_cis[..., 1] * q_[..., 1]
        k_out = freqs_cis[..., 0] * k_[..., 0] + freqs_cis[..., 1] * k_[..., 1]
        q_rope = q_out.reshape(*q.shape).type_as(q)
        k_rope = k_out.reshape(*k.shape).type_as(k)

        # Concatenate back
        return torch.cat([q_rope, k_rope, v], dim=0)

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        metadata: AttentionMetadata,
        *,
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Execute Torch SP Ulysses attention (includes internal all-to-all).
        
        Note: RoPE is already applied in preprocess_qkv.
        Input tensors are [batch, seq, heads, dim] (FastVideo convention).
        """
        # UlyssesAttention.forward() expects [batch, seq, heads, dim] - already correct!
        output, _ = self._attn.forward(q, k, v)
        
        return output
