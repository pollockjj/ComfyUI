"""TorchSP Ulysses backend - pure PyTorch sequence parallelism implementation.

Built from scratch using PyTorch distributed primitives. No xFuser dependencies.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
from torch import Tensor
from torch.distributed import ProcessGroup

try:
    import flash_attn
    from flash_attn.flash_attn_interface import _flash_attn_forward
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False
    _flash_attn_forward = None
    flash_attn = None

from .abstract import AttentionBackend, AttentionImpl, AttentionMetadata
from .torch_sp_ulysses import communicator

LOGGER = logging.getLogger(__name__)
LOG_PREFIX = "⚡ [Parallel-Attention][BPS-TorchSP]"


class TorchSPUlyssesBackend(AttentionBackend):
    """Factory for TorchSP Ulysses attention implementation."""

    @staticmethod
    def get_name() -> str:
        return "TORCH_SP_ULYSSES"

    @staticmethod
    def get_impl_cls() -> type[TorchSPUlyssesImpl]:
        return TorchSPUlyssesImpl
    
    @staticmethod
    def needs_sequence_parallel_wrapper() -> bool:
        """TorchSP needs explicit all-to-all wrapper (unlike xFuser)."""
        return True
    
    @staticmethod
    def get_world_size() -> int:
        """Return TorchSP sequence parallel world size."""
        return communicator.get_sp_world_size()
    
    @staticmethod
    def get_rank() -> int:
        """Return TorchSP sequence parallel rank."""
        return communicator.get_sp_rank()
    
    @staticmethod
    def all_to_all_4d(tensor: Tensor, scatter_dim: int, gather_dim: int) -> Tensor:
        """TorchSP all-to-all implementation."""
        return communicator.all_to_all_4d(tensor, scatter_dim, gather_dim)
    
    @staticmethod
    def all_gather_nd(tensor: Tensor, dim: int) -> Tensor:
        """TorchSP all-gather implementation."""
        return communicator.all_gather_nd(tensor, dim)


class TorchSPUlyssesImpl(AttentionImpl):
    """TorchSP Ulysses attention implementation using flash-attn.

    RoPE handling: Applied in preprocess_qkv() hook (BPS contract).
    Local attention: flash_attn.flash_attn_interface._flash_attn_forward (SAME as xFuser).
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
        self.attn_type = attn_type.upper()
        
        # Validate FlashAttention requirement
        if not HAS_FLASH_ATTN:
            raise RuntimeError(
                "TorchSP Ulysses requires flash-attn package. "
                "Install: pip install flash-attn --no-build-isolation"
            )
        
        if self.attn_type != "FLASH_ATTN":
            raise ValueError(
                f"TorchSP Ulysses only supports FLASH_ATTN backend. "
                f"Got: {self.attn_type}"
            )

        # NOP: TorchSP backend initialization logging removed for performance

    def preprocess_qkv(self, qkv: Tensor, metadata: AttentionMetadata) -> Tensor:
        """No-op: RoPE handling depends on context (double vs single blocks).
        
        For double blocks: RoPE applied in bps_double_forward before attention call.
        For single blocks: RoPE must be applied inside forward() for img portion only.
        
        Shape: qkv [batch*3, seq, heads, dim] → [batch*3, seq, heads, dim]
               (Q, K, V are concatenated along batch dimension)
        """
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
        """Execute local FlashAttention using the SAME kernel as xFuser.
        
        Input: [batch, seq, heads, dim]
        Output: [batch, seq, heads, dim]
        """
        if not HAS_FLASH_ATTN:
            raise RuntimeError(
                "flash-attn is required for TorchSP Ulysses. "
                "Install: pip install flash-attn --no-build-isolation"
            )
        
        # Apply RoPE if provided (matches xFuser's usp_attention pattern)
        if metadata.freqs_cis is not None:
            from comfy.ldm.flux.math import apply_rope
            # Transpose to [batch, heads, seq, dim] for apply_rope
            q_transposed = q.transpose(1, 2)
            k_transposed = k.transpose(1, 2)
            
            # apply_rope handles PE broadcast automatically (xFuser pattern)
            q_transposed, k_transposed = apply_rope(q_transposed, k_transposed, metadata.freqs_cis)
            
            # Transpose back to [batch, seq, heads, dim]
            q = q_transposed.transpose(1, 2)
            k = k_transposed.transpose(1, 2)
        
        # Calculate softmax scale (EXACT copy from yunchang)
        softmax_scale = q.shape[-1] ** (-0.5)
        
        # EXACT copy from yunchang/kernels/attention.py flash_attn_forward()
        if flash_attn.__version__ < '2.6.3':
            block_out, _, _, _, _, block_lse, _, _ = _flash_attn_forward(
                q,
                k,
                v,
                dropout_p=0.0,
                softmax_scale=softmax_scale,
                causal=False,
                window_size=(-1, -1),
                softcap=0.0,
                alibi_slopes=None,
                return_softmax=False,
            )
        else:
            block_out, block_lse, _, _ = _flash_attn_forward(
                q,
                k,
                v,
                dropout_p=0.0,
                softmax_scale=softmax_scale,
                causal=False,
                window_size_left=-1,
                window_size_right=-1,
                softcap=0.0,
                alibi_slopes=None,
                return_softmax=False,
            )
        
        return block_out
