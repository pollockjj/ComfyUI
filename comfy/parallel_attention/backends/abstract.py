"""Abstract interfaces for the Parallel Attention Backend Plugin System.

Canonical contracts:
- AttentionMetadata carries backend-specific context (initial fields cover RoPE).
- AttentionBackend acts as a lightweight factory bound to an AttentionImpl.
- AttentionImpl instances own the functional kernels and pre/post hooks.

RoPE handling lives in AttentionImpl.preprocess_qkv(). Forward hooks must remain
backend-agnostic and never inline rotary logic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import torch
from torch import Tensor
from torch.distributed import ProcessGroup


@dataclass(slots=True)
class AttentionMetadata:
    """Container for backend-specific context passed through attention calls.

    Canonical fields
    -----------------
    freqs_cis: Tensor | None
        Canonical RoPE tensor with shape (batch, seq, rotary_dim, 2, 2).
    seq_lens: Tensor | None
        Optional per-token lengths for backends that need masking context.
    extras: Mapping[str, Any]
        Free-form metadata reserved for future extensions (kept lightweight).
    """

    freqs_cis: Optional[Tensor] = None
    seq_lens: Optional[Tensor] = None
    extras: Mapping[str, Any] = field(default_factory=dict)


class AttentionBackend(ABC):
    """Abstract factory describing an attention backend implementation."""

    @staticmethod
    @abstractmethod
    def get_name() -> str:
        """Return the stable backend identifier (e.g. 'XFUSER_USP')."""

    @staticmethod
    @abstractmethod
    def get_impl_cls() -> type["AttentionImpl"]:
        """Return the concrete AttentionImpl class bound to this backend."""

    @staticmethod
    def get_metadata_cls() -> type[AttentionMetadata]:
        """Return the metadata dataclass used for this backend."""

        return AttentionMetadata


class AttentionImpl(ABC):
    """Base class for backend-specific attention kernels and hooks."""

    def __init__(
        self,
        *,
        num_heads: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
        process_group: Optional[ProcessGroup] = None,
    ) -> None:
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype
        self.process_group = process_group

    def preprocess_qkv(self, qkv: Tensor, metadata: AttentionMetadata) -> Tensor:
        """Optional hook to massage QKV before the core attention call.

        Default implementation is a no-op. Subclasses should override to apply
        RoPE or perform backend-specific layout conversions.
        """

        return qkv

    def postprocess_output(self, output: Tensor, metadata: AttentionMetadata) -> Tensor:
        """Optional hook to transform attention outputs prior to all-gather."""

        return output

    @abstractmethod
    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        metadata: AttentionMetadata,
        *,
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute attention using backend-specific kernels."""
        ...