"""Backend selector and registry for the Parallel Attention Plugin System.

Process Group Ownership Model
------------------------------
Backends consume process groups provided by the worker; they must never call
`dist.init_process_group()` or create groups internally. The `DistributedAttention`
layer receives a group from the FSDP2 worker and passes it to the backend impl
during instantiation.
"""

from __future__ import annotations

import logging
import os
from enum import Enum
from typing import Optional

import torch

from .abstract import AttentionBackend

LOGGER = logging.getLogger(__name__)
LOG_PREFIX = "⚡ [Parallel-Attention][BPS-Selector]"


class AttentionBackendEnum(Enum):
    """Registry of supported attention backend implementations."""

    XFUSER_USP = "xfuser_usp"
    TORCH_SP_ULYSSES = "torch_sp_ulysses"


_BACKEND_REGISTRY: dict[AttentionBackendEnum, type[AttentionBackend]] = {}
_FORCED_BACKEND: Optional[AttentionBackendEnum] = None


def register_backend(
    backend_enum: AttentionBackendEnum,
    backend_cls: type[AttentionBackend],
) -> None:
    """Register a concrete backend class in the global registry."""
    _BACKEND_REGISTRY[backend_enum] = backend_cls
    LOGGER.debug("%s Registered backend: %s → %s", LOG_PREFIX, backend_enum.value, backend_cls.__name__)


def get_attn_backend(
    head_dim: int,
    dtype: torch.dtype,
    supported_backends: Optional[tuple[AttentionBackendEnum, ...]] = None,
) -> type[AttentionBackend]:
    """Resolve the attention backend class for the given constraints.

    Selection priority:
    1. Global forced backend (set via `global_force_attn_backend`).
    2. Environment variable `ATTENTION_BACKEND`.
    3. First entry in `supported_backends` tuple.
    4. Default: `XFUSER_USP`.

    Args:
        head_dim: Head dimension for the attention layer (reserved for future routing logic).
        dtype: Data type for the attention computation (reserved for future routing).
        supported_backends: Ordered tuple of acceptable backends for this layer.

    Returns:
        The resolved `AttentionBackend` class.

    Raises:
        ValueError: If the resolved backend is not in the registry or not supported.
    """
    if _FORCED_BACKEND is not None:
        selected = _FORCED_BACKEND
        LOGGER.debug("%s Using forced backend: %s", LOG_PREFIX, selected.value)
    else:
        env_backend = os.getenv("ATTENTION_BACKEND", "").strip().lower()
        if env_backend:
            try:
                selected = AttentionBackendEnum(env_backend)
                LOGGER.debug("%s Using backend from ATTENTION_BACKEND env: %s", LOG_PREFIX, selected.value)
            except ValueError:
                LOGGER.warning(
                    "%s Invalid ATTENTION_BACKEND=%s, falling back to default",
                    LOG_PREFIX,
                    env_backend,
                )
                selected = AttentionBackendEnum.XFUSER_USP
        elif supported_backends:
            selected = supported_backends[0]
            LOGGER.debug("%s Using first supported backend: %s", LOG_PREFIX, selected.value)
        else:
            selected = AttentionBackendEnum.XFUSER_USP
            LOGGER.debug("%s Using default backend: %s", LOG_PREFIX, selected.value)

    if supported_backends and selected not in supported_backends:
        raise ValueError(
            f"Backend {selected.value} is not in the supported list: "
            f"{[b.value for b in supported_backends]}"
        )

    backend_cls = _BACKEND_REGISTRY.get(selected)
    if backend_cls is None:
        raise ValueError(
            f"Backend {selected.value} is not registered. "
            f"Available: {[b.value for b in _BACKEND_REGISTRY.keys()]}"
        )

    return backend_cls


def global_force_attn_backend(backend: Optional[AttentionBackendEnum]) -> None:
    """Override backend selection globally for testing/profiling."""
    global _FORCED_BACKEND
    _FORCED_BACKEND = backend
    if backend:
        LOGGER.info("%s Forcing backend globally: %s", LOG_PREFIX, backend.value)
    else:
        LOGGER.info("%s Cleared forced backend", LOG_PREFIX)
