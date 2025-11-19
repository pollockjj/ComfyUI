"""Backend Plugin System scaffolding for parallel attention."""

from .abstract import AttentionBackend, AttentionImpl, AttentionMetadata
from .selector import (
    AttentionBackendEnum,
    get_attn_backend,
    global_force_attn_backend,
    register_backend,
)
from .xfuser_backend import XFuserUSPBackend
from .torch_sp_ulysses_backend import TorchSPUlyssesBackend

# Auto-register backends on module import
register_backend(AttentionBackendEnum.XFUSER_USP, XFuserUSPBackend)
register_backend(AttentionBackendEnum.TORCH_SP_ULYSSES, TorchSPUlyssesBackend)

__all__ = [
    "AttentionBackend",
    "AttentionImpl",
    "AttentionMetadata",
    "AttentionBackendEnum",
    "get_attn_backend",
    "global_force_attn_backend",
    "register_backend",
]
