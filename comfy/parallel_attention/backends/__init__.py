"""Backend Plugin System scaffolding for parallel attention."""

from .abstract import AttentionBackend, AttentionImpl, AttentionMetadata
from .selector import (
    AttentionBackendEnum,
    get_attn_backend,
    global_force_attn_backend,
    register_backend,
)
from .xfuser_backend import XFuserUSPBackend

# Auto-register backends on module import
register_backend(AttentionBackendEnum.XFUSER_USP, XFuserUSPBackend)

__all__ = [
    "AttentionBackend",
    "AttentionImpl",
    "AttentionMetadata",
    "AttentionBackendEnum",
    "get_attn_backend",
    "global_force_attn_backend",
    "register_backend",
]
