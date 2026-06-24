"""Compatibility shim for the indexed serializer path."""

from __future__ import annotations

from typing import Any


def register_custom_node_serializers(_registry: Any) -> None:
    """Legacy no-op shim; serializer registration lives in the isolation adapter, kept importable because the isolation index still references it."""
    return None

__all__ = ["register_custom_node_serializers"]
