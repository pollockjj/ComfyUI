"""Shim to expose ModelPatcherRegistry/Proxy outside development namespace."""
from .development.model_patcher_proxy import ModelPatcherProxy, ModelPatcherRegistry

__all__ = ["ModelPatcherProxy", "ModelPatcherRegistry"]
