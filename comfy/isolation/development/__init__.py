"""Experimental isolation features (gated behind IS_DEV flag).

V1.0 ships with core dependency isolation only. Advanced features like
CLIP/ModelPatcher serialization and HTTP route forwarding are considered
experimental and require explicit opt-in.

Enable with: export PYISOLATE_DEV=1
"""
import os

IS_DEV = os.environ.get("PYISOLATE_DEV") == "1"

if IS_DEV:
    from .clip_proxy import CLIPRegistry, CLIPProxy, maybe_wrap_clip_for_isolation
    from .model_patcher_proxy import (
        ModelPatcherRegistry,
        ModelPatcherProxy,
        maybe_wrap_model_for_isolation,
    )
    from .rpc_bridge import RpcBridge, get_rpc_bridge
    from .rpc_handlers import rpc_execute_model_method
    from .route_extractor import extract_routes, generate_route_manifest
    from .route_injector import inject_routes
    
    __all__ = [
        "CLIPRegistry",
        "CLIPProxy",
        "maybe_wrap_clip_for_isolation",
        "ModelPatcherRegistry",
        "ModelPatcherProxy",
        "maybe_wrap_model_for_isolation",
        "RpcBridge",
        "get_rpc_bridge",
        "rpc_execute_model_method",
        "extract_routes",
        "generate_route_manifest",
        "inject_routes",
    ]
else:
    # Stub exports for when IS_DEV is disabled
    __all__ = []
