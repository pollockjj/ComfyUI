from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from pyisolate.interfaces import IsolationAdapter, SerializerRegistryProtocol  # type: ignore[import-untyped]
from pyisolate._internal.shared import AsyncRPC, ProxiedSingleton  # type: ignore[import-untyped]

try:
    import folder_paths
    from comfy.isolation.clip_proxy import CLIPProxy, CLIPRegistry
    from comfy.isolation.model_patcher_proxy import ModelPatcherProxy, ModelPatcherRegistry
    from comfy.isolation.model_sampling_proxy import ModelSamplingProxy, ModelSamplingRegistry
    from comfy.isolation.vae_proxy import VAEProxy, VAERegistry
    from comfy.isolation.proxies.folder_paths_proxy import FolderPathsProxy
    from comfy.isolation.proxies.prompt_server_proxy import PromptServerProxy
except ImportError as exc:  # Fail loud if Comfy environment is incomplete
    raise ImportError(f"ComfyUI environment incomplete: {exc}")

logger = logging.getLogger(__name__)


class ComfyUIAdapter(IsolationAdapter):
    """ComfyUI-specific IsolationAdapter implementation."""

    @property
    def identifier(self) -> str:
        return "comfyui"

    def get_path_config(self, module_path: str) -> Optional[Dict[str, Any]]:
        if "ComfyUI" in module_path and "custom_nodes" in module_path:
            parts = module_path.split("ComfyUI")
            if len(parts) > 1:
                comfy_root = parts[0] + "ComfyUI"
                return {
                    "preferred_root": comfy_root,
                    "additional_paths": [
                        os.path.join(comfy_root, "custom_nodes"),
                        os.path.join(comfy_root, "comfy"),
                    ],
                }
        return None

    def setup_child_environment(self, snapshot: Dict[str, Any]) -> None:
        comfy_root = snapshot.get("preferred_root")
        if not comfy_root:
            return

        requirements_path = Path(comfy_root) / "requirements.txt"
        if requirements_path.exists():
            import re

            for line in requirements_path.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                pkg_name = re.split(r"[<>=!~\[]", line)[0].strip()
                if pkg_name:
                    logging.getLogger(pkg_name).setLevel(logging.ERROR)

    def register_serializers(self, registry: SerializerRegistryProtocol) -> None:
        def serialize_model_patcher(obj: Any) -> Dict[str, Any]:
            # Child-side: must already have _instance_id (proxy)
            if os.environ.get("PYISOLATE_CHILD") == "1":
                if hasattr(obj, "_instance_id"):
                    return {"__type__": "ModelPatcherRef", "model_id": obj._instance_id}
                raise RuntimeError(
                    f"ModelPatcher in child lacks _instance_id: "
                    f"{type(obj).__module__}.{type(obj).__name__}"
                )
            # Host-side: register with registry
            if hasattr(obj, "_instance_id"):
                return {"__type__": "ModelPatcherRef", "model_id": obj._instance_id}
            model_id = ModelPatcherRegistry().register(obj)
            return {"__type__": "ModelPatcherRef", "model_id": model_id}

        def deserialize_model_patcher(data: Dict[str, Any]) -> ModelPatcherProxy:
            """Child-side deserializer: create proxy."""
            return ModelPatcherProxy(data["model_id"], registry=None, manage_lifecycle=False)

        def deserialize_model_patcher_ref(data: Dict[str, Any]) -> Any:
            """Context-aware ModelPatcherRef deserializer for both host and child."""
            is_child = os.environ.get("PYISOLATE_CHILD") == "1"
            if is_child:
                return ModelPatcherProxy(data["model_id"], registry=None, manage_lifecycle=False)
            else:
                return ModelPatcherRegistry()._get_instance(data["model_id"])

        # Register ModelPatcher type for serialization
        registry.register("ModelPatcher", serialize_model_patcher, deserialize_model_patcher)
        # Register ModelPatcherProxy type (already a proxy, just return ref)
        registry.register("ModelPatcherProxy", serialize_model_patcher, deserialize_model_patcher)
        # Register ModelPatcherRef for deserialization (context-aware: host or child)
        registry.register("ModelPatcherRef", None, deserialize_model_patcher_ref)

        def serialize_clip(obj: Any) -> Dict[str, Any]:
            if hasattr(obj, "_instance_id"):
                return {"__type__": "CLIPRef", "clip_id": obj._instance_id}
            clip_id = CLIPRegistry().register(obj)
            return {"__type__": "CLIPRef", "clip_id": clip_id}

        def deserialize_clip(data: Dict[str, Any]) -> CLIPProxy:
            return CLIPProxy(data["clip_id"], registry=None, manage_lifecycle=False)

        def deserialize_clip_ref(data: Dict[str, Any]) -> Any:
            """Context-aware CLIPRef deserializer for both host and child."""
            is_child = os.environ.get("PYISOLATE_CHILD") == "1"
            if is_child:
                return CLIPProxy(data["clip_id"], registry=None, manage_lifecycle=False)
            else:
                return CLIPRegistry()._get_instance(data["clip_id"])

        # Register CLIP type for serialization
        registry.register("CLIP", serialize_clip, deserialize_clip)
        # Register CLIPProxy type (already a proxy, just return ref)
        registry.register("CLIPProxy", serialize_clip, deserialize_clip)
        # Register CLIPRef for deserialization (context-aware: host or child)
        registry.register("CLIPRef", None, deserialize_clip_ref)

        def serialize_vae(obj: Any) -> Dict[str, Any]:
            if hasattr(obj, "_instance_id"):
                return {"__type__": "VAERef", "vae_id": obj._instance_id}
            vae_id = VAERegistry().register(obj)
            return {"__type__": "VAERef", "vae_id": vae_id}

        def deserialize_vae(data: Dict[str, Any]) -> VAEProxy:
            return VAEProxy(data["vae_id"])

        def deserialize_vae_ref(data: Dict[str, Any]) -> Any:
            """Context-aware VAERef deserializer for both host and child."""
            is_child = os.environ.get("PYISOLATE_CHILD") == "1"
            if is_child:
                # Child: create a proxy
                return VAEProxy(data["vae_id"])
            else:
                # Host: lookup real VAE from registry
                return VAERegistry()._get_instance(data["vae_id"])

        # Register VAE type for serialization
        registry.register("VAE", serialize_vae, deserialize_vae)
        # Register VAEProxy type (already a proxy, just return ref)
        registry.register("VAEProxy", serialize_vae, deserialize_vae)
        # Register VAERef for deserialization (context-aware: host or child)
        registry.register("VAERef", None, deserialize_vae_ref)

        # ModelSampling serialization - handles ModelSampling* types
        # Note: The registry is type-name based, so we register common variants
        import copyreg

        def serialize_model_sampling(obj: Any) -> Dict[str, Any]:
            # Child-side: must already have _instance_id (proxy)
            if os.environ.get("PYISOLATE_CHILD") == "1":
                if hasattr(obj, "_instance_id"):
                    return {"__type__": "ModelSamplingRef", "ms_id": obj._instance_id}
                raise RuntimeError(
                    f"ModelSampling in child lacks _instance_id: "
                    f"{type(obj).__module__}.{type(obj).__name__}"
                )
            # Host-side: register with registry and set up copyreg
            def _reduce_model_sampling(ms: Any) -> tuple[type[ModelSamplingProxy], tuple[int]]:
                ms_id_local = ModelSamplingRegistry().register(ms)
                return (ModelSamplingProxy, (ms_id_local,))

            copyreg.pickle(type(obj), _reduce_model_sampling)

            ms_id = ModelSamplingRegistry().register(obj)
            return {"__type__": "ModelSamplingRef", "ms_id": ms_id}

        def deserialize_model_sampling(data: Dict[str, Any]) -> ModelSamplingProxy:
            """Child-side deserializer: create proxy."""
            return ModelSamplingProxy(data["ms_id"])

        def deserialize_model_sampling_ref(data: Dict[str, Any]) -> Any:
            """Context-aware ModelSamplingRef deserializer for both host and child."""
            is_child = os.environ.get("PYISOLATE_CHILD") == "1"
            if is_child:
                return ModelSamplingProxy(data["ms_id"])
            else:
                return ModelSamplingRegistry()._get_instance(data["ms_id"])

        # Register ModelSampling type and proxy
        registry.register("ModelSamplingDiscrete", serialize_model_sampling, deserialize_model_sampling)
        registry.register("ModelSamplingContinuousEDM", serialize_model_sampling, deserialize_model_sampling)
        registry.register("ModelSamplingContinuousV", serialize_model_sampling, deserialize_model_sampling)
        registry.register("ModelSamplingProxy", serialize_model_sampling, deserialize_model_sampling)
        # Register ModelSamplingRef for deserialization (context-aware: host or child)
        registry.register("ModelSamplingRef", None, deserialize_model_sampling_ref)

        # NodeOutput is a V3 API container - deserialize by unwrapping .args
        # This is an async-aware deserializer that recurses into the args
        def deserialize_node_output(data: Any) -> Any:
            """Unwrap NodeOutput container to its args for deserialization."""
            # Return the args attribute for further deserialization
            # The caller (deserialize_from_isolation) will recurse into this
            return getattr(data, 'args', data)

        registry.register("NodeOutput", None, deserialize_node_output)

        logger.info("Registered ComfyUI serializers: ModelPatcher, CLIP, VAE, ModelSampling, NodeOutput")

    def provide_rpc_services(self) -> List[type[ProxiedSingleton]]:
        return [PromptServerProxy, FolderPathsProxy]

    def handle_api_registration(self, api: ProxiedSingleton, rpc: AsyncRPC) -> None:
        api_name = api.__class__.__name__

        if api_name == "PromptServerProxy":
            # Defer heavy import to child context
            import server

            proxy = api.instance
            original_register_route = proxy.register_route

            def register_route_wrapper(method: str, path: str, handler: Callable[..., Any]) -> None:
                callback_id = rpc.register_callback(handler)
                loop = getattr(rpc, "loop", None)
                if loop and loop.is_running():
                    import asyncio

                    asyncio.create_task(
                        original_register_route(method, path, handler=callback_id, is_callback=True)
                    )
                else:
                    original_register_route(method, path, handler=callback_id, is_callback=True)
                return None

            proxy.register_route = register_route_wrapper

            class RouteTableDefProxy:
                def __init__(self, proxy_instance: Any):
                    self.proxy = proxy_instance

                def get(self, path: str, **kwargs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
                    def decorator(handler: Callable[..., Any]) -> Callable[..., Any]:
                        self.proxy.register_route("GET", path, handler)
                        return handler

                    return decorator

                def post(self, path: str, **kwargs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
                    def decorator(handler: Callable[..., Any]) -> Callable[..., Any]:
                        self.proxy.register_route("POST", path, handler)
                        return handler

                    return decorator

                def patch(self, path: str, **kwargs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
                    def decorator(handler: Callable[..., Any]) -> Callable[..., Any]:
                        self.proxy.register_route("PATCH", path, handler)
                        return handler

                    return decorator

                def put(self, path: str, **kwargs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
                    def decorator(handler: Callable[..., Any]) -> Callable[..., Any]:
                        self.proxy.register_route("PUT", path, handler)
                        return handler

                    return decorator

                def delete(self, path: str, **kwargs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
                    def decorator(handler: Callable[..., Any]) -> Callable[..., Any]:
                        self.proxy.register_route("DELETE", path, handler)
                        return handler

                    return decorator

            proxy.routes = RouteTableDefProxy(proxy)

            if hasattr(server, "PromptServer") and getattr(server.PromptServer, "instance", None) != proxy:
                server.PromptServer.instance = proxy
