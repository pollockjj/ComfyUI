"""PyIsolate Extension wrapper for isolated ComfyUI nodes."""

from __future__ import annotations

import asyncio
import inspect
import logging
import pickle
import traceback
import uuid
from dataclasses import asdict
from typing import Any, Dict, Tuple, List

from pyisolate import ExtensionBase

from comfy_api.internal import _ComfyNodeInternal
from comfy_api.latest import _io as latest_io

LOG_PREFIX = "[I]"
V3_DISCOVERY_TIMEOUT = 30  # seconds

logger = logging.getLogger(__name__)


def _sanitize_for_transport(value):
    """Convert arbitrary node metadata into transport-safe primitives."""
    primitives = (str, int, float, bool, type(None))
    if isinstance(value, primitives):
        return value
    
    # Check for special types BEFORE dict (since they may be dict subclasses)
    cls_name = value.__class__.__name__
    if cls_name == "FlexibleOptionalInputType":
        # FlexibleOptionalInputType is a dict subclass that accepts any key
        # We need to preserve this behavior across the process boundary
        flex_type = _sanitize_for_transport(getattr(value, "type", "*"))
        return {
            "__pyisolate_flexible_optional__": True,
            "type": flex_type,
        }
    if cls_name == "AnyType":
        return {"__pyisolate_any_type__": True, "value": str(value)}
    if cls_name == "ByPassTypeTuple":
        return {
            "__pyisolate_bypass_tuple__": [_sanitize_for_transport(v) for v in tuple(value)]
        }
    
    # Now check standard containers
    if isinstance(value, dict):
        return {k: _sanitize_for_transport(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return {"__pyisolate_tuple__": [_sanitize_for_transport(v) for v in value]}
    if isinstance(value, list):
        return [_sanitize_for_transport(v) for v in value]

    # Fallback: best-effort string conversion
    return str(value)


class RemoteObjectHandle:
    """Lightweight handle for objects kept in the isolated process."""
    # Explicitly set module for pickle compatibility
    __module__ = 'comfy.isolation.extension_wrapper'
    
    def __init__(self, object_id: str, type_name: str):
        self.object_id = object_id
        self.type_name = type_name
    
    def __repr__(self):
        return f"<RemoteObject id={self.object_id} type={self.type_name}>"


class ComfyNodeExtension(ExtensionBase):
    """Surface NODE_CLASS_MAPPINGS from isolated modules over RPC."""

    def __init__(self) -> None:
        super().__init__()
        self.node_classes: Dict[str, type] = {}
        self.display_names: Dict[str, str] = {}
        self.node_instances: Dict[str, Any] = {}
        self.remote_objects: Dict[str, Any] = {}  # Cache for objects kept in isolated process
        self._route_handlers: Dict[str, Any] = {}  # Cache for route handler functions
        self._module: Any = None  # Reference to loaded module

    async def on_module_loaded(self, module: Any) -> None:
        """Cache node metadata when the isolated module is imported."""
        self._module = module  # Keep reference for route handler lookup
        self.node_classes = getattr(module, "NODE_CLASS_MAPPINGS", {}) or {}
        self.display_names = getattr(module, "NODE_DISPLAY_NAME_MAPPINGS", {}) or {}

        # --- V3 Extension Support ---
        try:
            from comfy_api.latest import ComfyExtension
            
            # Find all ComfyExtension subclasses in the module
            for name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and issubclass(obj, ComfyExtension) and obj is not ComfyExtension:
                    # Check if it's defined in this module (optional, but good practice)
                    # We relax this check slightly to allow for split files, but ensure it's from the same package
                    if not obj.__module__.startswith(module.__name__):
                        continue
                        
                    logger.info(f"{LOG_PREFIX} Found V3 Extension: {name}")
                    try:
                        ext_instance = obj()
                        
                        # Call on_load with timeout
                        try:
                            await asyncio.wait_for(ext_instance.on_load(), timeout=V3_DISCOVERY_TIMEOUT)
                        except asyncio.TimeoutError:
                            logger.error(f"{LOG_PREFIX}[Loader] V3 Extension {name} timed out during on_load() after {V3_DISCOVERY_TIMEOUT}s")
                            continue
                        
                        # Call get_node_list with timeout
                        try:
                            v3_nodes = await asyncio.wait_for(ext_instance.get_node_list(), timeout=V3_DISCOVERY_TIMEOUT)
                        except asyncio.TimeoutError:
                            logger.error(f"{LOG_PREFIX}[Loader] V3 Extension {name} timed out during get_node_list() after {V3_DISCOVERY_TIMEOUT}s")
                            continue
                        
                        for node_cls in v3_nodes:
                            # Get schema to find node_id
                            if hasattr(node_cls, "GET_SCHEMA"):
                                schema = node_cls.GET_SCHEMA()
                                node_id = schema.node_id
                                self.node_classes[node_id] = node_cls
                                if schema.display_name:
                                    self.display_names[node_id] = schema.display_name
                                logger.info(f"{LOG_PREFIX} Registered V3 Node: {node_id}")
                    except Exception as e:
                        logger.error(f"{LOG_PREFIX} Failed to initialize V3 Extension {name}: {e}")
                        logger.error(f"{LOG_PREFIX} Traceback:\n{traceback.format_exc()}")
                        
        except ImportError:
            pass  # comfy_api not available
        except Exception as e:
            logger.error(f"{LOG_PREFIX} Error during V3 extension discovery: {e}")
            logger.error(f"{LOG_PREFIX} Traceback:\n{traceback.format_exc()}")
        # ----------------------------

        # Fix __module__ for all node classes to avoid pickle issues
        # When PyIsolate loads modules, __module__ can be set to the file path
        # instead of a proper module name, breaking pickle
        module_name = getattr(module, '__name__', 'isolated_nodes')
        for node_name, node_cls in self.node_classes.items():
            if hasattr(node_cls, '__module__') and '/' in str(node_cls.__module__):
                # Path detected in __module__, fix it
                node_cls.__module__ = module_name

        # Don't instantiate yet - wait until execution time
        # This avoids pickle issues with classes that have bad __module__ attributes
        self.node_instances = {}
        
        logger.info(
            "%s[ExtensionWrapper] Loaded %d nodes",
            LOG_PREFIX,
            len(self.node_classes),
        )

    async def list_nodes(self) -> Dict[str, str]:
        """Return mapping of node names to display names."""
        return {
            name: self.display_names.get(name, name)
            for name in self.node_classes
        }

    async def get_node_info(self, node_name: str) -> Dict[str, Any]:
        """Return minimal metadata; kept for backward compatibility."""
        return await self.get_node_details(node_name)

    async def get_node_details(self, node_name: str) -> Dict[str, Any]:
        """Get full node details - called during loading, must return JSON-serializable data."""
        node_cls = self._get_node_class(node_name)
        is_v3 = issubclass(node_cls, _ComfyNodeInternal)

        input_types_raw = node_cls.INPUT_TYPES() if hasattr(node_cls, "INPUT_TYPES") else {}
        input_types_safe = _sanitize_for_transport(input_types_raw)

        # Handle OUTPUT_IS_LIST
        output_is_list = getattr(node_cls, "OUTPUT_IS_LIST", None)
        if output_is_list is not None:
            # Convert to tuple of bools for JSON serialization
            output_is_list = tuple(bool(x) for x in output_is_list)

        details: Dict[str, Any] = {
            "input_types": input_types_safe,
            "return_types": tuple(str(t) for t in getattr(node_cls, "RETURN_TYPES", ())),
            "return_names": getattr(node_cls, "RETURN_NAMES", None),
            "function": str(getattr(node_cls, "FUNCTION", "execute")),
            "category": str(getattr(node_cls, "CATEGORY", "")),
            "output_node": bool(getattr(node_cls, "OUTPUT_NODE", False)),
            "output_is_list": output_is_list,
            "is_v3": is_v3,
        }

        if is_v3:
            try:
                schema = node_cls.GET_SCHEMA()
                schema_v1 = asdict(schema.get_v1_info(node_cls))
                try:
                    schema_v3 = asdict(schema.get_v3_info(node_cls))
                except TypeError:
                    # Fallback: build minimal v3 info without python_module bug
                    schema_v3 = self._build_schema_v3_fallback(schema)

                details.update(
                    {
                        "schema_v1": schema_v1,
                        "schema_v3": schema_v3,
                        "hidden": [h.value for h in (schema.hidden or [])],
                        "description": getattr(schema, "description", ""),
                        "deprecated": bool(getattr(node_cls, "DEPRECATED", False)),
                        "experimental": bool(getattr(node_cls, "EXPERIMENTAL", False)),
                        "api_node": bool(getattr(node_cls, "API_NODE", False)),
                        "input_is_list": bool(getattr(node_cls, "INPUT_IS_LIST", False)),
                        "not_idempotent": bool(getattr(node_cls, "NOT_IDEMPOTENT", False)),
                    }
                )
            except Exception as exc:  # pragma: no cover - defensive serialization path
                logger.warning(
                    "%s[ExtensionWrapper] Failed to serialize V3 schema for %s: %s",
                    LOG_PREFIX,
                    node_name,
                    exc,
                )
        return details

    def _build_schema_v3_fallback(self, schema) -> Dict[str, Any]:
        """Reconstruct v3 schema dict without using NodeInfoV3 constructor.

        This avoids upstream bugs when python_module is passed to NodeInfoV3.
        """
        input_dict: Dict[str, Any] = {}
        output_dict: Dict[str, Any] = {}
        hidden_list: List[str] = []

        if getattr(schema, "inputs", None):
            for input_obj in schema.inputs:
                latest_io.add_to_dict_v3(input_obj, input_dict)
        if getattr(schema, "outputs", None):
            for output_obj in schema.outputs:
                latest_io.add_to_dict_v3(output_obj, output_dict)
        if getattr(schema, "hidden", None):
            for hidden in schema.hidden:
                hidden_list.append(getattr(hidden, "value", str(hidden)))

        return {
            "input": input_dict,
            "output": output_dict,
            "hidden": hidden_list,
            "name": getattr(schema, "node_id", None),
            "display_name": getattr(schema, "display_name", None),
            "description": getattr(schema, "description", None),
            "category": getattr(schema, "category", None),
            "output_node": getattr(schema, "is_output_node", False),
            "deprecated": getattr(schema, "is_deprecated", False),
            "experimental": getattr(schema, "is_experimental", False),
            "api_node": getattr(schema, "is_api_node", False),
        }
    
    async def get_input_types(self, node_name: str) -> Dict[str, Any]:
        """Get INPUT_TYPES for a node - called on-demand when needed."""
        node_cls = self._get_node_class(node_name)
        if hasattr(node_cls, "INPUT_TYPES"):
            return node_cls.INPUT_TYPES()
        return {}

    async def execute_node(self, node_name: str, **inputs: Any) -> Tuple[Any, ...]:
        """Invoke the node's FUNCTION with provided inputs."""
        # Resolve any RemoteObjectHandles in inputs back to actual objects
        resolved_inputs = self._resolve_remote_objects(inputs)
        
        instance = self._get_node_instance(node_name)
        node_cls = self._get_node_class(node_name)
        function_name = getattr(node_cls, "FUNCTION", "execute")
        if not hasattr(instance, function_name):
            raise AttributeError(f"Node {node_name} missing callable '{function_name}'")
        method = getattr(instance, function_name)
        result = method(**resolved_inputs)
        
        # Unwrap V3 API NodeOutput to its underlying tuple
        type_name = type(result).__name__
        if type_name == 'NodeOutput':
            result = result.args
        
        if not isinstance(result, tuple):
            result = (result,)
        
        # Replace unpicklable objects with handles
        result = self._wrap_unpicklable_objects(result)
        return result

    def _is_picklable(self, obj: Any) -> bool:
        """Check if an object can be pickled AND unpickled across process boundary."""
        try:
            # Serialize
            serialized = pickle.dumps(obj)
            # The real test: can we deserialize it?
            # If the object references modules not in host, this will fail
            pickle.loads(serialized)
            return True
        except (pickle.PicklingError, TypeError, AttributeError, ModuleNotFoundError, ImportError):
            return False

    def _wrap_unpicklable_objects(self, data: Any) -> Any:
        """Recursively replace non-primitive objects with RemoteObjectHandles."""
        import torch
        
        type_name = type(data).__name__
        
        # Primitives: pass through
        if isinstance(data, (str, int, float, bool, type(None))):
            return data
        
        # Torch tensors: pass through (handled by share_torch)
        if isinstance(data, torch.Tensor):
            return data
        
        # ModelPatcherProxy: convert to ref (child returning to host)
        if type_name == 'ModelPatcherProxy':
            model_id = data._instance_id
            return {
                "__type__": "ModelPatcherRef",
                "model_id": model_id,
            }
        
        # CLIPProxy: convert to ref (child returning to host)
        if type_name == 'CLIPProxy':
            clip_id = data._instance_id
            return {
                "__type__": "CLIPRef",
                "clip_id": clip_id,
            }
        
        # Lists/tuples: recursively wrap contents
        if isinstance(data, (list, tuple)):
            wrapped = [self._wrap_unpicklable_objects(item) for item in data]
            return tuple(wrapped) if isinstance(data, tuple) else wrapped
        
        # Dicts: recursively wrap values
        if isinstance(data, dict):
            wrapped = {}
            for key, value in data.items():
                wrapped[key] = self._wrap_unpicklable_objects(value)
            return wrapped
        
        # Everything else: assume unpicklable, wrap it
        object_id = str(uuid.uuid4())
        self.remote_objects[object_id] = data
        type_name = type(data).__name__
        return RemoteObjectHandle(object_id, type_name)

    def _resolve_remote_objects(self, data: Any) -> Any:
        """Recursively replace RemoteObjectHandles and serialized refs with actual objects."""
        # Handle old RemoteObjectHandle format
        if isinstance(data, RemoteObjectHandle):
            if data.object_id not in self.remote_objects:
                raise KeyError(f"Remote object {data.object_id} not found in cache")
            return self.remote_objects[data.object_id]
        
        # Handle new serialization format (CLIPRef, ModelPatcherRef)
        if isinstance(data, dict):
            ref_type = data.get("__type__")
            
            if ref_type in ("CLIPRef", "ModelPatcherRef"):
                # Use pyisolate's deserialization to create proxy
                try:
                    from pyisolate._internal.model_serialization import deserialize_proxy_result
                    # This will create CLIPProxy or ModelPatcherProxy
                    return deserialize_proxy_result(data)
                except ImportError:
                    logger.error(
                        "%s[ExtensionWrapper] Cannot deserialize %s: pyisolate not available",
                        LOG_PREFIX,
                        ref_type,
                    )
                    raise
            
            # Regular dict - recurse
            return {
                key: self._resolve_remote_objects(value)
                for key, value in data.items()
            }
        
        if isinstance(data, (list, tuple)):
            resolved = [self._resolve_remote_objects(item) for item in data]
            return tuple(resolved) if isinstance(data, tuple) else resolved
        
        return data

    def _get_node_class(self, node_name: str) -> type:
        if node_name not in self.node_classes:
            raise KeyError(f"Unknown node: {node_name}")
        return self.node_classes[node_name]

    def _get_node_instance(self, node_name: str) -> Any:
        # Lazy instantiation - create instance on first use
        if node_name not in self.node_instances:
            if node_name not in self.node_classes:
                raise KeyError(f"Unknown node: {node_name}")
            self.node_instances[node_name] = self.node_classes[node_name]()
        return self.node_instances[node_name]

    async def before_module_loaded(self) -> None:
        """Patch ComfyUI singletons before loading the user module."""
        await super().before_module_loaded()
        # Progress proxy wiring for isolated children
        try:
            from comfy_api.latest import ComfyAPI_latest
            from .proxies.progress_proxy import ProgressProxy
            from .proxies.folder_paths_proxy import FolderPathsProxy
            import comfy_api.latest._ui as latest_ui
            import comfy_api.latest._resources as latest_resources

            # Replace Execution singleton with host-backed proxy
            ComfyAPI_latest.Execution = ProgressProxy
            ComfyAPI_latest.execution = ProgressProxy()
            # Align folder_paths usage for V3 helpers
            fp_proxy = FolderPathsProxy()
            latest_ui.folder_paths = fp_proxy
            latest_resources.folder_paths = fp_proxy
            logger.debug("%s[ExtensionWrapper] Progress proxy wired for ComfyAPI_latest", LOG_PREFIX)
        except Exception as exc:  # pragma: no cover
            logger.debug("%s[ExtensionWrapper] Progress proxy not applied: %s", LOG_PREFIX, exc)

    # =========================================================================
    # Route Handler Support (Rev 1.0)
    # =========================================================================
    
    async def call_route_handler(
        self,
        handler_module: str,
        handler_func: str,
        request_data: Dict[str, Any],
    ) -> Any:
        """Execute a route handler and return serializable result.
        
        Called by the host RouteInjector to forward HTTP requests to
        the isolated process.
        
        Args:
            handler_module: Module name relative to node root (e.g., "image_filter_messaging")
            handler_func: Function name (e.g., "cg_image_filter_message")
            request_data: Serialized request dict
        
        Returns:
            Serializable response dict
        """
        import asyncio
        import importlib
        import importlib.util
        import sys
        import os
        
        # Get or cache handler function
        cache_key = f"{handler_module}.{handler_func}"
        if cache_key not in self._route_handlers:
            try:
                # Get node directory from loaded module
                if self._module is not None and hasattr(self._module, '__file__'):
                    node_dir = os.path.dirname(self._module.__file__)
                    # Add node directory to sys.path temporarily
                    if node_dir not in sys.path:
                        sys.path.insert(0, node_dir)
                
                # Try to import the module
                module = importlib.import_module(handler_module)
                handler = getattr(module, handler_func)
                self._route_handlers[cache_key] = handler
            except (ImportError, AttributeError) as e:
                logger.error(
                    "%s[RouteHandler]  Failed to load handler %s: %s",
                    LOG_PREFIX,
                    cache_key,
                    e,
                )
                raise ValueError(f"Route handler not found: {cache_key}")
        
        handler = self._route_handlers[cache_key]
        
        # Create mock request
        mock_request = MockRequest(request_data)
        
        # Call handler
        if asyncio.iscoroutinefunction(handler):
            result = await handler(mock_request)
        else:
            result = handler(mock_request)
        
        # Serialize response
        return self._serialize_response(result)
    
    def _serialize_response(self, response: Any) -> Dict[str, Any]:
        """Convert aiohttp-style response to serializable dict."""
        # None response
        if response is None:
            return {"type": "text", "body": "", "status": 204}
        
        # Dict response (most common)
        if isinstance(response, dict):
            return {"type": "json", "body": response, "status": 200}
        
        # String response
        if isinstance(response, str):
            return {"type": "text", "body": response, "status": 200}
        
        # aiohttp Response objects
        if hasattr(response, 'text') and hasattr(response, 'status'):
            # web.Response with text
            return {
                "type": "text",
                "body": response.text if hasattr(response, 'text') else str(response.body),
                "status": response.status,
                "headers": dict(response.headers) if hasattr(response, 'headers') else {},
            }
        
        if hasattr(response, 'body') and hasattr(response, 'status'):
            # Generic response with body
            body = response.body
            if isinstance(body, bytes):
                try:
                    body = body.decode('utf-8')
                    return {"type": "text", "body": body, "status": response.status}
                except UnicodeDecodeError:
                    return {"type": "binary", "body": body.hex(), "status": response.status}
            return {"type": "json", "body": body, "status": response.status}
        
        # Fallback: convert to string
        return {"type": "text", "body": str(response), "status": 200}


class MockRequest:
    """Mock aiohttp Request for isolated route handler execution.
    
    Provides the interface that route handlers expect from aiohttp.web.Request.
    """
    
    def __init__(self, data: Dict[str, Any]):
        self.method = data.get("method", "GET")
        self.path = data.get("path", "/")
        self.query = data.get("query", {})
        self._body = data.get("body", {})
        self._text = data.get("text", "")
        self.headers = data.get("headers", {})
        self.content_type = data.get("content_type", self.headers.get("Content-Type", "application/json"))
        self.match_info = data.get("match_info", {})
    
    async def json(self) -> Any:
        """Get request body as JSON."""
        if isinstance(self._body, dict):
            return self._body
        if isinstance(self._body, str):
            import json
            return json.loads(self._body)
        return {}
    
    async def post(self) -> Dict[str, Any]:
        """Get form data."""
        if isinstance(self._body, dict):
            return self._body
        return {}
    
    async def text(self) -> str:
        """Get request body as text."""
        if self._text:
            return self._text
        if isinstance(self._body, str):
            return self._body
        if isinstance(self._body, dict):
            import json
            return json.dumps(self._body)
        return ""
    
    async def read(self) -> bytes:
        """Get raw request body as bytes."""
        text = await self.text()
        return text.encode('utf-8')
