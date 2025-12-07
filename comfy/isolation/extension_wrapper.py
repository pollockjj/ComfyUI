from __future__ import annotations

import asyncio
import importlib
import inspect
import json
import logging
import os
import pickle
import sys
import uuid
from dataclasses import asdict
from typing import Any, Dict, List, Tuple

from pyisolate import ExtensionBase

from comfy_api.internal import _ComfyNodeInternal
from comfy_api.latest import _io as latest_io

LOG_PREFIX = "]["
V3_DISCOVERY_TIMEOUT = 30

logger = logging.getLogger(__name__)


def _sanitize_for_transport(value):
    primitives = (str, int, float, bool, type(None))
    if isinstance(value, primitives):
        return value

    cls_name = value.__class__.__name__
    if cls_name == "FlexibleOptionalInputType":
        return {
            "__pyisolate_flexible_optional__": True,
            "type": _sanitize_for_transport(getattr(value, "type", "*")),
        }
    if cls_name == "AnyType":
        return {"__pyisolate_any_type__": True, "value": str(value)}
    if cls_name == "ByPassTypeTuple":
        return {"__pyisolate_bypass_tuple__": [_sanitize_for_transport(v) for v in tuple(value)]}

    if isinstance(value, dict):
        return {k: _sanitize_for_transport(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return {"__pyisolate_tuple__": [_sanitize_for_transport(v) for v in value]}
    if isinstance(value, list):
        return [_sanitize_for_transport(v) for v in value]

    return str(value)


class RemoteObjectHandle:
    __module__ = 'comfy.isolation.extension_wrapper'

    def __init__(self, object_id: str, type_name: str):
        self.object_id = object_id
        self.type_name = type_name

    def __repr__(self):
        return f"<RemoteObject id={self.object_id} type={self.type_name}>"


class ComfyNodeExtension(ExtensionBase):
    def __init__(self) -> None:
        super().__init__()
        self.node_classes: Dict[str, type] = {}
        self.display_names: Dict[str, str] = {}
        self.node_instances: Dict[str, Any] = {}
        self.remote_objects: Dict[str, Any] = {}
        self._route_handlers: Dict[str, Any] = {}
        self._module: Any = None

    async def on_module_loaded(self, module: Any) -> None:
        self._module = module
        self.node_classes = getattr(module, "NODE_CLASS_MAPPINGS", {}) or {}
        self.display_names = getattr(module, "NODE_DISPLAY_NAME_MAPPINGS", {}) or {}

        try:
            from comfy_api.latest import ComfyExtension
            for name, obj in inspect.getmembers(module):
                if not (inspect.isclass(obj) and issubclass(obj, ComfyExtension) and obj is not ComfyExtension):
                    continue
                if not obj.__module__.startswith(module.__name__):
                    continue
                try:
                    ext_instance = obj()
                    try:
                        await asyncio.wait_for(ext_instance.on_load(), timeout=V3_DISCOVERY_TIMEOUT)
                    except asyncio.TimeoutError:
                        logger.error("%s[Loader] V3 Extension %s timed out in on_load()", LOG_PREFIX, name)
                        continue
                    try:
                        v3_nodes = await asyncio.wait_for(ext_instance.get_node_list(), timeout=V3_DISCOVERY_TIMEOUT)
                    except asyncio.TimeoutError:
                        logger.error("%s[Loader] V3 Extension %s timed out in get_node_list()", LOG_PREFIX, name)
                        continue
                    for node_cls in v3_nodes:
                        if hasattr(node_cls, "GET_SCHEMA"):
                            schema = node_cls.GET_SCHEMA()
                            self.node_classes[schema.node_id] = node_cls
                            if schema.display_name:
                                self.display_names[schema.node_id] = schema.display_name
                except Exception as e:
                    logger.error("%s[Loader] V3 Extension %s failed: %s", LOG_PREFIX, name, e)
        except ImportError:
            pass

        module_name = getattr(module, '__name__', 'isolated_nodes')
        for node_cls in self.node_classes.values():
            if hasattr(node_cls, '__module__') and '/' in str(node_cls.__module__):
                node_cls.__module__ = module_name

        self.node_instances = {}

    async def list_nodes(self) -> Dict[str, str]:
        return {name: self.display_names.get(name, name) for name in self.node_classes}

    async def get_node_info(self, node_name: str) -> Dict[str, Any]:
        return await self.get_node_details(node_name)

    async def get_node_details(self, node_name: str) -> Dict[str, Any]:
        node_cls = self._get_node_class(node_name)
        is_v3 = issubclass(node_cls, _ComfyNodeInternal)

        input_types_raw = node_cls.INPUT_TYPES() if hasattr(node_cls, "INPUT_TYPES") else {}
        output_is_list = getattr(node_cls, "OUTPUT_IS_LIST", None)
        if output_is_list is not None:
            output_is_list = tuple(bool(x) for x in output_is_list)

        details: Dict[str, Any] = {
            "input_types": _sanitize_for_transport(input_types_raw),
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
                    schema_v3 = self._build_schema_v3_fallback(schema)
                details.update({
                    "schema_v1": schema_v1,
                    "schema_v3": schema_v3,
                    "hidden": [h.value for h in (schema.hidden or [])],
                    "description": getattr(schema, "description", ""),
                    "deprecated": bool(getattr(node_cls, "DEPRECATED", False)),
                    "experimental": bool(getattr(node_cls, "EXPERIMENTAL", False)),
                    "api_node": bool(getattr(node_cls, "API_NODE", False)),
                    "input_is_list": bool(getattr(node_cls, "INPUT_IS_LIST", False)),
                    "not_idempotent": bool(getattr(node_cls, "NOT_IDEMPOTENT", False)),
                })
            except Exception as exc:
                logger.warning("%s[Loader] V3 schema serialization failed for %s: %s", LOG_PREFIX, node_name, exc)
        return details

    def _build_schema_v3_fallback(self, schema) -> Dict[str, Any]:
        input_dict: Dict[str, Any] = {}
        output_dict: Dict[str, Any] = {}
        hidden_list: List[str] = []

        if getattr(schema, "inputs", None):
            for inp in schema.inputs:
                latest_io.add_to_dict_v3(inp, input_dict)
        if getattr(schema, "outputs", None):
            for out in schema.outputs:
                latest_io.add_to_dict_v3(out, output_dict)
        if getattr(schema, "hidden", None):
            for h in schema.hidden:
                hidden_list.append(getattr(h, "value", str(h)))

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
        node_cls = self._get_node_class(node_name)
        if hasattr(node_cls, "INPUT_TYPES"):
            return node_cls.INPUT_TYPES()
        return {}

    async def execute_node(self, node_name: str, **inputs: Any) -> Tuple[Any, ...]:
        resolved_inputs = self._resolve_remote_objects(inputs)
        instance = self._get_node_instance(node_name)
        node_cls = self._get_node_class(node_name)
        function_name = getattr(node_cls, "FUNCTION", "execute")
        if not hasattr(instance, function_name):
            raise AttributeError(f"Node {node_name} missing callable '{function_name}'")

        result = getattr(instance, function_name)(**resolved_inputs)
        if type(result).__name__ == 'NodeOutput':
            result = result.args
        if not isinstance(result, tuple):
            result = (result,)
        return self._wrap_unpicklable_objects(result)

    def _wrap_unpicklable_objects(self, data: Any) -> Any:
        import torch

        if isinstance(data, (str, int, float, bool, type(None))):
            return data
        if isinstance(data, torch.Tensor):
            return data

        type_name = type(data).__name__
        if type_name == 'ModelPatcherProxy':
            return {"__type__": "ModelPatcherRef", "model_id": data._instance_id}
        if type_name == 'CLIPProxy':
            return {"__type__": "CLIPRef", "clip_id": data._instance_id}

        if isinstance(data, (list, tuple)):
            wrapped = [self._wrap_unpicklable_objects(item) for item in data]
            return tuple(wrapped) if isinstance(data, tuple) else wrapped
        if isinstance(data, dict):
            return {k: self._wrap_unpicklable_objects(v) for k, v in data.items()}

        object_id = str(uuid.uuid4())
        self.remote_objects[object_id] = data
        return RemoteObjectHandle(object_id, type(data).__name__)

    def _resolve_remote_objects(self, data: Any) -> Any:
        if isinstance(data, RemoteObjectHandle):
            if data.object_id not in self.remote_objects:
                raise KeyError(f"Remote object {data.object_id} not found")
            return self.remote_objects[data.object_id]

        if isinstance(data, dict):
            ref_type = data.get("__type__")
            if ref_type in ("CLIPRef", "ModelPatcherRef"):
                from pyisolate._internal.model_serialization import deserialize_proxy_result
                return deserialize_proxy_result(data)
            return {k: self._resolve_remote_objects(v) for k, v in data.items()}

        if isinstance(data, (list, tuple)):
            resolved = [self._resolve_remote_objects(item) for item in data]
            return tuple(resolved) if isinstance(data, tuple) else resolved
        return data

    def _get_node_class(self, node_name: str) -> type:
        if node_name not in self.node_classes:
            raise KeyError(f"Unknown node: {node_name}")
        return self.node_classes[node_name]

    def _get_node_instance(self, node_name: str) -> Any:
        if node_name not in self.node_instances:
            if node_name not in self.node_classes:
                raise KeyError(f"Unknown node: {node_name}")
            self.node_instances[node_name] = self.node_classes[node_name]()
        return self.node_instances[node_name]

    async def before_module_loaded(self) -> None:
        await super().before_module_loaded()
        try:
            from comfy_api.latest import ComfyAPI_latest
            from .proxies.progress_proxy import ProgressProxy
            from .proxies.folder_paths_proxy import FolderPathsProxy
            import comfy_api.latest._ui as latest_ui
            import comfy_api.latest._resources as latest_resources

            ComfyAPI_latest.Execution = ProgressProxy
            ComfyAPI_latest.execution = ProgressProxy()
            fp_proxy = FolderPathsProxy()
            latest_ui.folder_paths = fp_proxy
            latest_resources.folder_paths = fp_proxy
        except Exception:
            pass

    async def call_route_handler(
        self,
        handler_module: str,
        handler_func: str,
        request_data: Dict[str, Any],
    ) -> Any:
        cache_key = f"{handler_module}.{handler_func}"
        if cache_key not in self._route_handlers:
            if self._module is not None and hasattr(self._module, '__file__'):
                node_dir = os.path.dirname(self._module.__file__)
                if node_dir not in sys.path:
                    sys.path.insert(0, node_dir)
            try:
                module = importlib.import_module(handler_module)
                self._route_handlers[cache_key] = getattr(module, handler_func)
            except (ImportError, AttributeError) as e:
                raise ValueError(f"Route handler not found: {cache_key}") from e

        handler = self._route_handlers[cache_key]
        mock_request = MockRequest(request_data)

        if asyncio.iscoroutinefunction(handler):
            result = await handler(mock_request)
        else:
            result = handler(mock_request)
        return self._serialize_response(result)

    def _serialize_response(self, response: Any) -> Dict[str, Any]:
        if response is None:
            return {"type": "text", "body": "", "status": 204}
        if isinstance(response, dict):
            return {"type": "json", "body": response, "status": 200}
        if isinstance(response, str):
            return {"type": "text", "body": response, "status": 200}
        if hasattr(response, 'text') and hasattr(response, 'status'):
            return {
                "type": "text",
                "body": response.text if hasattr(response, 'text') else str(response.body),
                "status": response.status,
                "headers": dict(response.headers) if hasattr(response, 'headers') else {},
            }
        if hasattr(response, 'body') and hasattr(response, 'status'):
            body = response.body
            if isinstance(body, bytes):
                try:
                    return {"type": "text", "body": body.decode('utf-8'), "status": response.status}
                except UnicodeDecodeError:
                    return {"type": "binary", "body": body.hex(), "status": response.status}
            return {"type": "json", "body": body, "status": response.status}
        return {"type": "text", "body": str(response), "status": 200}


class MockRequest:
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
        if isinstance(self._body, dict):
            return self._body
        if isinstance(self._body, str):
            return json.loads(self._body)
        return {}

    async def post(self) -> Dict[str, Any]:
        if isinstance(self._body, dict):
            return self._body
        return {}

    async def text(self) -> str:
        if self._text:
            return self._text
        if isinstance(self._body, str):
            return self._body
        if isinstance(self._body, dict):
            return json.dumps(self._body)
        return ""

    async def read(self) -> bytes:
        return (await self.text()).encode('utf-8')
