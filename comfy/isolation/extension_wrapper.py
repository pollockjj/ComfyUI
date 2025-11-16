"""PyIsolate Extension wrapper for isolated ComfyUI nodes."""

from __future__ import annotations

import logging
import pickle
import uuid
from typing import Any, Dict, Tuple

from pyisolate import ExtensionBase

LOG_PREFIX = "📚 [PyIsolate]"

logger = logging.getLogger(__name__)


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

    async def on_module_loaded(self, module: Any) -> None:
        """Cache node metadata when the isolated module is imported."""
        self.node_classes = getattr(module, "NODE_CLASS_MAPPINGS", {}) or {}
        self.display_names = getattr(module, "NODE_DISPLAY_NAME_MAPPINGS", {}) or {}

        # Fix __module__ for all node classes to avoid pickle issues
        # When PyIsolate loads modules, __module__ can be set to the file path
        # instead of a proper module name, breaking pickle
        module_name = getattr(module, '__name__', 'isolated_nodes')
        for node_name, node_cls in self.node_classes.items():
            if hasattr(node_cls, '__module__') and '/' in str(node_cls.__module__):
                # Path detected in __module__, fix it
                node_cls.__module__ = module_name
                logger.debug(
                    "%s[ExtensionWrapper] Fixed __module__ for %s",
                    LOG_PREFIX,
                    node_name,
                )

        # Don't instantiate yet - wait until execution time
        # This avoids pickle issues with classes that have bad __module__ attributes
        self.node_instances = {}
        
        logger.info(
            "%s[ExtensionWrapper] Loaded %d nodes: %s",
            LOG_PREFIX,
            len(self.node_classes),
            list(self.node_classes.keys()),
        )

    async def list_nodes(self) -> Dict[str, str]:
        """Return mapping of node names to display names."""
        return {
            name: self.display_names.get(name, name)
            for name in self.node_classes
        }

    async def get_node_info(self, node_name: str) -> Dict[str, Any]:
        """Return metadata required by ComfyUI for a named node."""
        # This is called during loading - must return only JSON-serializable primitives
        # All complex objects must be converted to strings/primitives HERE in the isolated process
        return {
            "input_types": {},  # Will be populated by get_node_details
            "return_types": (),
            "return_names": None,
            "function": "execute",
            "category": "",
            "output_node": False,
            "display_name": self.display_names.get(node_name, node_name),
        }
    
    async def get_node_details(self, node_name: str) -> Dict[str, Any]:
        """Get full node details - called during loading, must return JSON-serializable data."""
        node_cls = self._get_node_class(node_name)
        
        # Call INPUT_TYPES() - this WILL cause imports but we're in the isolated process where imports work
        input_types_raw = node_cls.INPUT_TYPES() if hasattr(node_cls, "INPUT_TYPES") else {}
        
        # Force serialization to JSON and back to ensure NO object references remain
        import json
        
        # Convert to JSON string (this will call str() on any non-serializable objects)
        input_types_json_str = json.dumps(input_types_raw, default=str)
        # Parse back to get fresh Python objects with no references to original
        input_types_safe = json.loads(input_types_json_str)
        
        logger.info("%s[ExtensionWrapper] INPUT_TYPES for %s converted successfully", LOG_PREFIX, node_name)
        
        return {
            "input_types": input_types_safe,
            "return_types": tuple(str(t) for t in getattr(node_cls, "RETURN_TYPES", ())),
            "return_names": getattr(node_cls, "RETURN_NAMES", None),
            "function": str(getattr(node_cls, "FUNCTION", "execute")),
            "category": str(getattr(node_cls, "CATEGORY", "")),
            "output_node": bool(getattr(node_cls, "OUTPUT_NODE", False)),
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
        
        # Primitives: pass through
        if isinstance(data, (str, int, float, bool, type(None))):
            return data
        
        # Torch tensors: pass through (handled by share_torch)
        if isinstance(data, torch.Tensor):
            return data
        
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
        logger.info(
            "%s[RemoteObject] Stored %s as remote object %s",
            LOG_PREFIX,
            type_name,
            object_id[:8],
        )
        return RemoteObjectHandle(object_id, type_name)

    def _resolve_remote_objects(self, data: Any) -> Any:
        """Recursively replace RemoteObjectHandles with actual objects."""
        if isinstance(data, RemoteObjectHandle):
            if data.object_id not in self.remote_objects:
                raise KeyError(f"Remote object {data.object_id} not found in cache")
            return self.remote_objects[data.object_id]
        
        if isinstance(data, dict):
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
