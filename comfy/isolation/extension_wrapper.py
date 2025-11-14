"""PyIsolate Extension wrapper for isolated ComfyUI nodes."""

from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

from pyisolate import ExtensionBase

LOG_PREFIX = "🔒 [PyIsolate]"

logger = logging.getLogger(__name__)


class ComfyNodeExtension(ExtensionBase):
    """Surface NODE_CLASS_MAPPINGS from isolated modules over RPC."""

    def __init__(self) -> None:
        super().__init__()
        self.node_classes: Dict[str, type] = {}
        self.display_names: Dict[str, str] = {}
        self.node_instances: Dict[str, Any] = {}

    async def on_module_loaded(self, module: Any) -> None:
        """Cache node metadata when the isolated module is imported."""
        self.node_classes = getattr(module, "NODE_CLASS_MAPPINGS", {}) or {}
        self.display_names = getattr(module, "NODE_DISPLAY_NAME_MAPPINGS", {}) or {}

        # Instantiate each node class once so execute_node can call methods directly.
        self.node_instances = {}
        for name, node_cls in self.node_classes.items():
            self.node_instances[name] = node_cls()
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
        """Return metadata required by ComfyUI for a given node."""
        node_cls = self._get_node_class(node_name)
        return {
            "input_types": node_cls.INPUT_TYPES() if hasattr(node_cls, "INPUT_TYPES") else {},
            "return_types": getattr(node_cls, "RETURN_TYPES", ()),
            "return_names": getattr(node_cls, "RETURN_NAMES", None),
            "function": getattr(node_cls, "FUNCTION", "execute"),
            "category": getattr(node_cls, "CATEGORY", ""),
            "output_node": getattr(node_cls, "OUTPUT_NODE", False),
            "display_name": self.display_names.get(node_name, node_name),
        }

    async def execute_node(self, node_name: str, **inputs: Any) -> Tuple[Any, ...]:
        """Invoke the node's FUNCTION with provided inputs."""
        instance = self._get_node_instance(node_name)
        node_cls = self._get_node_class(node_name)
        function_name = getattr(node_cls, "FUNCTION", "execute")
        if not hasattr(instance, function_name):
            raise AttributeError(f"Node {node_name} missing callable '{function_name}'")
        method = getattr(instance, function_name)
        result = method(**inputs)
        if not isinstance(result, tuple):
            result = (result,)
        return result

    def _get_node_class(self, node_name: str) -> type:
        if node_name not in self.node_classes:
            raise KeyError(f"Unknown node: {node_name}")
        return self.node_classes[node_name]

    def _get_node_instance(self, node_name: str) -> Any:
        if node_name not in self.node_instances:
            raise KeyError(f"Unknown node instance: {node_name}")
        return self.node_instances[node_name]
