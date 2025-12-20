"""ProxiedSingleton for nodes module.

Rev 1.0 Implementation - Read-only access to node registry and constants.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Any

try:
    from pyisolate import ProxiedSingleton
except ImportError:
    # Fallback when pyisolate not installed
    class ProxiedSingleton:
        pass

LOG_PREFIX = "]["
logger = logging.getLogger(__name__)


class NodesProxy(ProxiedSingleton):
    """Read-only proxy for nodes module constants and registry.
    
    Provides access to NODE_CLASS_MAPPINGS (snapshot) and MAX_RESOLUTION
    for isolated nodes that need to reference other node classes or constants.
    
    Note: NODE_CLASS_MAPPINGS is snapshotted at load time. Mutations in 
    child process do NOT affect the host.
    
    Rev 1.0: Per PYISOLATE_COMFY_INTEGRATION_ARCHITECTURE.md
    """
    
    _instance: Optional['NodesProxy'] = None
    
    def __new__(cls) -> 'NodesProxy':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._max_resolution = None
            cls._instance._node_class_mappings = None
            cls._instance._node_display_name_mappings = None
        return cls._instance
    
    # =========================================================================
    # Constants
    # =========================================================================
    
    @property
    def MAX_RESOLUTION(self) -> int:
        """Get maximum resolution constant (8192 by default)."""
        if self._max_resolution is None:
            import nodes
            self._max_resolution = nodes.MAX_RESOLUTION
            logger.debug(f"{LOG_PREFIX}[NodesProxy] MAX_RESOLUTION → {self._max_resolution}")
        return self._max_resolution
    
    # =========================================================================
    # Node Registry (Read-Only Snapshots)
    # =========================================================================
    
    @property
    def NODE_CLASS_MAPPINGS(self) -> Dict[str, type]:
        """Get snapshot of node class mappings.
        
        Returns a COPY of the mappings - mutations don't affect host.
        This is intentional to prevent isolated nodes from registering
        classes in the host process.
        """
        if self._node_class_mappings is None:
            import nodes
            # Shallow copy - class references preserved but dict is new
            self._node_class_mappings = dict(nodes.NODE_CLASS_MAPPINGS)
            logger.debug(f"{LOG_PREFIX}[NodesProxy] NODE_CLASS_MAPPINGS snapshot → {len(self._node_class_mappings)} nodes")
        return self._node_class_mappings
    
    @property
    def NODE_DISPLAY_NAME_MAPPINGS(self) -> Dict[str, str]:
        """Get snapshot of node display name mappings."""
        if self._node_display_name_mappings is None:
            import nodes
            self._node_display_name_mappings = dict(nodes.NODE_DISPLAY_NAME_MAPPINGS)
            logger.debug(f"{LOG_PREFIX}[NodesProxy] NODE_DISPLAY_NAME_MAPPINGS snapshot → {len(self._node_display_name_mappings)} names")
        return self._node_display_name_mappings
    
    # =========================================================================
    # Base Classes for Inheritance
    # =========================================================================
    
    @property
    def PreviewImage(self) -> type:
        """Get PreviewImage base class for node inheritance."""
        import nodes
        return nodes.PreviewImage
    
    @property
    def SaveImage(self) -> type:
        """Get SaveImage base class for node inheritance."""
        import nodes
        return nodes.SaveImage
    
    @property
    def LoadImage(self) -> type:
        """Get LoadImage base class for node inheritance."""
        import nodes
        return nodes.LoadImage
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def get_node_class(self, node_name: str) -> Optional[type]:
        """Get a specific node class by name.
        
        Safe wrapper that returns None instead of raising KeyError.
        """
        import nodes
        result = nodes.NODE_CLASS_MAPPINGS.get(node_name)
        logger.debug(f"{LOG_PREFIX}[NodesProxy] get_node_class({node_name}) → {'found' if result else 'not found'}")
        return result
    
    def get_node_display_name(self, node_name: str) -> str:
        """Get display name for a node."""
        import nodes
        result = nodes.NODE_DISPLAY_NAME_MAPPINGS.get(node_name, node_name)
        logger.debug(f"{LOG_PREFIX}[NodesProxy] get_node_display_name({node_name}) → {result}")
        return result

