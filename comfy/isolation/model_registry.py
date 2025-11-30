"""
comfy.isolation.model_registry
Host-side registry for ModelPatcher instances with scoped lifetime management.
"""

import uuid
import threading
import logging
from typing import Dict, Optional, Any

logger = logging.getLogger(__name__)


class ScopedModelRegistry:
    """Manages active ModelPatcher instances during a single RPC call scope.
    
    Design Principles:
    1. Scoped Lifetime: Registry lives only during isolated node execution
    2. Automatic Cleanup: Python GC handles cleanup when scope exits
    3. Thread-Safe: Multiple concurrent workflows supported
    4. Identity Preservation: Same Python object → same model_id
    
    Usage:
        registry = ScopedModelRegistry()
        model_id = registry.register(model_patcher)
        # ... RPC calls ...
        # Registry auto-cleans when scope exits
    """
    
    def __init__(self):
        self._active_patchers: Dict[str, Any] = {}  # model_id → ModelPatcher
        self._id_map: Dict[int, str] = {}  # id(patcher) → model_id (reverse lookup)
        self._lock = threading.Lock()
        logger.debug("📚 [ModelRegistry] Created new scoped registry")
    
    def register(self, patcher: Any) -> str:
        """Register a ModelPatcher and return unique ID.
        
        If this patcher was already registered (same Python object),
        returns the existing model_id. This preserves identity semantics
        required by ModelPatcher.is_clone().
        
        Args:
            patcher: ModelPatcher instance from host
            
        Returns:
            model_id: UUID string for cross-process reference
        """
        with self._lock:
            obj_id = id(patcher)
            
            # Check if already registered (identity preservation)
            if obj_id in self._id_map:
                existing_id = self._id_map[obj_id]
                logger.debug(f"📚 [ModelRegistry] Re-using model_id {existing_id} for object {obj_id}")
                return existing_id
            
            # New registration
            model_id = str(uuid.uuid4())
            self._active_patchers[model_id] = patcher
            self._id_map[obj_id] = model_id
            
            # Safe size check
            try:
                size = patcher.model_size()
                logger.info(f"📚 [ModelRegistry] Registered ModelPatcher as {model_id} (size: {size} bytes)")
            except Exception:
                logger.info(f"📚 [ModelRegistry] Registered ModelPatcher as {model_id}")
            
            return model_id
    
    def get(self, model_id: str) -> Optional[Any]:
        """Retrieve ModelPatcher by ID.
        
        Returns None if model_id not found (scope expired or invalid).
        """
        with self._lock:
            patcher = self._active_patchers.get(model_id)
            if patcher is None:
                logger.warning(f"📚 [ModelRegistry] model_id {model_id} not found (expired or invalid)")
            return patcher
    
    def unregister(self, model_id: str) -> None:
        """Explicitly remove a ModelPatcher from registry.
        
        Normally not needed (scoped cleanup handles this), but provided
        for manual cleanup if required.
        """
        with self._lock:
            if model_id in self._active_patchers:
                patcher = self._active_patchers[model_id]
                obj_id = id(patcher)
                
                del self._active_patchers[model_id]
                if obj_id in self._id_map:
                    del self._id_map[obj_id]
                
                logger.debug(f"📚 [ModelRegistry] Unregistered model_id {model_id}")
    
    def count(self) -> int:
        """Return number of registered patchers (for debugging)."""
        with self._lock:
            return len(self._active_patchers)
    
    def __del__(self):
        """Cleanup logging when registry is garbage collected."""
        try:
            count = self.count()
            logger.debug(f"📚 [ModelRegistry] Registry destroyed, {count} patchers released")
        except Exception:
            pass  # Suppress errors during cleanup


# Thread-local storage for current registry (set during execution)
# Using threading.local instead of ContextVar for better compatibility with RPC threading
import threading
_thread_local = threading.local()


def get_current_registry() -> ScopedModelRegistry:
    """Get the registry for the current execution scope.
    
    Raises:
        RuntimeError: If called outside of execution scope
    """
    registry = getattr(_thread_local, 'current_registry', None)
    if registry is None:
        raise RuntimeError("No ScopedModelRegistry in current context. Must be called within execution scope.")
    return registry


def set_current_registry(registry: Optional[ScopedModelRegistry]) -> None:
    """Set the registry for current execution scope (internal use)."""
    _thread_local.current_registry = registry


def get_current_registry_if_exists() -> Optional[ScopedModelRegistry]:
    """Get registry if it exists, None otherwise (for optional use)."""
    return getattr(_thread_local, 'current_registry', None)
