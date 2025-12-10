"""
Stateless RPC pattern for ModelPatcher instances.

This module provides:
1. ModelPatcherRegistry - Host-side registry of ModelPatcher instances (ProxiedSingleton)
2. ModelPatcherProxy - Picklable handle that forwards calls via RPC
3. maybe_wrap_model_for_isolation - Integration hook for checkpoint loading

Architecture mirrors clip_proxy.py exactly.

Phase 1 Scope (PuLID Core):
- clone() → Deep Remote Copy
- get_model_object() → Return proxied nested objects
- model_options property getter/setter
- load_device / offload_device properties
- size property

Research References:
- Shadow-Reference Protocol (MODELPATCHER_PHASE1_PLAN.md)
- CLIPProxy implementation (proven pattern)
"""

import asyncio
import logging
import os
import threading
import time
import weakref
from typing import Any, Dict, Optional

try:
    from pyisolate import ProxiedSingleton
except ImportError:
    # Graceful degradation if pyisolate not available
    class ProxiedSingleton:
        """Fallback when pyisolate not installed."""
        pass

logger = logging.getLogger(__name__)

# Host/child detection
IS_CHILD_PROCESS = os.environ.get("PYISOLATE_CHILD") == "1"


def _timing_decorator(func):
    """
    Decorator to log RPC method timing.
    
    Debug Checkpoint: Logs entry/exit with duration for every RPC call.
    Target: <5ms average per call.
    """
    async def wrapper(*args, **kwargs):
        method_name = func.__name__
        instance_id = args[1] if len(args) > 1 else "unknown"
        start = time.perf_counter()
        try:
            result = await func(*args, **kwargs)
            duration_ms = (time.perf_counter() - start) * 1000
            logger.debug(
                f"][[ModelPatcherRegistry] "
                f"{method_name}({instance_id}) completed in {duration_ms:.2f}ms"
            )
            return result
        except Exception as e:
            duration_ms = (time.perf_counter() - start) * 1000
            logger.error(
                f"][[ModelPatcherRegistry] "
                f"{method_name}({instance_id}) FAILED in {duration_ms:.2f}ms: {e}"
            )
            raise
    return wrapper


class ModelPatcherRegistry(ProxiedSingleton):
    """
    Host-side registry of ModelPatcher instances using ProxiedSingleton pattern.
    
    Thread-safe singleton that manages ModelPatcher object lifecycle and provides
    async RPC methods for isolated child processes.
    
    CRITICAL: Inherits from ProxiedSingleton to enable RPC from child processes.
    
    Research References:
    - Shadow-Reference Protocol Section 5.2 (Global Model Registry)
    - CLIPProxy implementation (production proof)
    
    Design Principles:
    - Identity preservation via _id_map (Finding 3: is_clone logic)
    - Lazy patch application (Finding 2: metadata operations only)
    - Zero CUDA tensor serialization (bandwidth reduction ~14,000x)
    """
    
    def __init__(self) -> None:
        """Initialize registry state (called once by ProxiedSingleton)."""
        if hasattr(ProxiedSingleton, "__init__") and ProxiedSingleton is not object:
            super().__init__()
        
        self._registry: Dict[str, Any] = {}
        self._id_map: Dict[int, str] = {}  # id(model_patcher) → instance_id
        self._counter = 0
        self._lock = threading.Lock()
        
        # Debug stats
        self._register_count = 0
        self._id_map_hits = 0
        
        logger.debug("][[ModelPatcherRegistry] Initialized")
    
    def register(self, model_patcher) -> str:
        """
        Register a ModelPatcher instance and return unique ID.
        
        If the same Python object (by id()) was already registered,
        returns the existing ID to preserve identity semantics.
        
        Research: Shadow-Reference Protocol Finding 1 (Model Reference Stability)
        
        Debug Checkpoint: Logs register with id() and instance_id.
        Logs identity preservation hits (same id() returns cached instance_id).
        
        Args:
            model_patcher: ModelPatcher object to register
            
        Returns:
            Unique instance ID (e.g., "model_0")
            
        Raises:
            RuntimeError: If called from child process
        """
        if IS_CHILD_PROCESS:
            raise RuntimeError(
                "][[ModelPatcherRegistry] FAIL-LOUD: "
                "Cannot register ModelPatcher in child process"
            )
        
        with self._lock:
            # Check if already registered (identity preservation - Finding 3)
            obj_id = id(model_patcher)
            if obj_id in self._id_map:
                existing_id = self._id_map[obj_id]
                self._id_map_hits += 1
                logger.debug(
                    f"][[ModelPatcherRegistry] Identity hit: "
                    f"Re-using {existing_id} for object {obj_id} "
                    f"(total hits: {self._id_map_hits})"
                )
                return existing_id
            
            # New registration
            instance_id = f"model_{self._counter}"
            self._counter += 1
            self._registry[instance_id] = model_patcher
            self._id_map[obj_id] = instance_id
            self._register_count += 1
            
            logger.debug(
                f"][[ModelPatcherRegistry] Registered {instance_id} "
                f"(id={obj_id}, total registered: {self._register_count})"
            )
        
        return instance_id
    
    def unregister_sync(self, instance_id: str) -> None:
        """
        Unregister a ModelPatcher instance (called by weakref.finalize).
        
        This is a synchronous method designed to be called from finalizers.
        Must not raise exceptions.
        
        Debug Checkpoint: Logs unregister confirmation.
        
        Args:
            instance_id: ID to unregister
        """
        try:
            with self._lock:
                if instance_id in self._registry:
                    # Also clean up _id_map
                    model_patcher = self._registry[instance_id]
                    obj_id = id(model_patcher)
                    if obj_id in self._id_map:
                        del self._id_map[obj_id]
                    del self._registry[instance_id]
                    logger.debug(
                        f"][[ModelPatcherRegistry] Unregistered {instance_id}"
                    )
        except Exception as e:
            logger.error(
                f"][[ModelPatcherRegistry] Unregister failed for {instance_id}: {e}"
            )
    
    def _get_instance(self, instance_id: str):
        """
        Internal: Get ModelPatcher instance by ID.
        
        Args:
            instance_id: ID to lookup
            
        Returns:
            ModelPatcher instance
            
        Raises:
            ValueError: If instance_id not found
        """
        instance = self._registry.get(instance_id)
        if instance is None:
            raise ValueError(
                f"][[ModelPatcherRegistry] FAIL-LOUD: "
                f"Instance {instance_id} not found in registry "
                f"(registry size: {len(self._registry)})"
            )
        return instance
    
    def get_stats(self) -> dict:
        """
        Get registry statistics for debugging.
        
        Returns:
            Dict with register_count, id_map_hits, current_size
        """
        with self._lock:
            return {
                "register_count": self._register_count,
                "id_map_hits": self._id_map_hits,
                "current_size": len(self._registry),
            }
    
    # ============================================================
    # Phase 1 RPC Methods (7 core operations for PuLID)
    # ============================================================
    
    @_timing_decorator
    async def clone(self, instance_id: str) -> str:
        """
        RPC: Clone ModelPatcher instance (Deep Remote Copy pattern).
        
        Creates a new ModelPatcher via clone(), registers it,
        and returns the new ID.
        
        Research: Shadow-Reference Protocol Section 5.3 validates this.
        
        Args:
            instance_id: Source ModelPatcher ID
            
        Returns:
            New instance ID for the clone
        """
        instance = self._get_instance(instance_id)
        new_model = instance.clone()
        new_id = self.register(new_model)
        logger.debug(
            f"][[ModelPatcherRegistry] Cloned {instance_id} → {new_id}"
        )
        return new_id
    
    @_timing_decorator
    async def get_model_object(self, instance_id: str, name: str) -> Any:
        """
        RPC: Get a named object from the model.
        
        Used by PuLID for: model.get_model_object("model_sampling")
        
        Args:
            instance_id: ModelPatcher ID
            name: Object name (e.g., "model_sampling")
            
        Returns:
            The requested object (may need wrapping for nested access)
        """
        instance = self._get_instance(instance_id)
        result = instance.get_model_object(name)
        # Special-case model_sampling to ensure it goes through ModelSamplingRegistry
        if name == "model_sampling":
            from comfy.isolation.model_sampling_proxy import ModelSamplingRegistry, ModelSamplingProxy
            registry = ModelSamplingRegistry()
            sampling_id = registry.register(result)
            proxy = ModelSamplingProxy(sampling_id, registry)
            logger.debug(
                f"][[ModelPatcherRegistry] get_model_object({instance_id}, '{name}') "
                f"returned ModelSamplingProxy({sampling_id})"
            )
            return proxy

        logger.debug(
            f"][[ModelPatcherRegistry] get_model_object({instance_id}, '{name}') "
            f"returned type: {type(result).__name__}"
        )
        return result
    
    @_timing_decorator
    async def get_model_options(self, instance_id: str) -> dict:
        """
        RPC: Get model_options dict.
        
        Research: Shadow-Reference Protocol Finding 2 (Lazy Patch Application)
        This is CPU-bound metadata, not VRAM weights.
        
        Args:
            instance_id: ModelPatcher ID
            
        Returns:
            model_options dict (copy to prevent mutation issues)
        """
        instance = self._get_instance(instance_id)
        # Return a copy to prevent unintended mutation
        import copy
        options = copy.deepcopy(instance.model_options)
        logger.debug(
            f"][[ModelPatcherRegistry] get_model_options({instance_id}) "
            f"keys: {list(options.keys())}"
        )
        return options
    
    @_timing_decorator
    async def set_model_options(self, instance_id: str, options: dict) -> None:
        """
        RPC: Set model_options dict.
        
        Debug Checkpoint: Logs before/after keys for state preservation validation.
        
        Args:
            instance_id: ModelPatcher ID
            options: New model_options dict
        """
        instance = self._get_instance(instance_id)
        old_keys = set(instance.model_options.keys())
        instance.model_options = options
        new_keys = set(options.keys())
        logger.debug(
            f"][[ModelPatcherRegistry] set_model_options({instance_id}) "
            f"old_keys: {old_keys}, new_keys: {new_keys}"
        )
    
    @_timing_decorator
    async def get_load_device(self, instance_id: str) -> Any:
        """
        RPC: Get load_device property.
        
        Args:
            instance_id: ModelPatcher ID
            
        Returns:
            torch.device for model loading
        """
        instance = self._get_instance(instance_id)
        device = instance.load_device
        logger.debug(
            f"][[ModelPatcherRegistry] get_load_device({instance_id}) = {device}"
        )
        return device
    
    @_timing_decorator
    async def get_offload_device(self, instance_id: str) -> Any:
        """
        RPC: Get offload_device property.
        
        Args:
            instance_id: ModelPatcher ID
            
        Returns:
            torch.device for model offloading
        """
        instance = self._get_instance(instance_id)
        device = instance.offload_device
        logger.debug(
            f"][[ModelPatcherRegistry] get_offload_device({instance_id}) = {device}"
        )
        return device

    @_timing_decorator
    async def get_hook_mode(self, instance_id: str) -> Any:
        """RPC: Get hook_mode property (used by samplers)."""
        instance = self._get_instance(instance_id)
        return getattr(instance, "hook_mode", None)

    @_timing_decorator
    async def set_hook_mode(self, instance_id: str, value: Any) -> None:
        """RPC: Set hook_mode property (used by samplers)."""
        instance = self._get_instance(instance_id)
        setattr(instance, "hook_mode", value)

    @_timing_decorator
    async def model_dtype(self, instance_id: str) -> Any:
        """RPC: Get model dtype (used by sampler helpers)."""
        instance = self._get_instance(instance_id)
        return instance.model_dtype()

    @_timing_decorator
    async def pre_run(self, instance_id: str) -> None:
        """RPC: Forward pre_run lifecycle hook."""
        instance = self._get_instance(instance_id)
        instance.pre_run()

    @_timing_decorator
    async def cleanup(self, instance_id: str) -> None:
        """RPC: Forward cleanup lifecycle hook."""
        instance = self._get_instance(instance_id)
        instance.cleanup()

    @_timing_decorator
    async def restore_hook_patches(self, instance_id: str) -> None:
        """RPC: Forward restore_hook_patches lifecycle hook."""
        instance = self._get_instance(instance_id)
        instance.restore_hook_patches()

    @_timing_decorator
    async def register_all_hook_patches(self, instance_id: str, hooks: Any, target_dict: Any, model_options: Any, registered: Any) -> None:
        """RPC: Forward register_all_hook_patches used by sampler helpers."""
        instance = self._get_instance(instance_id)
        instance.register_all_hook_patches(hooks, target_dict, model_options, registered)

    @_timing_decorator
    async def get_nested_additional_models(self, instance_id: str) -> Any:
        """RPC: Return nested additional models list."""
        instance = self._get_instance(instance_id)
        return instance.get_nested_additional_models()

    @_timing_decorator
    async def model_patches_models(self, instance_id: str) -> Any:
        """RPC: Return list of patched models for loading decisions."""
        instance = self._get_instance(instance_id)
        return instance.model_patches_models()

    @_timing_decorator
    async def get_parent(self, instance_id: str) -> Any:
        """RPC: Get parent attribute of model patcher (may be None)."""
        instance = self._get_instance(instance_id)
        return getattr(instance, "parent", None)

    @_timing_decorator
    async def current_loaded_device(self, instance_id: str) -> Any:
        """RPC: Get current_loaded_device() from model patcher."""
        instance = self._get_instance(instance_id)
        return instance.current_loaded_device()

    @_timing_decorator
    async def model_size(self, instance_id: str) -> Any:
        """RPC: Get model_size() from underlying model patcher."""
        instance = self._get_instance(instance_id)
        return instance.model_size()

    @_timing_decorator
    async def loaded_size(self, instance_id: str) -> Any:
        """RPC: Get loaded_size() from underlying model patcher."""
        instance = self._get_instance(instance_id)
        return instance.loaded_size()

    @_timing_decorator
    async def model_patches_to(self, instance_id: str, device: Any) -> Any:
        """RPC: Move model patches to target device."""
        instance = self._get_instance(instance_id)
        return instance.model_patches_to(device)

    @_timing_decorator
    async def partially_load(self, instance_id: str, device: Any, extra_memory: Any, force_patch_weights: bool = False) -> Any:
        """RPC: Partially load model to device with optional extra memory usage."""
        instance = self._get_instance(instance_id)
        return instance.partially_load(device, extra_memory, force_patch_weights=force_patch_weights)

    @_timing_decorator
    async def process_latent_in(self, instance_id: str, args: tuple, kwargs: dict) -> Any:
        """RPC: Call model.process_latent_in(*args, **kwargs)."""
        instance = self._get_instance(instance_id)
        return instance.model.process_latent_in(*args, **kwargs)

    @_timing_decorator
    async def process_latent_out(self, instance_id: str, args: tuple, kwargs: dict) -> Any:
        """RPC: Call model.process_latent_out(*args, **kwargs)."""
        instance = self._get_instance(instance_id)
        result = instance.model.process_latent_out(*args, **kwargs)
        try:
            target = None
            if args and hasattr(args[0], "device"):
                target = args[0].device
            elif kwargs:
                for v in kwargs.values():
                    if hasattr(v, "device"):
                        target = v.device
                        break
            if target is not None and hasattr(result, "to"):
                return result.to(target)
        except Exception:
            pass
        return result

    @_timing_decorator
    async def scale_latent_inpaint(self, instance_id: str, args: tuple, kwargs: dict) -> Any:
        """RPC: Call model.scale_latent_inpaint(*args, **kwargs)."""
        instance = self._get_instance(instance_id)
        result = instance.model.scale_latent_inpaint(*args, **kwargs)
        try:
            target = None
            if args and hasattr(args[0], "device"):
                target = args[0].device
            elif kwargs:
                for v in kwargs.values():
                    if hasattr(v, "device"):
                        target = v.device
                        break
            if target is not None and hasattr(result, "to"):
                return result.to(target)
        except Exception:
            pass
        return result

    @_timing_decorator
    async def get_model_sampling(self, instance_id: str) -> Any:
        """RPC: Return model_sampling via proxy registration."""
        instance = self._get_instance(instance_id)
        ms_obj = instance.model.model_sampling
        from comfy.isolation.model_sampling_proxy import ModelSamplingRegistry, ModelSamplingProxy
        ms_id = ModelSamplingRegistry().register(ms_obj)
        return ModelSamplingProxy(ms_id)

    @_timing_decorator
    async def prepare_state(self, instance_id: str, timestep: Any) -> Any:
        """RPC: Forward prepare_state on current_patcher (scheduler state)."""
        instance = self._get_instance(instance_id)
        cp = getattr(instance.model, "current_patcher", instance)
        return cp.prepare_state(timestep)

    @_timing_decorator
    async def apply_hooks(self, instance_id: str, hooks: Any) -> Any:
        """RPC: Forward apply_hooks on current_patcher."""
        instance = self._get_instance(instance_id)
        cp = getattr(instance.model, "current_patcher", instance)
        return cp.apply_hooks(hooks=hooks)
    
    @_timing_decorator
    async def get_size(self, instance_id: str) -> int:
        """
        RPC: Get model size in bytes.
        
        Research: Shadow-Reference Protocol Section 6.4 (VRAM Reporting)
        Child should see ~0 VRAM since it doesn't hold the model.
        
        Args:
            instance_id: ModelPatcher ID
            
        Returns:
            Size in bytes
        """
        instance = self._get_instance(instance_id)
        size = instance.size
        logger.debug(
            f"][[ModelPatcherRegistry] get_size({instance_id}) = {size} bytes"
        )
        return size

    @_timing_decorator
    async def get_wrappers(self, instance_id: str) -> Any:
        """RPC: Get wrappers mapping (used by sampler helpers)."""
        instance = self._get_instance(instance_id)
        # Wrappers can contain local classes that are not pickleable; return empty
        return {}

    @_timing_decorator
    async def get_callbacks(self, instance_id: str) -> Any:
        """RPC: Get callbacks mapping (used by sampler helpers)."""
        instance = self._get_instance(instance_id)
        # Callbacks may capture local objects; avoid sending across the wire
        return {}
    
    # ============================================================
    # Sync versions for host-side direct calls (avoid async loop issues)
    # ============================================================
    
    def clone_sync(self, instance_id: str) -> str:
        """Sync version of clone() for host-side calls."""
        instance = self._get_instance(instance_id)
        new_model = instance.clone()
        new_id = self.register(new_model)
        logger.debug(
            f"][[ModelPatcherRegistry] Cloned {instance_id} → {new_id} (sync)"
        )
        return new_id
    
    def get_model_object_sync(self, instance_id: str, name: str) -> Any:
        """Sync version of get_model_object() for host-side calls."""
        instance = self._get_instance(instance_id)
        result = instance.get_model_object(name)
        if name == "model_sampling":
            from comfy.isolation.model_sampling_proxy import ModelSamplingRegistry, ModelSamplingProxy
            registry = ModelSamplingRegistry()
            sampling_id = registry.register(result)
            return ModelSamplingProxy(sampling_id, registry)
        return result
    
    def get_model_options_sync(self, instance_id: str) -> dict:
        """Sync version of get_model_options() for host-side calls."""
        instance = self._get_instance(instance_id)
        import copy
        return copy.deepcopy(instance.model_options)
    
    def set_model_options_sync(self, instance_id: str, options: dict) -> None:
        """Sync version of set_model_options() for host-side calls."""
        instance = self._get_instance(instance_id)
        instance.model_options = options
    
    def get_load_device_sync(self, instance_id: str) -> Any:
        """Sync version of get_load_device() for host-side calls."""
        instance = self._get_instance(instance_id)
        return instance.load_device
    
    def get_offload_device_sync(self, instance_id: str) -> Any:
        """Sync version of get_offload_device() for host-side calls."""
        instance = self._get_instance(instance_id)
        return instance.offload_device
    
    def get_size_sync(self, instance_id: str) -> int:
        """Sync version of get_size() for host-side calls."""
        instance = self._get_instance(instance_id)
        return instance.size

    def get_wrappers_sync(self, instance_id: str) -> Any:
        return {}

    def get_callbacks_sync(self, instance_id: str) -> Any:
        return {}

    def get_hook_mode_sync(self, instance_id: str) -> Any:
        instance = self._get_instance(instance_id)
        return getattr(instance, "hook_mode", None)

    def set_hook_mode_sync(self, instance_id: str, value: Any) -> None:
        instance = self._get_instance(instance_id)
        setattr(instance, "hook_mode", value)

    def model_dtype_sync(self, instance_id: str) -> Any:
        instance = self._get_instance(instance_id)
        return instance.model_dtype()

    def pre_run_sync(self, instance_id: str) -> None:
        instance = self._get_instance(instance_id)
        instance.pre_run()

    def cleanup_sync(self, instance_id: str) -> None:
        instance = self._get_instance(instance_id)
        instance.cleanup()

    def restore_hook_patches_sync(self, instance_id: str) -> None:
        instance = self._get_instance(instance_id)
        instance.restore_hook_patches()

    def register_all_hook_patches_sync(self, instance_id: str, hooks: Any, target_dict: Any, model_options: Any, registered: Any) -> None:
        instance = self._get_instance(instance_id)
        instance.register_all_hook_patches(hooks, target_dict, model_options, registered)

    def get_nested_additional_models_sync(self, instance_id: str) -> Any:
        instance = self._get_instance(instance_id)
        return instance.get_nested_additional_models()

    def model_patches_models_sync(self, instance_id: str) -> Any:
        instance = self._get_instance(instance_id)
        return instance.model_patches_models()

    def get_parent_sync(self, instance_id: str) -> Any:
        instance = self._get_instance(instance_id)
        return getattr(instance, "parent", None)

    def current_loaded_device_sync(self, instance_id: str) -> Any:
        instance = self._get_instance(instance_id)
        return instance.current_loaded_device()

    def model_size_sync(self, instance_id: str) -> Any:
        instance = self._get_instance(instance_id)
        return instance.model_size()

    def loaded_size_sync(self, instance_id: str) -> Any:
        instance = self._get_instance(instance_id)
        return instance.loaded_size()

    def model_patches_to_sync(self, instance_id: str, device: Any) -> Any:
        instance = self._get_instance(instance_id)
        return instance.model_patches_to(device)

    def partially_load_sync(self, instance_id: str, device: Any, extra_memory: Any, force_patch_weights: bool = False) -> Any:
        instance = self._get_instance(instance_id)
        return instance.partially_load(device, extra_memory, force_patch_weights=force_patch_weights)

    def process_latent_in_sync(self, instance_id: str, args: tuple, kwargs: dict) -> Any:
        instance = self._get_instance(instance_id)
        return instance.model.process_latent_in(*args, **kwargs)

    def process_latent_out_sync(self, instance_id: str, args: tuple, kwargs: dict) -> Any:
        instance = self._get_instance(instance_id)
        result = instance.model.process_latent_out(*args, **kwargs)
        try:
            target = None
            if args and hasattr(args[0], "device"):
                target = args[0].device
            elif kwargs:
                for v in kwargs.values():
                    if hasattr(v, "device"):
                        target = v.device
                        break
            if target is not None and hasattr(result, "to"):
                return result.to(target)
        except Exception:
            pass
        return result

    def scale_latent_inpaint_sync(self, instance_id: str, args: tuple, kwargs: dict) -> Any:
        instance = self._get_instance(instance_id)
        result = instance.model.scale_latent_inpaint(*args, **kwargs)
        try:
            target = None
            if args and hasattr(args[0], "device"):
                target = args[0].device
            elif kwargs:
                for v in kwargs.values():
                    if hasattr(v, "device"):
                        target = v.device
                        break
            if target is not None and hasattr(result, "to"):
                return result.to(target)
        except Exception:
            pass
        return result

    def get_model_sampling_sync(self, instance_id: str) -> Any:
        instance = self._get_instance(instance_id)
        ms_obj = instance.model.model_sampling
        from comfy.isolation.model_sampling_proxy import ModelSamplingRegistry, ModelSamplingProxy
        ms_id = ModelSamplingRegistry().register(ms_obj)
        return ModelSamplingProxy(ms_id)

    def prepare_state_sync(self, instance_id: str, timestep: Any) -> Any:
        instance = self._get_instance(instance_id)
        cp = getattr(instance.model, "current_patcher", instance)
        return cp.prepare_state(timestep)

    def apply_hooks_sync(self, instance_id: str, hooks: Any) -> Any:
        instance = self._get_instance(instance_id)
        cp = getattr(instance.model, "current_patcher", instance)
        return cp.apply_hooks(hooks=hooks)
    
    # ============================================================
    # Inner model access (for model.model.model_config patterns)
    # ============================================================
    
    @_timing_decorator
    async def get_inner_model_config(self, instance_id: str) -> Any:
        """
        RPC: Get model_config from the inner model.
        
        Used for: model.model.model_config patterns
        """
        instance = self._get_instance(instance_id)
        config = instance.model.model_config
        logger.debug(
            f"][[ModelPatcherRegistry] get_inner_model_config({instance_id}) "
            f"type: {type(config).__name__}"
        )
        return config
    
    def get_inner_model_config_sync(self, instance_id: str) -> Any:
        """Sync version of get_inner_model_config() for host-side calls."""
        instance = self._get_instance(instance_id)
        return instance.model.model_config
    
    @_timing_decorator
    async def get_inner_model_attr(self, instance_id: str, name: str) -> Any:
        """
        RPC: Get an attribute from the inner model.
        
        Used for: model.model.latent_format, model.model.model_type, etc.
        """
        instance = self._get_instance(instance_id)
        try:
            attr = getattr(instance.model, name)
            logger.debug(
                f"][[ModelPatcherRegistry] get_inner_model_attr({instance_id}, '{name}') "
                f"type: {type(attr).__name__}"
            )
            return attr
        except AttributeError:
            logger.debug(
                f"][[ModelPatcherRegistry] get_inner_model_attr({instance_id}, '{name}') missing attribute"
            )
            return None

    @_timing_decorator
    async def inner_model_memory_required(self, instance_id: str, args: tuple, kwargs: dict) -> Any:
        """RPC: Call inner model.memory_required(*args, **kwargs)."""
        instance = self._get_instance(instance_id)
        return instance.model.memory_required(*args, **kwargs)

    @_timing_decorator
    async def inner_model_apply_model(self, instance_id: str, args: tuple, kwargs: dict) -> Any:
        """RPC: Call inner model.apply_model(*args, **kwargs)."""
        instance = self._get_instance(instance_id)
        import torch

        target = getattr(instance, "load_device", None)
        if target is None and args and hasattr(args[0], "device"):
            target = args[0].device
        elif target is None:
            for v in kwargs.values():
                if hasattr(v, "device"):
                    target = v.device
                    break

        def _move(obj):
            if target is None:
                return obj
            if isinstance(obj, (tuple, list)):
                return type(obj)(_move(o) for o in obj)
            if hasattr(obj, "to"):
                return obj.to(target)
            return obj

        moved_args = tuple(_move(a) for a in args)
        moved_kwargs = {k: _move(v) for k, v in kwargs.items()}
        result = instance.model.apply_model(*moved_args, **moved_kwargs)
        return _move(result)
    
    def get_inner_model_attr_sync(self, instance_id: str, name: str) -> Any:
        """Sync version of get_inner_model_attr() for host-side calls."""
        instance = self._get_instance(instance_id)
        try:
            return getattr(instance.model, name)
        except AttributeError:
            return None

    def inner_model_memory_required_sync(self, instance_id: str, args: tuple, kwargs: dict) -> Any:
        instance = self._get_instance(instance_id)
        return instance.model.memory_required(*args, **kwargs)

    def inner_model_apply_model_sync(self, instance_id: str, args: tuple, kwargs: dict) -> Any:
        instance = self._get_instance(instance_id)
        target = getattr(instance, "load_device", None)
        if target is None and args and hasattr(args[0], "device"):
            target = args[0].device
        elif target is None:
            for v in kwargs.values():
                if hasattr(v, "device"):
                    target = v.device
                    break

        def _move(obj):
            if target is None:
                return obj
            if isinstance(obj, (tuple, list)):
                return type(obj)(_move(o) for o in obj)
            if hasattr(obj, "to"):
                return obj.to(target)
            return obj

        moved_args = tuple(_move(a) for a in args)
        moved_kwargs = {k: _move(v) for k, v in kwargs.items()}
        result = instance.model.apply_model(*moved_args, **moved_kwargs)
        return _move(result)
    
    # ============================================================
    # Object Patching (for ModelSamplingAdvanced etc)
    # ============================================================
    
    @_timing_decorator
    async def add_object_patch(self, instance_id: str, name: str, obj: Any) -> None:
        """
        RPC: Add an object patch to the model.
        
        Used by: m.add_object_patch("model_sampling", model_sampling)
        """
        instance = self._get_instance(instance_id)
        instance.add_object_patch(name, obj)
        logger.debug(
            f"][[ModelPatcherRegistry] add_object_patch({instance_id}, '{name}')"
        )
    
    def add_object_patch_sync(self, instance_id: str, name: str, obj: Any) -> None:
        """Sync version of add_object_patch() for host-side calls."""
        instance = self._get_instance(instance_id)
        instance.add_object_patch(name, obj)
    
    # ============================================================
    # ModelSampling RPC (for percent_to_sigma calls)
    # ============================================================
    
    @_timing_decorator
    async def model_sampling_percent_to_sigma(self, instance_id: str, percent: float) -> float:
        """
        RPC: Call model_sampling.percent_to_sigma().
        
        Used by PuLID: work_model.get_model_object("model_sampling").percent_to_sigma(start_at)
        
        Args:
            instance_id: ModelPatcher ID
            percent: Percentage value (0.0 - 1.0)
            
        Returns:
            Sigma value
        """
        instance = self._get_instance(instance_id)
        model_sampling = instance.get_model_object("model_sampling")
        result = model_sampling.percent_to_sigma(percent)
        logger.debug(
            f"][[ModelPatcherRegistry] model_sampling_percent_to_sigma"
            f"({instance_id}, {percent}) = {result}"
        )
        return float(result)
    
    def model_sampling_percent_to_sigma_sync(self, instance_id: str, percent: float) -> float:
        """Sync version of model_sampling_percent_to_sigma()."""
        instance = self._get_instance(instance_id)
        model_sampling = instance.get_model_object("model_sampling")
        return float(model_sampling.percent_to_sigma(percent))
    
    # ============================================================
    # Model Patching (for PuLID attention patching)
    # ============================================================
    
    @_timing_decorator
    async def set_model_patch_replace(
        self, 
        instance_id: str, 
        patch: Any, 
        name: str, 
        block_name: str, 
        number: int, 
        transformer_index: Optional[int] = None
    ) -> None:
        """
        RPC: Set model patch replace.
        
        Used by PuLID: work_model.set_model_patch_replace(patch, "attn1", block_name, number, ...)
        """
        instance = self._get_instance(instance_id)
        instance.set_model_patch_replace(patch, name, block_name, number, transformer_index)
        logger.debug(
            f"][[ModelPatcherRegistry] set_model_patch_replace"
            f"({instance_id}, '{name}', {block_name}, {number})"
        )
    
    def set_model_patch_replace_sync(
        self, 
        instance_id: str, 
        patch: Any, 
        name: str, 
        block_name: str, 
        number: int, 
        transformer_index: Optional[int] = None
    ) -> None:
        """Sync version of set_model_patch_replace()."""
        instance = self._get_instance(instance_id)
        instance.set_model_patch_replace(patch, name, block_name, number, transformer_index)
    
    # ============================================================
    # LoRA Loading (High-level operation that runs entirely on host)
    # ============================================================
    
    @_timing_decorator
    async def load_lora(
        self, 
        instance_id: str, 
        lora_path: str, 
        strength_model: float,
        clip_id: Optional[str] = None,
        strength_clip: float = 1.0
    ) -> dict:
        """
        RPC: Load LoRA and apply to model (and optionally clip).
        
        This is a high-level operation that runs entirely on the host side:
        1. Load LoRA file
        2. Build key maps from model/clip state_dicts
        3. Clone model/clip
        4. Apply patches via add_patches()
        5. Return new model/clip IDs
        
        Args:
            instance_id: ModelPatcher ID
            lora_path: Path to LoRA file (can be relative name or full path)
            strength_model: LoRA strength for model
            clip_id: Optional CLIPRegistry ID (if applying to CLIP too)
            strength_clip: LoRA strength for CLIP
            
        Returns:
            Dict with 'model_id' and optionally 'clip_id' for new patched objects
        """
        import comfy.utils
        import comfy.sd
        import folder_paths
        
        model = self._get_instance(instance_id)
        
        # Get CLIP if provided
        clip = None
        if clip_id:
            from comfy.isolation.clip_proxy import CLIPRegistry
            clip_registry = CLIPRegistry()
            clip = clip_registry._get_instance(clip_id)
        
        # Resolve LoRA path
        lora_full_path = folder_paths.get_full_path("loras", lora_path)
        if lora_full_path is None:
            raise ValueError(f"LoRA file not found: {lora_path}")
        
        # Load LoRA weights
        lora = comfy.utils.load_torch_file(lora_full_path)
        
        # Apply LoRA using standard ComfyUI function
        new_model, new_clip = comfy.sd.load_lora_for_models(
            model, clip, lora, strength_model, strength_clip
        )
        
        # Register new model
        new_model_id = self.register(new_model) if new_model else None
        
        # Register new CLIP if applicable
        new_clip_id = None
        if new_clip and clip_id:
            from comfy.isolation.clip_proxy import CLIPRegistry
            clip_registry = CLIPRegistry()
            new_clip_id = clip_registry.register(new_clip)
        
        logger.info(
            f"][[ModelPatcherRegistry] load_lora({instance_id}, '{lora_path}') "
            f"→ model={new_model_id}, clip={new_clip_id}"
        )
        
        return {"model_id": new_model_id, "clip_id": new_clip_id}
    
    def load_lora_sync(
        self, 
        instance_id: str, 
        lora_path: str, 
        strength_model: float,
        clip_id: Optional[str] = None,
        strength_clip: float = 1.0
    ) -> dict:
        """Sync version of load_lora() for host-side calls."""
        import comfy.utils
        import comfy.sd
        import folder_paths
        
        model = self._get_instance(instance_id)
        
        # Get CLIP if provided
        clip = None
        if clip_id:
            from comfy.isolation.clip_proxy import CLIPRegistry
            clip_registry = CLIPRegistry()
            clip = clip_registry._get_instance(clip_id)
        
        # Resolve LoRA path
        lora_full_path = folder_paths.get_full_path("loras", lora_path)
        if lora_full_path is None:
            raise ValueError(f"LoRA file not found: {lora_path}")
        
        # Load LoRA weights
        lora = comfy.utils.load_torch_file(lora_full_path)
        
        # Apply LoRA using standard ComfyUI function
        new_model, new_clip = comfy.sd.load_lora_for_models(
            model, clip, lora, strength_model, strength_clip
        )
        
        # Register new model
        new_model_id = self.register(new_model) if new_model else None
        
        # Register new CLIP if applicable
        new_clip_id = None
        if new_clip and clip_id:
            from comfy.isolation.clip_proxy import CLIPRegistry
            clip_registry = CLIPRegistry()
            new_clip_id = clip_registry.register(new_clip)
        
        logger.info(
            f"][[ModelPatcherRegistry] load_lora_sync({instance_id}, '{lora_path}') "
            f"→ model={new_model_id}, clip={new_clip_id}"
        )
        
        return {"model_id": new_model_id, "clip_id": new_clip_id}


class ModelSamplingProxy:
    """
    Proxy for ModelSampling objects that can't be pickled.
    
    The actual ModelSampling class is dynamically created inside model_sampling()
    function, making it unpicklable. This proxy forwards method calls via RPC.
    """
    
    def __init__(self, instance_id: str, registry: Optional[ModelPatcherRegistry] = None):
        """
        Initialize ModelSamplingProxy.
        
        Args:
            instance_id: ModelPatcher ID that owns this model_sampling
            registry: ModelPatcherRegistry for RPC calls
        """
        self._instance_id = instance_id
        self._registry = registry if registry is not None else ModelPatcherRegistry()
        self._is_child = os.environ.get("PYISOLATE_CHILD") == "1"
    
    def __reduce__(self):
        """Custom pickle - only serialize instance_id."""
        return (_reconstruct_model_sampling_proxy, (self._instance_id,))
    
    def percent_to_sigma(self, percent: float) -> float:
        """
        Forward percent_to_sigma call to host.
        
        Args:
            percent: Percentage value (0.0 - 1.0)
            
        Returns:
            Sigma value
        """
        if self._is_child:
            from comfy.isolation.rpc_bridge import RpcBridge
            bridge = RpcBridge()
            method = getattr(self._registry, 'model_sampling_percent_to_sigma')
            return bridge.run_sync(method(self._instance_id, percent))
        else:
            return self._registry.model_sampling_percent_to_sigma_sync(self._instance_id, percent)


def _reconstruct_model_sampling_proxy(instance_id: str) -> ModelSamplingProxy:
    """Pickle reconstruction for ModelSamplingProxy."""
    return ModelSamplingProxy(instance_id, registry=None)


class ModelPatcherProxy:
    """
    Lightweight, picklable handle to a ModelPatcher instance.
    
    Design Principles (from CLIPProxy):
    1. Zero State: Only stores instance_id + registry reference
    2. Host Optimization: Bypasses RPC when running on host (_is_child=False)
    3. Transparent: Appears identical to ModelPatcher from node's perspective
    4. Fail-Loud: Any RPC failure raises immediately (no silent failures)
    
    Research References:
    - Shadow-Reference Protocol Section 5.3 (Shadow Patcher concept)
    - CLIPProxy implementation (proven pattern)
    """
    
    def __init__(
        self,
        instance_id: str,
        registry: Optional[ModelPatcherRegistry] = None,
        manage_lifecycle: bool = False
    ):
        """
        Initialize ModelPatcherProxy.
        
        Args:
            instance_id: Registry ID of the ModelPatcher instance
            registry: ModelPatcherRegistry singleton (auto-created if None)
            manage_lifecycle: If True, proxy manages cleanup via weakref.finalize
        """
        self._instance_id = instance_id
        self._manage_lifecycle = manage_lifecycle
        self._is_child = os.environ.get("PYISOLATE_CHILD") == "1"
        
        # Registry passed in explicitly (from deserialization or manual creation)
        self._registry = registry if registry is not None else ModelPatcherRegistry()
        
        # Lifecycle: only host-side proxy cleans up
        if manage_lifecycle and not self._is_child:
            self._finalizer = weakref.finalize(
                self, self._registry.unregister_sync, instance_id
            )
            logger.debug(
                f"][[ModelPatcherProxy] Lifecycle management enabled for {instance_id}"
            )
    
    def __reduce__(self):
        """
        Custom pickle - only serialize instance_id.
        
        Returns tuple for pickle reconstruction with is_new_object=False
        to prevent double-finalize on round-trip.
        """
        return (_reconstruct_model_patcher_proxy, (self._instance_id, False))
    
    def _call_registry(self, method_name: str, *args, **kwargs):
        """
        Call registry method with host-side optimization and timing.
        
        If running on host: call directly (sync - registry methods are NOT async on host)
        If running in child: use RpcBridge for sync-to-async RPC
        
        Debug Checkpoint: Logs RPC call with duration.
        
        Args:
            method_name: Name of ModelPatcherRegistry method to call
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Result from registry method
        """
        start = time.perf_counter()
        
        # Get registry dynamically (important for child processes)
        if self._registry is None:
            self._registry = ModelPatcherRegistry()
        
        try:
            if self._is_child:
                # Child process: RPC to host via ProxiedSingleton mechanism
                from comfy.isolation.rpc_bridge import RpcBridge
                bridge = RpcBridge()
                method = getattr(self._registry, method_name)
                result = bridge.run_sync(method(self._instance_id, *args, **kwargs))
            else:
                # Host process: direct SYNC call to registry
                # Registry methods are sync on host side - no async needed
                sync_method_name = f"{method_name}_sync"
                if hasattr(self._registry, sync_method_name):
                    method = getattr(self._registry, sync_method_name)
                    result = method(self._instance_id, *args, **kwargs)
                else:
                    # Fallback: call the method directly if it's not async
                    method = getattr(self._registry, method_name)
                    result = method(self._instance_id, *args, **kwargs)
            
            duration_ms = (time.perf_counter() - start) * 1000
            logger.debug(
                f"][[ModelPatcherProxy] "
                f"{method_name}() completed in {duration_ms:.2f}ms "
                f"(child={self._is_child})"
            )
            return result
            
        except Exception as e:
            duration_ms = (time.perf_counter() - start) * 1000
            logger.error(
                f"][[ModelPatcherProxy] "
                f"{method_name}() FAILED in {duration_ms:.2f}ms: {e}"
            )
            raise
    
    # ============================================================
    # Phase 1 Methods (7 core operations for PuLID)
    # ============================================================
    
    def clone(self) -> 'ModelPatcherProxy':
        """
        Clone ModelPatcher instance (Deep Remote Copy pattern).
        
        Creates a new ModelPatcher in the registry and returns a new proxy.
        Host-side clones manage lifecycle; child-side do not.
        
        Returns:
            New ModelPatcherProxy for the cloned instance
        """
        new_id = self._call_registry('clone')
        # Host-side clones manage lifecycle; child-side do not
        return ModelPatcherProxy(new_id, self._registry, manage_lifecycle=not self._is_child)
    
    def get_model_object(self, name: str) -> Any:
        """
        Get a named object from the model.
        
        Used by PuLID for: model.get_model_object("model_sampling")
        
        For "model_sampling", returns a ModelSamplingProxy since the actual
        ModelSampling class is dynamically created and can't be pickled.
        
        Args:
            name: Object name (e.g., "model_sampling")
            
        Returns:
            The requested object or a proxy for unpicklable objects
        """
        # ModelSampling is a dynamically-created local class that can't be pickled
        # Return a proxy that forwards method calls via RPC
        if name == "model_sampling":
            return ModelSamplingProxy(self._instance_id, self._registry)
        
        # For other objects, try direct RPC (may fail for unpicklable objects)
        return self._call_registry('get_model_object', name)
    
    @property
    def model_options(self) -> dict:
        """
        Get model_options dict.
        
        Research: Shadow-Reference Protocol Finding 2 (Lazy Patch Application)
        """
        return self._call_registry('get_model_options')
    
    @model_options.setter
    def model_options(self, value: dict) -> None:
        """
        Set model_options dict.
        
        Debug Checkpoint: Value is logged in registry method for validation.
        """
        self._call_registry('set_model_options', value)
    
    @property
    def load_device(self) -> Any:
        """Get load_device property."""
        return self._call_registry('get_load_device')
    
    @property
    def offload_device(self) -> Any:
        """Get offload_device property."""
        return self._call_registry('get_offload_device')

    @property
    def current_loaded_device(self) -> Any:
        """Get current_loaded_device() value."""
        return self._call_registry('current_loaded_device')

    def prepare_state(self, timestep):
        """Prepare sampler state (delegated to current_patcher)."""
        return self._call_registry('prepare_state', timestep)

    def apply_hooks(self, hooks):
        """Apply attention hooks via current_patcher."""
        return self._call_registry('apply_hooks', hooks)
    
    @property
    def size(self) -> int:
        """Get model size in bytes."""
        return self._call_registry('get_size')

    @property
    def wrappers(self):
        """Wrappers mapping used by sampler helpers."""
        return self._call_registry('get_wrappers')

    @property
    def callbacks(self):
        """Callbacks mapping used by sampler helpers."""
        return self._call_registry('get_callbacks')

    @property
    def hook_mode(self):
        """Forward hook_mode property used by samplers."""
        return self._call_registry('get_hook_mode')

    @hook_mode.setter
    def hook_mode(self, value) -> None:
        self._call_registry('set_hook_mode', value)

    def model_dtype(self):
        """Return model dtype (delegated)."""
        return self._call_registry('model_dtype')

    def pre_run(self) -> None:
        """Lifecycle hook before sampling."""
        return self._call_registry('pre_run')

    def cleanup(self) -> None:
        """Lifecycle cleanup after sampling."""
        return self._call_registry('cleanup')

    def restore_hook_patches(self) -> None:
        """Restore hook patches after sampling."""
        return self._call_registry('restore_hook_patches')

    def register_all_hook_patches(self, hooks, target_dict, model_options, registered) -> None:
        """Forward register_all_hook_patches used during sampler prep."""
        return self._call_registry('register_all_hook_patches', hooks, target_dict, model_options, registered)

    def get_nested_additional_models(self):
        """Return nested additional models for sampler helpers."""
        return self._call_registry('get_nested_additional_models')

    def model_patches_models(self):
        """Return patched models list for model_management.load_models_gpu."""
        return self._call_registry('model_patches_models')

    @property
    def parent(self):
        """Expose parent attribute (may be None)."""
        return self._call_registry('get_parent')

    def current_loaded_device(self):
        """Expose current_loaded_device from model patcher."""
        return self._call_registry('current_loaded_device')

    def model_size(self):
        """Expose model_size from model patcher."""
        return self._call_registry('model_size')

    def loaded_size(self):
        """Expose loaded_size from model patcher."""
        return self._call_registry('loaded_size')

    def model_patches_to(self, device):
        """Move model patches to device."""
        return self._call_registry('model_patches_to', device)

    def partially_load(self, device, extra_memory, force_patch_weights: bool = False):
        """Partially load model to device."""
        return self._call_registry('partially_load', device, extra_memory, force_patch_weights)
    
    # ============================================================
    # Object Patching Methods (needed for ModelSamplingAdvanced etc)
    # ============================================================
    
    def add_object_patch(self, name: str, obj: Any) -> None:
        """
        Add an object patch to the model.
        
        Used by ModelSamplingAdvanced: m.add_object_patch("model_sampling", model_sampling)
        """
        return self._call_registry('add_object_patch', name, obj)
    
    # ============================================================
    # Inner Model Access (for model.model.model_config patterns)
    # ============================================================
    
    @property
    def model(self):
        """
        Access to inner model object.
        
        Returns a lightweight proxy that forwards attribute access to
        the actual model on the host side.
        
        Research: Shadow-Reference Protocol - this is needed for:
        - model.model.model_config access patterns
        - ModelSamplingAdvanced(model.model.model_config)
        """
        return _InnerModelProxy(self._instance_id, self._registry, self._is_child)
    
    # ============================================================
    # Property Guards (Phase 1 - Raise AttributeError for unsupported)
    # ============================================================
    
    @property
    def patches(self):
        """Property guard: patches access not supported in Phase 1."""
        raise AttributeError(
            "][[ModelPatcherProxy] Direct access to 'patches' is not supported "
            "in isolated mode Phase 1. This will be added in Phase 2/3."
        )
    
    @property
    def object_patches(self):
        """Property guard: object_patches access not supported in Phase 1."""
        raise AttributeError(
            "][[ModelPatcherProxy] Direct access to 'object_patches' is not supported "
            "in isolated mode Phase 1. This will be added in Phase 2/3."
        )
    
    def add_patches(self, *args, **kwargs):
        """Method guard: add_patches not supported in Phase 1."""
        raise NotImplementedError(
            "][[ModelPatcherProxy] add_patches() is not supported "
            "in isolated mode Phase 1. Use load_lora() instead for LoRA loading."
        )
    
    def load_lora(
        self, 
        lora_path: str, 
        strength_model: float,
        clip: Optional['CLIPProxy'] = None,
        strength_clip: float = 1.0
    ) -> tuple:
        """
        Load LoRA and apply to model (and optionally clip).
        
        This is the high-level LoRA loading API for isolated nodes.
        Runs entirely on the host side where the real ModelPatcher lives.
        
        Args:
            lora_path: Path to LoRA file (relative name or full path)
            strength_model: LoRA strength for model
            clip: Optional CLIPProxy (if applying to CLIP too)
            strength_clip: LoRA strength for CLIP
            
        Returns:
            Tuple of (new_model_proxy, new_clip_proxy or None)
        """
        # Get clip_id from proxy if provided
        clip_id = None
        if clip is not None:
            # CLIPProxy uses _instance_id
            clip_id = getattr(clip, '_instance_id', None)
            if clip_id is None:
                # Fallback for any legacy proxies
                clip_id = getattr(clip, '_clip_id', None)
        
        logger.debug(
            f"][[ModelPatcherProxy] load_lora: clip={type(clip).__name__ if clip else None}, "
            f"clip_id={clip_id}"
        )
        
        result = self._call_registry(
            'load_lora', 
            lora_path, 
            strength_model,
            clip_id,
            strength_clip
        )
        
        # Reconstruct proxies from returned IDs
        new_model = None
        if result.get("model_id"):
            new_model = ModelPatcherProxy(
                result["model_id"], 
                self._registry, 
                manage_lifecycle=not self._is_child
            )
        
        new_clip = None
        if result.get("clip_id"):
            from comfy.isolation.clip_proxy import CLIPProxy
            new_clip = CLIPProxy(result["clip_id"])
        
        return (new_model, new_clip)
    
    def patch_model(self, *args, **kwargs):
        """Method guard: patch_model not supported in Phase 1."""
        raise NotImplementedError(
            "][[ModelPatcherProxy] patch_model() is not supported "
            "in isolated mode Phase 1. This will be added in Phase 4."
        )
    
    def unpatch_model(self, *args, **kwargs):
        """Method guard: unpatch_model not supported in Phase 1."""
        raise NotImplementedError(
            "][[ModelPatcherProxy] unpatch_model() is not supported "
            "in isolated mode Phase 1. This will be added in Phase 4."
        )


def _reconstruct_model_patcher_proxy(model_id: str, is_new_object: bool = True) -> ModelPatcherProxy:
    """
    Pickle reconstruction helper.
    
    Args:
        model_id: Registry ID of the ModelPatcher instance
        is_new_object: True if this is a NEW object (e.g., clone result),
                       False if this is a round-trip of existing proxy.
                       
    Lifecycle Rules:
        - Child process: NEVER manage lifecycle (always False)
        - Host process, new object: manage lifecycle (True)
        - Host process, round-trip: do NOT manage (False) - original proxy owns it
    
    Returns:
        Reconstructed ModelPatcherProxy
    """
    IS_CHILD = os.environ.get("PYISOLATE_CHILD") == "1"
    # Don't instantiate ModelPatcherRegistry() here - let proxy._call_registry do it lazily
    # This prevents "Cannot inject instance after first instantiation" errors
    registry = None
    
    if IS_CHILD:
        # Child never manages lifecycle
        return ModelPatcherProxy(model_id, registry, manage_lifecycle=False)
    else:
        # Host: manage only if this is a new object (clone, etc.)
        return ModelPatcherProxy(model_id, registry, manage_lifecycle=is_new_object)


class _InnerModelProxy:
    """
    Lightweight proxy for accessing the inner model object.
    
    Supports patterns like: model.model.model_config
    Forwards attribute access to the actual model on the host side.
    """
    
    def __init__(self, instance_id: str, registry, is_child: bool):
        self._instance_id = instance_id
        self._registry = registry
        self._is_child = is_child
    
    def __getattr__(self, name: str):
        """
        Forward attribute access to the inner model.
        
        Supports common access patterns needed by nodes.
        """
        if name.startswith('_'):
            raise AttributeError(name)
        
        # Supported attributes - add more as needed
        if name in ('model_config', 'latent_format', 'model_type', 'extra_conds_shapes'):
            return self._get_inner_model_attr(name)

        if name == 'memory_required':
            def _memory_required(*args, **kwargs):
                if self._is_child:
                    from comfy.isolation.rpc_bridge import RpcBridge
                    bridge = RpcBridge()
                    method = getattr(self._registry, 'inner_model_memory_required')
                    return bridge.run_sync(method(self._instance_id, args, kwargs))
                else:
                    sync_method = getattr(self._registry, 'inner_model_memory_required_sync')
                    return sync_method(self._instance_id, args, kwargs)
            return _memory_required

        if name == 'apply_model':
            def _apply_model(*args, **kwargs):
                if self._is_child:
                    from comfy.isolation.rpc_bridge import RpcBridge
                    bridge = RpcBridge()
                    method = getattr(self._registry, 'inner_model_apply_model')
                    return bridge.run_sync(method(self._instance_id, args, kwargs))
                else:
                    sync_method = getattr(self._registry, 'inner_model_apply_model_sync')
                    return sync_method(self._instance_id, args, kwargs)
            return _apply_model

        if name == 'process_latent_in':
            def _process_latent_in(*args, **kwargs):
                if self._is_child:
                    from comfy.isolation.rpc_bridge import RpcBridge
                    bridge = RpcBridge()
                    method = getattr(self._registry, 'process_latent_in')
                    return bridge.run_sync(method(self._instance_id, args, kwargs))
                else:
                    sync_method = getattr(self._registry, 'process_latent_in_sync')
                    return sync_method(self._instance_id, args, kwargs)
            return _process_latent_in

        if name == 'extra_conds':
            return self._get_inner_model_attr('extra_conds')

        if name == 'model_sampling':
            if self._is_child:
                from comfy.isolation.model_sampling_proxy import ModelSamplingProxy
                from comfy.isolation.rpc_bridge import RpcBridge
                bridge = RpcBridge()
                proxy_id = bridge.run_sync(self._registry.get_model_sampling(self._instance_id))._instance_id
                return ModelSamplingProxy(proxy_id)
            else:
                return self._registry.get_model_sampling_sync(self._instance_id)

        if name == 'current_patcher':
            if self._is_child:
                return ModelPatcherProxy(self._instance_id, self._registry, manage_lifecycle=False)
            return self._get_inner_model_attr('current_patcher')

        if name == 'scale_latent_inpaint':
            def _scale_latent_inpaint(*args, **kwargs):
                if self._is_child:
                    from comfy.isolation.rpc_bridge import RpcBridge
                    bridge = RpcBridge()
                    method = getattr(self._registry, 'scale_latent_inpaint')
                    return bridge.run_sync(method(self._instance_id, args, kwargs))
                else:
                    sync_method = getattr(self._registry, 'scale_latent_inpaint_sync')
                    return sync_method(self._instance_id, args, kwargs)
            return _scale_latent_inpaint

        if name == 'process_latent_out':
            def _process_latent_out(*args, **kwargs):
                if self._is_child:
                    from comfy.isolation.rpc_bridge import RpcBridge
                    bridge = RpcBridge()
                    method = getattr(self._registry, 'process_latent_out')
                    return bridge.run_sync(method(self._instance_id, args, kwargs))
                else:
                    sync_method = getattr(self._registry, 'process_latent_out_sync')
                    return sync_method(self._instance_id, args, kwargs)
            return _process_latent_out

        if name == 'load_device':
            if self._is_child:
                from comfy.isolation.rpc_bridge import RpcBridge
                bridge = RpcBridge()
                method = getattr(self._registry, 'get_inner_model_attr')
                try:
                    return bridge.run_sync(method(self._instance_id, 'load_device'))
                except Exception:
                    return None
            else:
                try:
                    return self._get_inner_model_attr('load_device')
                except AttributeError:
                    return None
        
        # For other attributes, log and raise
        logger.warning(
            f"][[_InnerModelProxy] Unsupported attribute access: model.{name}"
        )
        raise AttributeError(
            f"][[_InnerModelProxy] Access to 'model.{name}' is not yet supported. "
            f"Please file an issue if you need this."
        )
    
    def _get_inner_model_attr(self, name: str):
        """Get an attribute from the inner model."""
        if self._is_child:
            from comfy.isolation.rpc_bridge import RpcBridge
            bridge = RpcBridge()
            method = getattr(self._registry, 'get_inner_model_attr')
            return bridge.run_sync(method(self._instance_id, name))
        else:
            # Host side: direct sync call
            return self._registry.get_inner_model_attr_sync(self._instance_id, name)


def maybe_wrap_model_for_isolation(model_patcher):
    """
    Wrap ModelPatcher in isolation proxy if isolation is active.
    
    Called from checkpoint loading path.
    Returns original model_patcher if:
    - Isolation not active
    - Already in child process
    - Already a ModelPatcherProxy
    
    Debug Checkpoint: Logs "MODEL wrapping: enabled/disabled" per call.
    
    Research: Shadow-Reference Protocol Section 5.4 (Transport Layer)
    
    Args:
        model_patcher: ModelPatcher instance to potentially wrap
        
    Returns:
        ModelPatcherProxy if isolation active, otherwise original model_patcher
    """
    isolation_active = os.environ.get("PYISOLATE_ISOLATION_ACTIVE") == "1"
    is_child = os.environ.get("PYISOLATE_CHILD") == "1"
    
    # Always log wrapper calls at INFO level for debugging
    logger.info(
        f"][[ModelPatcherProxy] maybe_wrap called: "
        f"isolation_active={isolation_active}, is_child={is_child}, "
        f"type={type(model_patcher).__name__}"
    )
    
    if not isolation_active:
        logger.debug(
            "][[ModelPatcherProxy] MODEL wrapping: disabled (isolation not active)"
        )
        return model_patcher
    
    if is_child:
        logger.debug(
            "][[ModelPatcherProxy] MODEL wrapping: disabled (in child process)"
        )
        return model_patcher
    
    if isinstance(model_patcher, ModelPatcherProxy):
        logger.debug(
            "][[ModelPatcherProxy] MODEL wrapping: skipped (already proxied)"
        )
        return model_patcher
    
    registry = ModelPatcherRegistry()
    model_id = registry.register(model_patcher)
    logger.debug(
        f"][[ModelPatcherProxy] MODEL wrapping: enabled → {model_id}"
    )
    return ModelPatcherProxy(model_id, registry, manage_lifecycle=True)
