"""
Stateless RPC pattern for VAE instances.

This module provides:
1. VAERegistry - Host-side registry of VAE instances (ProxiedSingleton)
2. VAEProxy - Picklable handle that forwards calls via RPC

Architecture mirrors clip_proxy.py exactly.
"""

import asyncio
import logging
import os
import pickle
import threading
import weakref
from typing import Any, Dict, Optional

from pyisolate import ProxiedSingleton

logger = logging.getLogger(__name__)

# Host/child detection
IS_CHILD_PROCESS = os.environ.get("PYISOLATE_CHILD") == "1"

_thread_local = threading.local()


def _get_thread_loop() -> asyncio.AbstractEventLoop:
    """Return a per-thread event loop, creating it if missing."""
    loop = getattr(_thread_local, "loop", None)
    if loop is None or loop.is_closed():
        loop = asyncio.new_event_loop()
        _thread_local.loop = loop
    return loop


def _run_coro_in_new_loop(coro):
    """Execute coroutine in a fresh thread-bound event loop and return result."""
    result_box = {}
    exc_box = {}

    def runner():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result_box["value"] = loop.run_until_complete(coro)
        except Exception as exc:  # noqa: BLE001
            exc_box["exc"] = exc
        finally:
            loop.close()

    t = threading.Thread(target=runner, daemon=True)
    t.start()
    t.join()
    if "exc" in exc_box:
        raise exc_box["exc"]
    return result_box.get("value")


def _detach_if_grad(obj):
    """Detach tensors that require grad to make them multiprocess-safe."""
    try:
        import torch
    except Exception:
        return obj

    if isinstance(obj, torch.Tensor):
        return obj.detach() if obj.requires_grad else obj
    if isinstance(obj, (list, tuple)):
        return type(obj)(_detach_if_grad(x) for x in obj)
    if isinstance(obj, dict):
        return {k: _detach_if_grad(v) for k, v in obj.items()}
    return obj


class VAERegistry(ProxiedSingleton):
    """
    Host-side registry of VAE instances using ProxiedSingleton pattern.
    
    Thread-safe singleton that manages VAE object lifecycle and provides
    async RPC methods for isolated child processes.
    
    CRITICAL: Inherits from ProxiedSingleton to enable RPC from child processes.
    """
    
    def __init__(self) -> None:
        """Initialize registry state (called once by ProxiedSingleton)."""
        if hasattr(ProxiedSingleton, "__init__") and ProxiedSingleton is not object:
            super().__init__()
        
        self._registry: Dict[str, Any] = {}
        self._id_map: Dict[int, str] = {}  # id(vae) → instance_id (identity preservation)
        self._counter = 0
        self._lock = threading.Lock()
        logger.debug("][[VAERegistry] Initialized")
    
    def register(self, vae_instance) -> str:
        """
        Register a VAE instance and return unique ID.
        
        If the same Python object (by id()) was already registered,
        returns the existing ID to preserve identity semantics.
        
        Args:
            vae_instance: VAE object to register
            
        Returns:
            Unique instance ID (e.g., "vae_0")
            
        Raises:
            RuntimeError: If called from child process
        """
        # No check needed - if we can call register(), we're on the host
        # The environment variable alone doesn't determine process context
        
        with self._lock:
            # Check if already registered (identity preservation)
            obj_id = id(vae_instance)
            if obj_id in self._id_map:
                existing_id = self._id_map[obj_id]
                logger.debug(
                    f"][[VAERegistry] Re-using {existing_id} for object {obj_id}"
                )
                return existing_id
            
            # New registration
            instance_id = f"vae_{self._counter}"
            self._counter += 1
            self._registry[instance_id] = vae_instance
            self._id_map[obj_id] = instance_id
            logger.debug(
                f"][[VAERegistry] Registered {instance_id}"
            )
        
        return instance_id
    
    def unregister_sync(self, instance_id: str) -> None:
        """
        Unregister a VAE instance (called by weakref.finalize).
        
        Thread-safe synchronous cleanup for weakref callback compatibility.
        
        Args:
            instance_id: ID to unregister
        """
        with self._lock:
            vae = self._registry.pop(instance_id, None)
            if vae:
                obj_id = id(vae)
                self._id_map.pop(obj_id, None)
                logger.debug(f"][[VAERegistry] Unregistered {instance_id}")
    
    def _get_instance(self, instance_id: str):
        """Get VAE instance by ID (internal, host-side only)."""
        if IS_CHILD_PROCESS:
            raise RuntimeError(
                "][[VAERegistry] FAIL-LOUD: "
                "_get_instance called in child process"
            )
        with self._lock:
            return self._registry.get(instance_id)
    
    # RPC methods below (async for pyisolate compatibility)
    
    async def encode(self, instance_id: str, pixels):
        """RPC: Encode pixels to latent via VAE.encode()."""
        vae = self._get_instance(instance_id)
        if vae is None:
            raise ValueError(f"VAE {instance_id} not found in registry")
        return _detach_if_grad(vae.encode(pixels))
    
    async def encode_tiled(self, instance_id: str, pixels, tile_x: int = 512, tile_y: int = 512, overlap: int = 64):
        """RPC: Tiled encode for large images."""
        vae = self._get_instance(instance_id)
        if vae is None:
            raise ValueError(f"VAE {instance_id} not found in registry")
        return _detach_if_grad(
            vae.encode_tiled(pixels, tile_x=tile_x, tile_y=tile_y, overlap=overlap)
        )
    
    async def decode(self, instance_id: str, samples, **kwargs):
        """RPC: Decode latent to pixels via VAE.decode()."""
        vae = self._get_instance(instance_id)
        if vae is None:
            raise ValueError(f"VAE {instance_id} not found in registry")
        return _detach_if_grad(vae.decode(samples, **kwargs))
    
    async def decode_tiled(self, instance_id: str, samples, tile_x: int = 64, tile_y: int = 64, overlap: int = 16, **kwargs):
        """RPC: Tiled decode for large latents."""
        vae = self._get_instance(instance_id)
        if vae is None:
            raise ValueError(f"VAE {instance_id} not found in registry")
        return _detach_if_grad(
            vae.decode_tiled(samples, tile_x=tile_x, tile_y=tile_y, overlap=overlap, **kwargs)
        )
    
    async def get_sd(self, instance_id: str):
        """RPC: Get VAE state dict."""
        vae = self._get_instance(instance_id)
        if vae is None:
            raise ValueError(f"VAE {instance_id} not found in registry")
        return vae.get_sd()


class VAEProxy:
    """
    Picklable proxy for VAE that forwards calls to host-side registry via RPC.
    
    Design:
    - Pickle-safe (stores only instance_id string)
    - Lazily acquires RPC caller on first method call
    - Mirrors VAE interface for transparent substitution
    """
    
    __module__ = 'comfy.sd'  # Mimic real VAE module for compatibility
    
    def __init__(self, instance_id: str):
        """
        Initialize proxy with registry ID.
        
        Args:
            instance_id: Unique ID from VAERegistry.register()
        """
        self._instance_id = instance_id
        self._rpc_caller = None
        logger.debug(f"][[VAEProxy] Created for {instance_id}")
    
    def _get_rpc(self):
        """Lazy RPC caller acquisition."""
        if self._rpc_caller is None:
            from pyisolate._internal.shared import get_child_rpc_instance
            rpc = get_child_rpc_instance()
            if rpc is None:
                raise RuntimeError(
                    "][[VAEProxy] FAIL-LOUD: "
                    "No RPC instance available in child process"
                )
            self._rpc_caller = rpc.create_caller(VAERegistry, VAERegistry.get_remote_id())
        return self._rpc_caller
    
    def encode(self, pixels):
        """Encode pixels → latent (async via RPC)."""
        rpc = self._get_rpc()
        try:
            asyncio.get_running_loop()
            return _run_coro_in_new_loop(rpc.encode(self._instance_id, pixels))
        except RuntimeError:
            loop = _get_thread_loop()
            return loop.run_until_complete(rpc.encode(self._instance_id, pixels))
    
    def encode_tiled(self, pixels, tile_x: int = 512, tile_y: int = 512, overlap: int = 64):
        """Tiled encode for large images."""
        rpc = self._get_rpc()
        try:
            asyncio.get_running_loop()
            return _run_coro_in_new_loop(
                rpc.encode_tiled(self._instance_id, pixels, tile_x, tile_y, overlap)
            )
        except RuntimeError:
            loop = _get_thread_loop()
            return loop.run_until_complete(
                rpc.encode_tiled(self._instance_id, pixels, tile_x, tile_y, overlap)
            )
    
    def decode(self, samples, **kwargs):
        """Decode latent → pixels (async via RPC)."""
        rpc = self._get_rpc()
        try:
            asyncio.get_running_loop()
            return _run_coro_in_new_loop(rpc.decode(self._instance_id, samples, **kwargs))
        except RuntimeError:
            loop = _get_thread_loop()
            return loop.run_until_complete(rpc.decode(self._instance_id, samples, **kwargs))

    def decode_tiled(self, samples, tile_x: int = 64, tile_y: int = 64, overlap: int = 16, **kwargs):
        """Tiled decode for large latents."""
        rpc = self._get_rpc()
        try:
            asyncio.get_running_loop()
            return _run_coro_in_new_loop(
                rpc.decode_tiled(self._instance_id, samples, tile_x, tile_y, overlap, **kwargs)
            )
        except RuntimeError:
            loop = _get_thread_loop()
            return loop.run_until_complete(
                rpc.decode_tiled(self._instance_id, samples, tile_x, tile_y, overlap, **kwargs)
            )
    
    def get_sd(self):
        """Get VAE state dict."""
        rpc = self._get_rpc()
        try:
            asyncio.get_running_loop()
            return _run_coro_in_new_loop(rpc.get_sd(self._instance_id))
        except RuntimeError:
            loop = _get_thread_loop()
            return loop.run_until_complete(rpc.get_sd(self._instance_id))
    
    def __getstate__(self):
        """Pickle support: only serialize instance_id."""
        return {"_instance_id": self._instance_id}
    
    def __setstate__(self, state):
        """Unpickle support: restore instance_id, reset RPC caller."""
        self._instance_id = state["_instance_id"]
        self._rpc_caller = None
        logger.debug(f"][[VAEProxy] Restored from pickle: {self._instance_id}")
    
    def __repr__(self):
        return f"<VAEProxy {self._instance_id}>"


# Registry instantiated in host_hooks.initialize_host_process; keep optional safety
if not IS_CHILD_PROCESS:
    _VAE_REGISTRY_SINGLETON = VAERegistry()
