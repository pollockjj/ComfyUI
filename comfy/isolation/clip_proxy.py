"""
Stateless RPC pattern for CLIP instances.

This module provides:
1. CLIPRegistry - Host-side registry of CLIP instances (ProxiedSingleton)
2. CLIPProxy - Picklable handle that forwards calls via RPC

Architecture mirrors vae_proxy.py exactly.
"""

import asyncio
import logging
import os
import threading
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


class CLIPRegistry(ProxiedSingleton):
    """
    Host-side registry of CLIP instances using ProxiedSingleton pattern.

    Thread-safe singleton that manages CLIP object lifecycle and provides
    async RPC methods for isolated child processes.

    CRITICAL: Inherits from ProxiedSingleton to enable RPC from child processes.
    """

    def __init__(self) -> None:
        """Initialize registry state (called once by ProxiedSingleton)."""
        if hasattr(ProxiedSingleton, "__init__") and ProxiedSingleton is not object:
            super().__init__()

        self._registry: Dict[str, Any] = {}
        self._id_map: Dict[int, str] = {}  # id(clip) → instance_id (identity preservation)
        self._counter = 0
        self._lock = threading.Lock()
        logger.debug("]] [CLIPRegistry] Initialized")

    def register(self, clip_instance) -> str:
        """
        Register a CLIP instance and return unique ID.

        If the same Python object (by id()) was already registered,
        returns the existing ID to preserve identity semantics.

        Args:
            clip_instance: CLIP object to register

        Returns:
            Unique instance ID (e.g., "clip_0")

        Raises:
            RuntimeError: If called from child process
        """
        # No check needed - if we can call register(), we're on the host
        # The environment variable alone doesn't determine process context

        with self._lock:
            # Check if already registered (identity preservation)
            obj_id = id(clip_instance)
            if obj_id in self._id_map:
                existing_id = self._id_map[obj_id]
                logger.debug(
                    f"][[CLIPRegistry] Re-using {existing_id} for object {obj_id}"
                )
                return existing_id

            # New registration
            instance_id = f"clip_{self._counter}"
            self._counter += 1
            self._registry[instance_id] = clip_instance
            self._id_map[obj_id] = instance_id
            logger.debug(
                f"][[CLIPRegistry] Registered {instance_id}"
            )

        return instance_id

    def unregister_sync(self, instance_id: str) -> None:
        """
        Unregister a CLIP instance (called by weakref.finalize).

        Thread-safe synchronous cleanup for weakref callback compatibility.

        Args:
            instance_id: ID to unregister
        """
        with self._lock:
            clip = self._registry.pop(instance_id, None)
            if clip:
                obj_id = id(clip)
                self._id_map.pop(obj_id, None)
                logger.debug(f"][[CLIPRegistry] Unregistered {instance_id}")

    def _get_instance(self, instance_id: str):
        """Get CLIP instance by ID (internal, host-side only)."""
        if IS_CHILD_PROCESS:
            raise RuntimeError(
                "]] [CLIPRegistry] FAIL-LOUD: "
                "_get_instance called in child process"
            )
        with self._lock:
            return self._registry.get(instance_id)

    # RPC methods below (async for pyisolate compatibility)

    async def get_ram_usage(self, instance_id: str) -> int:
        clip = self._get_instance(instance_id)
        if clip is None:
            raise ValueError(f"CLIP {instance_id} not found in registry")
        return clip.get_ram_usage()

    async def clip_layer(self, instance_id: str, layer_idx: int) -> None:
        clip = self._get_instance(instance_id)
        if clip is None:
            raise ValueError(f"CLIP {instance_id} not found in registry")
        clip.clip_layer(layer_idx)

    async def set_tokenizer_option(self, instance_id: str, option_name: str, value: Any) -> None:
        clip = self._get_instance(instance_id)
        if clip is None:
            raise ValueError(f"CLIP {instance_id} not found in registry")
        clip.set_tokenizer_option(option_name, value)

    async def tokenize(self, instance_id: str, text: str, return_word_ids: bool = False, **kwargs):
        clip = self._get_instance(instance_id)
        if clip is None:
            raise ValueError(f"CLIP {instance_id} not found in registry")
        return clip.tokenize(text, return_word_ids=return_word_ids, **kwargs)

    async def encode(self, instance_id: str, text: str):
        clip = self._get_instance(instance_id)
        if clip is None:
            raise ValueError(f"CLIP {instance_id} not found in registry")
        return _detach_if_grad(clip.encode(text))

    async def encode_from_tokens(self, instance_id: str, tokens, return_pooled: bool = False, return_dict: bool = False):
        clip = self._get_instance(instance_id)
        if clip is None:
            raise ValueError(f"CLIP {instance_id} not found in registry")
        return _detach_if_grad(
            clip.encode_from_tokens(tokens, return_pooled=return_pooled, return_dict=return_dict)
        )

    async def encode_from_tokens_scheduled(
        self,
        instance_id: str,
        tokens,
        unprojected: bool = False,
        add_dict: Optional[dict] = None,
        show_pbar: bool = True,
    ):
        clip = self._get_instance(instance_id)
        if clip is None:
            raise ValueError(f"CLIP {instance_id} not found in registry")
        add_dict = add_dict or {}
        return _detach_if_grad(
            clip.encode_from_tokens_scheduled(tokens, unprojected=unprojected, add_dict=add_dict, show_pbar=show_pbar)
        )

    async def add_patches(self, instance_id: str, patches, strength_patch: float = 1.0, strength_model: float = 1.0):
        clip = self._get_instance(instance_id)
        if clip is None:
            raise ValueError(f"CLIP {instance_id} not found in registry")
        return clip.add_patches(patches, strength_patch=strength_patch, strength_model=strength_model)

    async def get_key_patches(self, instance_id: str):
        clip = self._get_instance(instance_id)
        if clip is None:
            raise ValueError(f"CLIP {instance_id} not found in registry")
        return clip.get_key_patches()

    async def load_sd(self, instance_id: str, sd: dict, full_model: bool = False):
        clip = self._get_instance(instance_id)
        if clip is None:
            raise ValueError(f"CLIP {instance_id} not found in registry")
        return clip.load_sd(sd, full_model=full_model)

    async def get_sd(self, instance_id: str):
        clip = self._get_instance(instance_id)
        if clip is None:
            raise ValueError(f"CLIP {instance_id} not found in registry")
        return clip.get_sd()

    async def clone(self, instance_id: str) -> str:
        clip = self._get_instance(instance_id)
        if clip is None:
            raise ValueError(f"CLIP {instance_id} not found in registry")
        new_clip = clip.clone()
        return self.register(new_clip)


class CLIPProxy:
    """
    Picklable proxy for CLIP that forwards calls to host-side registry via RPC.

    Design:
    - Pickle-safe (stores only instance_id string)
    - Lazily acquires RPC caller on first method call
    - Mirrors CLIP interface for transparent substitution
    """

    __module__ = 'comfy.sd'  # Mimic real CLIP module for compatibility

    def __init__(self, instance_id: str, registry: Optional[CLIPRegistry] = None, manage_lifecycle: bool = False):
        """
        Initialize proxy with registry ID.

        Args:
            instance_id: Unique ID from CLIPRegistry.register()
        """
        self._instance_id = instance_id
        self._rpc_caller = None
        self._registry = registry if registry is not None else CLIPRegistry()
        self._manage_lifecycle = manage_lifecycle
        if manage_lifecycle and not IS_CHILD_PROCESS:
            self._finalizer = weakref.finalize(
                self, self._registry.unregister_sync, instance_id
            )
        logger.debug(f"][[CLIPProxy] Created for {instance_id}")

    def _get_rpc(self):
        """Lazy RPC caller acquisition."""
        if self._rpc_caller is None:
            from pyisolate._internal.shared import get_child_rpc_instance
            rpc = get_child_rpc_instance()
            if rpc is None:
                raise RuntimeError(
                    "]] [CLIPProxy] FAIL-LOUD: "
                    "No RPC instance available in child process"
                )
            self._rpc_caller = rpc.create_caller(CLIPRegistry, CLIPRegistry.get_remote_id())
        return self._rpc_caller

    def get_ram_usage(self) -> int:
        rpc = self._get_rpc()
        try:
            asyncio.get_running_loop()
            return _run_coro_in_new_loop(rpc.get_ram_usage(self._instance_id))
        except RuntimeError:
            loop = _get_thread_loop()
            return loop.run_until_complete(rpc.get_ram_usage(self._instance_id))

    def clip_layer(self, layer_idx: int) -> None:
        rpc = self._get_rpc()
        try:
            asyncio.get_running_loop()
            return _run_coro_in_new_loop(rpc.clip_layer(self._instance_id, layer_idx))
        except RuntimeError:
            loop = _get_thread_loop()
            return loop.run_until_complete(rpc.clip_layer(self._instance_id, layer_idx))

    def set_tokenizer_option(self, option_name: str, value: Any) -> None:
        rpc = self._get_rpc()
        try:
            asyncio.get_running_loop()
            return _run_coro_in_new_loop(rpc.set_tokenizer_option(self._instance_id, option_name, value))
        except RuntimeError:
            loop = _get_thread_loop()
            return loop.run_until_complete(rpc.set_tokenizer_option(self._instance_id, option_name, value))

    def tokenize(self, text: str, return_word_ids: bool = False, **kwargs):
        rpc = self._get_rpc()
        try:
            asyncio.get_running_loop()
            return _run_coro_in_new_loop(
                rpc.tokenize(self._instance_id, text, return_word_ids=return_word_ids, **kwargs)
            )
        except RuntimeError:
            loop = _get_thread_loop()
            return loop.run_until_complete(
                rpc.tokenize(self._instance_id, text, return_word_ids=return_word_ids, **kwargs)
            )

    def encode(self, text: str):
        rpc = self._get_rpc()
        try:
            asyncio.get_running_loop()
            return _run_coro_in_new_loop(rpc.encode(self._instance_id, text))
        except RuntimeError:
            loop = _get_thread_loop()
            return loop.run_until_complete(rpc.encode(self._instance_id, text))

    def encode_from_tokens(self, tokens, return_pooled: bool = False, return_dict: bool = False):
        rpc = self._get_rpc()
        try:
            asyncio.get_running_loop()
            return _run_coro_in_new_loop(
                rpc.encode_from_tokens(self._instance_id, tokens, return_pooled=return_pooled, return_dict=return_dict)
            )
        except RuntimeError:
            loop = _get_thread_loop()
            return loop.run_until_complete(
                rpc.encode_from_tokens(self._instance_id, tokens, return_pooled=return_pooled, return_dict=return_dict)
            )

    def encode_from_tokens_scheduled(
        self,
        tokens,
        unprojected: bool = False,
        add_dict: Optional[dict] = None,
        show_pbar: bool = True,
    ):
        rpc = self._get_rpc()
        add_dict = add_dict or {}
        try:
            asyncio.get_running_loop()
            return _run_coro_in_new_loop(
                rpc.encode_from_tokens_scheduled(
                    self._instance_id,
                    tokens,
                    unprojected=unprojected,
                    add_dict=add_dict,
                    show_pbar=show_pbar,
                )
            )
        except RuntimeError:
            loop = _get_thread_loop()
            return loop.run_until_complete(
                rpc.encode_from_tokens_scheduled(
                    self._instance_id,
                    tokens,
                    unprojected=unprojected,
                    add_dict=add_dict,
                    show_pbar=show_pbar,
                )
            )

    def add_patches(self, patches, strength_patch: float = 1.0, strength_model: float = 1.0):
        rpc = self._get_rpc()
        try:
            asyncio.get_running_loop()
            return _run_coro_in_new_loop(
                rpc.add_patches(self._instance_id, patches, strength_patch=strength_patch, strength_model=strength_model)
            )
        except RuntimeError:
            loop = _get_thread_loop()
            return loop.run_until_complete(
                rpc.add_patches(self._instance_id, patches, strength_patch=strength_patch, strength_model=strength_model)
            )

    def get_key_patches(self):
        rpc = self._get_rpc()
        try:
            asyncio.get_running_loop()
            return _run_coro_in_new_loop(rpc.get_key_patches(self._instance_id))
        except RuntimeError:
            loop = _get_thread_loop()
            return loop.run_until_complete(rpc.get_key_patches(self._instance_id))

    def load_sd(self, sd: dict, full_model: bool = False):
        rpc = self._get_rpc()
        try:
            asyncio.get_running_loop()
            return _run_coro_in_new_loop(rpc.load_sd(self._instance_id, sd, full_model=full_model))
        except RuntimeError:
            loop = _get_thread_loop()
            return loop.run_until_complete(rpc.load_sd(self._instance_id, sd, full_model=full_model))

    def get_sd(self):
        rpc = self._get_rpc()
        try:
            asyncio.get_running_loop()
            return _run_coro_in_new_loop(rpc.get_sd(self._instance_id))
        except RuntimeError:
            loop = _get_thread_loop()
            return loop.run_until_complete(rpc.get_sd(self._instance_id))

    def clone(self) -> 'CLIPProxy':
        rpc = self._get_rpc()
        try:
            asyncio.get_running_loop()
            new_id = _run_coro_in_new_loop(rpc.clone(self._instance_id))
        except RuntimeError:
            loop = _get_thread_loop()
            new_id = loop.run_until_complete(rpc.clone(self._instance_id))
        return CLIPProxy(new_id, self._registry, manage_lifecycle=not IS_CHILD_PROCESS)

    def __getstate__(self):
        """Pickle support: only serialize instance_id."""
        return {"_instance_id": self._instance_id}

    def __setstate__(self, state):
        """Unpickle support: restore instance_id, reset RPC caller."""
        self._instance_id = state["_instance_id"]
        self._rpc_caller = None
        self._registry = CLIPRegistry()
        self._manage_lifecycle = False
        logger.debug(f"][[CLIPProxy] Restored from pickle: {self._instance_id}")

    def __repr__(self):
        return f"<CLIPProxy {self._instance_id}>"


# Registry instantiated in host_hooks.initialize_host_process; keep optional safety
if not IS_CHILD_PROCESS:
    _CLIP_REGISTRY_SINGLETON = CLIPRegistry()
