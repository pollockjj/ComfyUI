from __future__ import annotations

import asyncio
import logging
import os
import threading
import weakref
from typing import Any, Callable, Dict, Generic, Optional, TypeVar

try:
    from pyisolate import ProxiedSingleton
except ImportError:
    class ProxiedSingleton:  # type: ignore[no-redef]
        pass

logger = logging.getLogger(__name__)

IS_CHILD_PROCESS = os.environ.get("PYISOLATE_CHILD") == "1"
_thread_local = threading.local()
T = TypeVar("T")


def get_thread_loop() -> asyncio.AbstractEventLoop:
    loop = getattr(_thread_local, "loop", None)
    if loop is None or loop.is_closed():
        loop = asyncio.new_event_loop()
        _thread_local.loop = loop
    return loop


def run_coro_in_new_loop(coro: Any) -> Any:
    result_box: Dict[str, Any] = {}
    exc_box: Dict[str, BaseException] = {}

    def runner() -> None:
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


def detach_if_grad(obj: Any) -> Any:
    try:
        import torch
    except Exception:
        return obj

    if isinstance(obj, torch.Tensor):
        return obj.detach() if obj.requires_grad else obj
    if isinstance(obj, (list, tuple)):
        return type(obj)(detach_if_grad(x) for x in obj)
    if isinstance(obj, dict):
        return {k: detach_if_grad(v) for k, v in obj.items()}
    return obj


class BaseRegistry(ProxiedSingleton, Generic[T]):
    _type_prefix: str = "base"

    def __init__(self) -> None:
        if hasattr(ProxiedSingleton, "__init__") and ProxiedSingleton is not object:
            super().__init__()
        self._registry: Dict[str, T] = {}
        self._id_map: Dict[int, str] = {}
        self._refcounts: Dict[str, int] = {}
        self._counter = 0
        self._lock = threading.Lock()

    def register(self, instance: T) -> str:
        """Register an object and return its ID. Refcount starts at 0.

        Refcount tracks ONLY Child proxy references. When all Child proxies die
        (via release()), refcount hits 0 and the entry is removed, allowing GC.
        The Host can still access the object while it's registered.
        """
        import logging
        with self._lock:
            obj_id = id(instance)
            if obj_id in self._id_map:
                instance_id = self._id_map[obj_id]
                # Already registered - don't change refcount, just return ID
                logging.info(f"[Registry] register: {instance_id} EXISTING refcount={self._refcounts.get(instance_id, 0)}")
                return instance_id
            instance_id = f"{self._type_prefix}_{self._counter}"
            self._counter += 1
            self._registry[instance_id] = instance
            self._id_map[obj_id] = instance_id
            self._refcounts[instance_id] = 0  # Start at 0 - only Child proxies add refcount
            # Attach instance_id to the real object for later lookup during unload
            try:
                instance._proxy_instance_id = instance_id
            except (AttributeError, TypeError):
                pass  # Object doesn't support attribute assignment
            logging.info(f"[Registry] register: {instance_id} NEW refcount=0")
        return instance_id

    def acquire(self, instance_id: str) -> bool:
        """Increment refcount for an existing ID. Returns True if successful."""
        import logging
        with self._lock:
            if instance_id not in self._registry:
                logging.info(f"[Registry] acquire: {instance_id} not found")
                return False
            self._refcounts[instance_id] = self._refcounts.get(instance_id, 1) + 1
            logging.info(f"[Registry] acquire: {instance_id} refcount++ -> {self._refcounts[instance_id]}")
            return True

    def release(self, instance_id: str) -> None:
        """Decrement refcount, unregister when zero."""
        import gc
        import logging
        with self._lock:
            if instance_id not in self._refcounts:
                logging.info(f"[Registry] release: {instance_id} not found (already released)")
                return
            self._refcounts[instance_id] -= 1
            new_count = self._refcounts[instance_id]
            logging.info(f"[Registry] release: {instance_id} refcount-- -> {new_count}")
            if new_count <= 0:
                del self._refcounts[instance_id]
                instance = self._registry.pop(instance_id, None)
                if instance:
                    self._id_map.pop(id(instance), None)
                logging.info(f"[Registry] release: {instance_id} REMOVED from registry (refcount=0)")
        # gc.collect outside lock to allow weakrefs in model_management to be processed
        if new_count <= 0:
            gc.collect()

    def unregister_sync(self, instance_id: str) -> None:
        """Legacy method - now calls release()."""
        self.release(instance_id)

    def clear_all(self) -> None:
        """Clear all registry entries. Called at workflow end to release all references."""
        import gc
        import logging
        with self._lock:
            count = len(self._registry)
            self._registry.clear()
            self._id_map.clear()
            self._refcounts.clear()
            logging.info(f"[Registry] clear_all: cleared {count} entries")
        gc.collect()

    async def acquire_async(self, instance_id: str) -> bool:
        """Async version of acquire for RPC calls from Child."""
        return self.acquire(instance_id)

    async def release_async(self, instance_id: str) -> None:
        """Async version of release for RPC calls from Child."""
        self.release(instance_id)

    def _get_instance(self, instance_id: str) -> T:
        if IS_CHILD_PROCESS:
            raise RuntimeError(f"[{self.__class__.__name__}] _get_instance called in child")
        with self._lock:
            instance = self._registry.get(instance_id)
        if instance is None:
            raise ValueError(f"{instance_id} not found")
        return instance


_GLOBAL_LOOP: Optional[asyncio.AbstractEventLoop] = None

def set_global_loop(loop: asyncio.AbstractEventLoop) -> None:
    global _GLOBAL_LOOP
    _GLOBAL_LOOP = loop

class BaseProxy(Generic[T]):
    _registry_class: type = BaseRegistry  # type: ignore[type-arg]
    __module__: str = "comfy.isolation.proxies.base"

    def __init__(self, instance_id: str, registry: Optional[Any] = None) -> None:
        import logging
        self._instance_id = instance_id
        self._rpc_caller: Optional[Any] = None
        self._registry = registry if registry is not None else self._registry_class()
        logging.info(f"[BaseProxy.__init__] {self.__class__.__name__}({instance_id}) IS_CHILD={IS_CHILD_PROCESS}")
        # On Host: acquire refcount and attach finalizer for release
        # On Child: don't acquire (Host owns the object), finalizer will release via RPC
        if not IS_CHILD_PROCESS:
            self._registry.acquire(instance_id)
            self._finalizer = weakref.finalize(self, self._registry.release, instance_id)

    def _get_rpc(self) -> Any:
        if self._rpc_caller is None:
            from pyisolate._internal.rpc_protocol import get_child_rpc_instance
            rpc = get_child_rpc_instance()
            if rpc is None:
                raise RuntimeError(f"[{self.__class__.__name__}] No RPC in child")
            self._rpc_caller = rpc.create_caller(self._registry_class, self._registry_class.get_remote_id())
        return self._rpc_caller

    def _call_rpc(self, method_name: str, *args: Any, **kwargs: Any) -> Any:
        rpc = self._get_rpc()
        method = getattr(rpc, method_name)
        coro = method(self._instance_id, *args, **kwargs)

        # If we have a global loop (Main Thread Loop), use it for dispatch from worker threads
        if _GLOBAL_LOOP is not None and _GLOBAL_LOOP.is_running():
            try:
                # If we are already in the global loop, we can't block on it?
                # Actually, this method is synchronous (__getattr__ -> lambda).
                # If called from async context in main loop, we need to handle that.
                curr_loop = asyncio.get_running_loop()
                if curr_loop is _GLOBAL_LOOP:
                     # We are in the main loop. We cannot await/block here if we are just a sync function.
                     # But proxies are often called from sync code.
                     # If called from sync code in main loop, creating a new loop is bad.
                     # But we can't await `coro`.
                     # This implies proxies MUST be awaited if called from async context?
                     # Existing code used `run_coro_in_new_loop` which is weird.
                     # Let's trust that if we are in a thread (RuntimeError on get_running_loop),
                     # we use run_coroutine_threadsafe.
                     pass
            except RuntimeError:
                # No running loop - we are in a worker thread.
                future = asyncio.run_coroutine_threadsafe(coro, _GLOBAL_LOOP)
                return future.result()

        try:
            asyncio.get_running_loop()
            return run_coro_in_new_loop(coro)
        except RuntimeError:
            loop = get_thread_loop()
            return loop.run_until_complete(coro)

    def __getstate__(self) -> Dict[str, Any]:
        return {"_instance_id": self._instance_id}

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self._instance_id = state["_instance_id"]
        self._rpc_caller = None
        self._registry = self._registry_class()
        # Child proxies need to acquire refcount on Host and release when done
        if IS_CHILD_PROCESS:
            _child_acquire_callback(self._registry_class, state["_instance_id"])
            self._finalizer = weakref.finalize(self, _child_release_callback, self._registry_class, state["_instance_id"])

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} {self._instance_id}>"


def _child_acquire_callback(registry_class: type, instance_id: str) -> None:
    """Callback for Child proxy deserialization to acquire refcount on Host."""
    try:
        from pyisolate._internal.rpc_protocol import get_child_rpc_instance
        rpc = get_child_rpc_instance()
        if rpc is None:
            return  # RPC not available
        caller = rpc.create_caller(registry_class, registry_class.get_remote_id())
        coro = caller.acquire_async(instance_id)
        loop = get_thread_loop()
        loop.run_until_complete(coro)
        logger.debug(f"[ChildAcquire] Acquired {instance_id} on Host")
    except Exception as e:
        logger.debug(f"[ChildAcquire] Failed to acquire {instance_id}: {e}")


def _child_release_callback(registry_class: type, instance_id: str) -> None:
    """Callback for Child proxy finalizers to release Host registry entry."""
    try:
        from pyisolate._internal.rpc_protocol import get_child_rpc_instance
        rpc = get_child_rpc_instance()
        if rpc is None:
            return  # RPC not available, can't release
        caller = rpc.create_caller(registry_class, registry_class.get_remote_id())
        # Call release_async via RPC - this decrements Host refcount
        coro = caller.release_async(instance_id)
        loop = get_thread_loop()
        loop.run_until_complete(coro)
        logger.debug(f"[ChildRelease] Released {instance_id} on Host")
    except Exception as e:
        # Swallow errors during shutdown/cleanup
        logger.debug(f"[ChildRelease] Failed to release {instance_id}: {e}")


def create_rpc_method(method_name: str) -> Callable[..., Any]:
    def method(self: BaseProxy[Any], *args: Any, **kwargs: Any) -> Any:
        return self._call_rpc(method_name, *args, **kwargs)
    method.__name__ = method_name
    return method
