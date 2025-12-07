"""
Dedicated sync-to-async bridge for PyIsolate RPC calls.

This module provides a thread-safe mechanism to execute async RPC calls
from synchronous contexts without causing nested event loop errors.

The RpcBridge singleton runs a persistent asyncio event loop in a dedicated
background thread. Synchronous code dispatches coroutines to this loop
using asyncio.run_coroutine_threadsafe() and blocks waiting for the result.

This pattern is required because:
1. ModelSampling API is synchronous (timestep(), sigma(), etc.)
2. PyIsolate RPC is asynchronous
3. loop.run_until_complete() causes "This event loop is already running" errors
"""

import asyncio
import atexit
import logging
import threading
from typing import Any, Coroutine, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class RpcBridge:
    """
    Singleton bridge for executing async RPC calls from synchronous code.
    
    Maintains a dedicated background thread running an asyncio event loop.
    Provides run_sync() to execute coroutines and block until completion.
    
    Thread-safe singleton pattern using __new__ and a lock.
    
    Usage:
        bridge = RpcBridge()
        result = bridge.run_sync(some_async_function(arg1, arg2))
    """
    
    _instance: Optional["RpcBridge"] = None
    _lock = threading.Lock()
    
    def __new__(cls) -> "RpcBridge":
        """Thread-safe singleton instantiation."""
        with cls._lock:
            if cls._instance is None:
                instance = super().__new__(cls)
                instance._initialize()
                cls._instance = instance
                logger.debug("][[RpcBridge] Singleton created")
        return cls._instance
    
    def _initialize(self) -> None:
        """Initialize the background event loop thread."""
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._started = threading.Event()
        self._shutdown = False
        
        # Create and start background thread
        self._thread = threading.Thread(
            target=self._run_loop,
            name="PyIsolate-RpcBridge",
            daemon=True,  # Don't block process exit
        )
        self._thread.start()
        
        # Wait for loop to be ready (with timeout to detect startup failures)
        if not self._started.wait(timeout=5.0):
            raise RuntimeError(
                "][[RpcBridge] FAIL-LOUD: Event loop thread failed to start"
            )
        
        # Register shutdown handler
        atexit.register(self._shutdown_loop)
        
        logger.debug("][[RpcBridge] Background thread started")
    
    def _run_loop(self) -> None:
        """Background thread entry point - runs the asyncio event loop."""
        try:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            
            # Signal that loop is ready
            self._started.set()
            
            # Run forever until shutdown
            self._loop.run_forever()
        except Exception as e:
            logger.error("][[RpcBridge] Loop error: %s", e)
            self._started.set()  # Unblock waiters even on failure
        finally:
            if self._loop is not None:
                self._loop.close()
            logger.debug("][[RpcBridge] Background thread exited")
    
    def _shutdown_loop(self) -> None:
        """Clean shutdown of the event loop (called at process exit)."""
        if self._shutdown:
            return
        self._shutdown = True
        
        if self._loop is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        
        logger.debug("][[RpcBridge] Shutdown complete")
    
    def run_sync(self, coro: Coroutine[Any, Any, T], timeout: Optional[float] = None) -> T:
        """
        Execute an async coroutine synchronously and return its result.
        
        Dispatches the coroutine to the background event loop thread and
        blocks until it completes. Propagates exceptions from the coroutine.
        
        Args:
            coro: The coroutine to execute
            timeout: Optional timeout in seconds (None = no timeout)
        
        Returns:
            The result of the coroutine
        
        Raises:
            RuntimeError: If the bridge is not initialized or shutting down
            TimeoutError: If timeout is exceeded
            Exception: Any exception raised by the coroutine
        """
        if self._loop is None or not self._loop.is_running():
            raise RuntimeError(
                "][[RpcBridge] FAIL-LOUD: Event loop not running"
            )
        
        if self._shutdown:
            raise RuntimeError(
                "][[RpcBridge] FAIL-LOUD: Bridge is shutting down"
            )
        
        # Submit coroutine to the background loop
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        
        try:
            # Block until result is available (with optional timeout)
            return future.result(timeout=timeout)
        except TimeoutError:
            future.cancel()
            raise TimeoutError(
                f"][[RpcBridge] RPC call timed out after {timeout}s"
            )
        except Exception as e:
            # Propagate the original exception with full traceback
            raise RuntimeError(f"][[RpcBridge] RPC call failed: {e}") from e
    
    @property
    def is_running(self) -> bool:
        """Check if the background loop is running."""
        return self._loop is not None and self._loop.is_running() and not self._shutdown


# Module-level singleton accessor for convenience
_bridge: Optional[RpcBridge] = None


def get_rpc_bridge() -> RpcBridge:
    """
    Get the global RpcBridge singleton.
    
    Lazily initializes the bridge on first call.
    Thread-safe.
    
    Returns:
        The RpcBridge singleton instance
    """
    global _bridge
    if _bridge is None:
        _bridge = RpcBridge()
    return _bridge
