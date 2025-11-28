"""
Unit tests for RpcBridge.

Tests:
1. Singleton pattern (thread-safe)
2. run_sync() with simple async coroutines
3. Error propagation
4. No nested loop errors
5. Timeout handling
"""

import asyncio
import pytest
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import sys
import os

# Add ComfyUI to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from comfy.isolation.rpc_bridge import RpcBridge, get_rpc_bridge


class TestRpcBridgeSingleton:
    """Test singleton pattern."""
    
    def test_singleton_same_instance(self):
        """Multiple calls return same instance."""
        bridge1 = RpcBridge()
        bridge2 = RpcBridge()
        assert bridge1 is bridge2
    
    def test_singleton_thread_safe(self):
        """Concurrent instantiation returns same instance."""
        instances = []
        
        def get_instance():
            instances.append(RpcBridge())
        
        threads = [threading.Thread(target=get_instance) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # All instances should be the same object
        assert all(inst is instances[0] for inst in instances)
    
    def test_get_rpc_bridge_convenience(self):
        """get_rpc_bridge() returns singleton."""
        bridge = get_rpc_bridge()
        assert bridge is RpcBridge()


class TestRpcBridgeRunSync:
    """Test run_sync() method."""
    
    def test_simple_coroutine(self):
        """Run a simple async function."""
        async def add(a, b):
            return a + b
        
        bridge = get_rpc_bridge()
        result = bridge.run_sync(add(2, 3))
        assert result == 5
    
    def test_coroutine_with_delay(self):
        """Run coroutine with async sleep."""
        async def delayed_return(value, delay):
            await asyncio.sleep(delay)
            return value
        
        bridge = get_rpc_bridge()
        start = time.time()
        result = bridge.run_sync(delayed_return("hello", 0.1))
        elapsed = time.time() - start
        
        assert result == "hello"
        assert elapsed >= 0.1
    
    def test_coroutine_returning_none(self):
        """Coroutine returning None."""
        async def return_none():
            return None
        
        bridge = get_rpc_bridge()
        result = bridge.run_sync(return_none())
        assert result is None
    
    def test_multiple_sequential_calls(self):
        """Multiple sequential run_sync calls."""
        async def increment(n):
            return n + 1
        
        bridge = get_rpc_bridge()
        
        result = 0
        for _ in range(10):
            result = bridge.run_sync(increment(result))
        
        assert result == 10
    
    def test_concurrent_calls_from_threads(self):
        """Multiple threads calling run_sync concurrently."""
        async def slow_square(n):
            await asyncio.sleep(0.01)
            return n * n
        
        bridge = get_rpc_bridge()
        results = []
        lock = threading.Lock()
        
        def compute(n):
            result = bridge.run_sync(slow_square(n))
            with lock:
                results.append((n, result))
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            executor.map(compute, range(10))
        
        # Verify all results are correct
        assert len(results) == 10
        for n, result in results:
            assert result == n * n


class TestRpcBridgeErrorPropagation:
    """Test error handling."""
    
    def test_exception_propagation(self):
        """Exceptions in coroutine are propagated."""
        async def raise_error():
            raise ValueError("test error")
        
        bridge = get_rpc_bridge()
        
        with pytest.raises(RuntimeError) as excinfo:
            bridge.run_sync(raise_error())
        
        assert "RPC call failed" in str(excinfo.value)
        assert "test error" in str(excinfo.value)
    
    def test_custom_exception_type_preserved_in_cause(self):
        """Original exception type is in __cause__."""
        class CustomError(Exception):
            pass
        
        async def raise_custom():
            raise CustomError("custom")
        
        bridge = get_rpc_bridge()
        
        with pytest.raises(RuntimeError) as excinfo:
            bridge.run_sync(raise_custom())
        
        assert excinfo.value.__cause__ is not None
        assert isinstance(excinfo.value.__cause__, CustomError)


class TestRpcBridgeTimeout:
    """Test timeout handling."""
    
    def test_timeout_raises_error(self):
        """Timeout raises TimeoutError."""
        async def slow_operation():
            await asyncio.sleep(10)
            return "done"
        
        bridge = get_rpc_bridge()
        
        with pytest.raises(TimeoutError) as excinfo:
            bridge.run_sync(slow_operation(), timeout=0.1)
        
        assert "timed out" in str(excinfo.value)
    
    def test_no_timeout_by_default(self):
        """Without timeout, waits indefinitely (test with short delay)."""
        async def quick_operation():
            await asyncio.sleep(0.05)
            return "done"
        
        bridge = get_rpc_bridge()
        result = bridge.run_sync(quick_operation())  # No timeout
        assert result == "done"


class TestRpcBridgeNoNestedLoopError:
    """Test that we don't get nested event loop errors."""
    
    def test_can_call_from_sync_context(self):
        """Call from regular synchronous code (no existing event loop)."""
        async def simple():
            return 42
        
        bridge = get_rpc_bridge()
        # This is the main test - should not raise "This event loop is already running"
        result = bridge.run_sync(simple())
        assert result == 42
    
    def test_can_call_from_thread_without_loop(self):
        """Call from a thread that has no event loop."""
        result_holder = []
        
        async def get_value():
            return "from thread"
        
        def thread_func():
            bridge = get_rpc_bridge()
            result = bridge.run_sync(get_value())
            result_holder.append(result)
        
        t = threading.Thread(target=thread_func)
        t.start()
        t.join()
        
        assert result_holder == ["from thread"]


class TestRpcBridgeState:
    """Test bridge state management."""
    
    def test_is_running_property(self):
        """is_running reflects loop state."""
        bridge = get_rpc_bridge()
        assert bridge.is_running is True
    
    def test_loop_persists_across_calls(self):
        """Same loop used for all calls."""
        bridge = get_rpc_bridge()
        
        async def get_loop_id():
            return id(asyncio.get_running_loop())
        
        id1 = bridge.run_sync(get_loop_id())
        id2 = bridge.run_sync(get_loop_id())
        
        assert id1 == id2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
