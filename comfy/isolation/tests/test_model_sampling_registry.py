"""
Unit tests for ModelSamplingRegistry and ModelSamplingProxy.

Tests:
1. Registry register/unregister_sync
2. Thread safety with concurrent access
3. RPC methods return correct values
4. weakref.finalize cleanup
5. Proxy pickle serialization
"""

import asyncio
import gc
import pickle
import pytest
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, AsyncMock

import sys
import os

# Add ComfyUI to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from comfy.isolation.model_sampling_proxy import (
    ModelSamplingRegistry, 
    ModelSamplingProxy,
    _reconstruct_model_sampling_proxy,
)


class MockModelSampling:
    """Mock ModelSampling object for testing."""
    
    def __init__(self, sigma_min=0.001, sigma_max=1000.0, sigma_data=1.0):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.sigmas = [0.1, 0.5, 1.0]
        self.log_sigmas = [-2.3, -0.7, 0.0]
    
    def calculate_input(self, sigma, noise):
        return noise * sigma
    
    def calculate_denoised(self, sigma, model_output, model_input):
        return model_output - sigma * model_input
    
    def noise_scaling(self, sigma, noise, latent_image, max_denoise=False):
        return noise * sigma + latent_image
    
    def timestep(self, sigma):
        return sigma * 1000
    
    def sigma(self, timestep):
        return timestep / 1000
    
    def percent_to_sigma(self, percent):
        return (1.0 - percent) * self.sigma_max


class TestModelSamplingRegistryBasic:
    """Test basic registry operations."""
    
    def setup_method(self):
        """Reset registry state before each test."""
        # Create fresh registry (not singleton in tests)
        self.registry = ModelSamplingRegistry.__new__(ModelSamplingRegistry)
        self.registry._registry = {}
        self.registry._counter = 0
        self.registry._lock = threading.Lock()
    
    def test_register_returns_unique_id(self):
        """Each registration returns a unique ID."""
        mock1 = MockModelSampling()
        mock2 = MockModelSampling()
        
        id1 = self.registry.register(mock1)
        id2 = self.registry.register(mock2)
        
        assert id1 != id2
        assert id1 == "model_sampling_0"
        assert id2 == "model_sampling_1"
    
    def test_register_stores_instance(self):
        """Registered instance is stored."""
        mock = MockModelSampling()
        instance_id = self.registry.register(mock)
        
        assert instance_id in self.registry._registry
        assert self.registry._registry[instance_id] is mock
    
    def test_unregister_sync_removes_instance(self):
        """unregister_sync removes instance from registry."""
        mock = MockModelSampling()
        instance_id = self.registry.register(mock)
        
        assert instance_id in self.registry._registry
        
        self.registry.unregister_sync(instance_id)
        
        assert instance_id not in self.registry._registry
    
    def test_unregister_sync_nonexistent_noop(self):
        """unregister_sync on nonexistent ID is safe."""
        # Should not raise
        self.registry.unregister_sync("nonexistent_id")
    
    def test_get_instance_returns_correct_instance(self):
        """_get_instance returns the correct instance."""
        mock = MockModelSampling(sigma_min=42)
        instance_id = self.registry.register(mock)
        
        retrieved = self.registry._get_instance(instance_id)
        
        assert retrieved is mock
        assert retrieved.sigma_min == 42
    
    def test_get_instance_raises_for_unknown(self):
        """_get_instance raises ValueError for unknown ID."""
        with pytest.raises(ValueError) as excinfo:
            self.registry._get_instance("unknown_id")
        
        assert "Unknown model_sampling instance" in str(excinfo.value)


class TestModelSamplingRegistryThreadSafety:
    """Test thread safety of registry operations."""
    
    def setup_method(self):
        """Reset registry state before each test."""
        self.registry = ModelSamplingRegistry.__new__(ModelSamplingRegistry)
        self.registry._registry = {}
        self.registry._counter = 0
        self.registry._lock = threading.Lock()
    
    def test_concurrent_registration(self):
        """Concurrent registrations produce unique IDs."""
        results = []
        lock = threading.Lock()
        
        def register_mock():
            mock = MockModelSampling()
            instance_id = self.registry.register(mock)
            with lock:
                results.append(instance_id)
        
        threads = [threading.Thread(target=register_mock) for _ in range(100)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # All IDs should be unique
        assert len(results) == 100
        assert len(set(results)) == 100
    
    def test_concurrent_register_unregister(self):
        """Concurrent register and unregister don't corrupt state."""
        registered_ids = []
        lock = threading.Lock()
        
        def register_and_unregister():
            mock = MockModelSampling()
            instance_id = self.registry.register(mock)
            with lock:
                registered_ids.append(instance_id)
            # Immediately unregister
            self.registry.unregister_sync(instance_id)
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            list(executor.map(lambda _: register_and_unregister(), range(50)))
        
        # All should be unregistered
        assert len(self.registry._registry) == 0


class TestModelSamplingRegistryRpcMethods:
    """Test RPC methods of registry."""
    
    def setup_method(self):
        """Reset registry state and create mock instance."""
        self.registry = ModelSamplingRegistry.__new__(ModelSamplingRegistry)
        self.registry._registry = {}
        self.registry._counter = 0
        self.registry._lock = threading.Lock()
        
        self.mock = MockModelSampling(sigma_min=0.01, sigma_max=100.0)
        self.instance_id = self.registry.register(self.mock)
    
    @pytest.mark.asyncio
    async def test_percent_to_sigma(self):
        """percent_to_sigma forwards correctly."""
        result = await self.registry.percent_to_sigma(self.instance_id, 0.5)
        expected = self.mock.percent_to_sigma(0.5)
        assert result == expected
    
    @pytest.mark.asyncio
    async def test_timestep(self):
        """timestep forwards correctly."""
        result = await self.registry.timestep(self.instance_id, 0.5)
        expected = self.mock.timestep(0.5)
        assert result == expected
    
    @pytest.mark.asyncio
    async def test_sigma(self):
        """sigma forwards correctly."""
        result = await self.registry.sigma(self.instance_id, 500)
        expected = self.mock.sigma(500)
        assert result == expected
    
    @pytest.mark.asyncio
    async def test_get_sigma_min(self):
        """get_sigma_min returns correct value."""
        result = await self.registry.get_sigma_min(self.instance_id)
        assert result == 0.01
    
    @pytest.mark.asyncio
    async def test_get_sigma_max(self):
        """get_sigma_max returns correct value."""
        result = await self.registry.get_sigma_max(self.instance_id)
        assert result == 100.0
    
    @pytest.mark.asyncio
    async def test_unknown_instance_raises(self):
        """RPC methods raise for unknown instance."""
        with pytest.raises(ValueError):
            await self.registry.percent_to_sigma("unknown", 0.5)


class TestModelSamplingProxyPickle:
    """Test pickle serialization of proxy."""
    
    def setup_method(self):
        """Create registry and proxy."""
        self.registry = ModelSamplingRegistry.__new__(ModelSamplingRegistry)
        self.registry._registry = {}
        self.registry._counter = 0
        self.registry._lock = threading.Lock()
        
        self.mock = MockModelSampling()
        self.instance_id = self.registry.register(self.mock)
    
    def test_proxy_reduce_returns_id_only(self):
        """__reduce__ returns reconstructor with just ID."""
        proxy = ModelSamplingProxy(self.instance_id, self.registry)
        
        reduced = proxy.__reduce__()
        
        assert reduced[0] == _reconstruct_model_sampling_proxy
        assert reduced[1] == (self.instance_id,)
    
    def test_pickle_serialization_size(self):
        """Pickled proxy is small (just ID string)."""
        proxy = ModelSamplingProxy(self.instance_id, self.registry)
        
        pickled = pickle.dumps(proxy)
        
        # Should be small - just function ref + ID string
        # "model_sampling_0" is ~16 bytes + pickle overhead
        assert len(pickled) < 200  # Conservative upper bound
    
    def test_pickle_roundtrip_preserves_id(self):
        """Pickle/unpickle preserves instance ID."""
        proxy = ModelSamplingProxy(self.instance_id, self.registry)
        
        pickled = pickle.dumps(proxy)
        restored = pickle.loads(pickled)
        
        assert restored._instance_id == self.instance_id


class TestModelSamplingProxyLifecycle:
    """Test lifecycle management with weakref.finalize."""
    
    def setup_method(self):
        """Create registry."""
        self.registry = ModelSamplingRegistry.__new__(ModelSamplingRegistry)
        self.registry._registry = {}
        self.registry._counter = 0
        self.registry._lock = threading.Lock()
    
    def test_manage_lifecycle_attaches_finalizer(self):
        """manage_lifecycle=True attaches finalizer."""
        mock = MockModelSampling()
        instance_id = self.registry.register(mock)
        
        # Need to patch IS_CHILD_PROCESS for this test
        import comfy.isolation.model_sampling_proxy as msp
        original = msp.IS_CHILD_PROCESS
        msp.IS_CHILD_PROCESS = False
        
        try:
            proxy = ModelSamplingProxy(instance_id, self.registry, manage_lifecycle=True)
            
            # Should have finalizer attached
            assert hasattr(proxy, '_finalizer')
        finally:
            msp.IS_CHILD_PROCESS = original
    
    def test_finalizer_cleans_up_on_gc(self):
        """Finalizer cleans up registry on proxy GC."""
        mock = MockModelSampling()
        instance_id = self.registry.register(mock)
        
        import comfy.isolation.model_sampling_proxy as msp
        original = msp.IS_CHILD_PROCESS
        msp.IS_CHILD_PROCESS = False
        
        try:
            # Create proxy with lifecycle management
            proxy = ModelSamplingProxy(instance_id, self.registry, manage_lifecycle=True)
            
            # Instance should be registered
            assert instance_id in self.registry._registry
            
            # Delete proxy and force GC
            del proxy
            gc.collect()
            
            # Instance should be unregistered
            assert instance_id not in self.registry._registry
        finally:
            msp.IS_CHILD_PROCESS = original
    
    def test_child_proxy_no_cleanup(self):
        """Proxy with manage_lifecycle=False doesn't clean up."""
        mock = MockModelSampling()
        instance_id = self.registry.register(mock)
        
        # Create proxy WITHOUT lifecycle management (like child process would)
        proxy = ModelSamplingProxy(instance_id, self.registry, manage_lifecycle=False)
        
        # Instance should be registered
        assert instance_id in self.registry._registry
        
        # Delete proxy and force GC
        del proxy
        gc.collect()
        
        # Instance should STILL be registered (no cleanup from child proxy)
        assert instance_id in self.registry._registry


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
