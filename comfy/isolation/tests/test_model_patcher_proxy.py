"""
Unit tests for model_patcher_proxy.py (ModelPatcherRegistry + ModelPatcherProxy).

Tests cover:
1. Registry CRUD operations with identity preservation (Finding 3)
2. Pickle round-trip (<500 bytes)
3. Clone lifecycle management
4. model_options getter/setter (Lazy Patching - Finding 2)
5. Property guards raise AttributeError

Research Reference: MODELPATCHER_PHASE1_PLAN.md
"""

import asyncio
import os
import pickle
import pytest
from unittest.mock import Mock, MagicMock

# Set environment to simulate host process
os.environ["PYISOLATE_CHILD"] = "0"
os.environ.pop("PYISOLATE_ISOLATION_ACTIVE", None)

from comfy.isolation.model_patcher_proxy import (
    ModelPatcherRegistry,
    ModelPatcherProxy,
    _reconstruct_model_patcher_proxy,
    maybe_wrap_model_for_isolation
)


class MockModelPatcher:
    """
    Mock ModelPatcher object for testing.
    
    Mirrors the key interface needed by PuLID:
    - clone()
    - get_model_object()
    - model_options property
    - load_device / offload_device properties
    - size property
    """
    
    def __init__(self, name="mock_model"):
        self.name = name
        self._model_options = {"transformer_options": {}}
        self._load_device = "cuda:0"
        self._offload_device = "cpu"
        self._size = 1024 * 1024 * 100  # 100MB
        self._cloned = False
        self._model_sampling = MockModelSampling()
    
    def clone(self):
        """Create a clone (as ComfyUI does)."""
        cloned = MockModelPatcher(f"{self.name}_clone")
        cloned._model_options = dict(self._model_options)
        cloned._cloned = True
        return cloned
    
    def get_model_object(self, name):
        """Get model sub-object (e.g., model_sampling)."""
        if name == "model_sampling":
            return self._model_sampling
        raise ValueError(f"Unknown model object: {name}")
    
    @property
    def model_options(self):
        return self._model_options
    
    @model_options.setter
    def model_options(self, value):
        self._model_options = value
    
    @property
    def load_device(self):
        return self._load_device
    
    @property
    def offload_device(self):
        return self._offload_device
    
    @property
    def size(self):
        return self._size


class MockModelSampling:
    """Mock model_sampling object returned by get_model_object."""
    
    def percent_to_sigma(self, percent):
        return percent * 100


# =============================================================================
# Registry Tests
# =============================================================================

def test_registry_register_unregister():
    """Test registry registration and cleanup."""
    registry = ModelPatcherRegistry()
    
    mock_model = MockModelPatcher()
    model_id = registry.register(mock_model)
    
    # Verify ID format
    assert model_id.startswith("model_")
    
    # Verify it's in registry
    retrieved = registry._get_instance(model_id)
    assert retrieved is mock_model
    
    # Unregister
    registry.unregister_sync(model_id)
    
    # Verify it's gone
    with pytest.raises(ValueError, match="not found in registry"):
        registry._get_instance(model_id)


def test_registry_identity_preservation():
    """
    Test that registering same object twice returns same ID.
    
    Research: Shadow-Reference Protocol Finding 3 (Identity-Based Logic)
    This is critical for is_clone() checks in ComfyUI.
    """
    registry = ModelPatcherRegistry()
    
    mock_model = MockModelPatcher()
    
    # Register once
    id1 = registry.register(mock_model)
    
    # Register same object again
    id2 = registry.register(mock_model)
    
    # Should return same ID (identity preservation)
    assert id1 == id2
    
    # Verify stats show a hit
    stats = registry.get_stats()
    assert stats["id_map_hits"] >= 1


def test_registry_different_objects_different_ids():
    """Test that different objects get different IDs."""
    registry = ModelPatcherRegistry()
    
    model1 = MockModelPatcher("model1")
    model2 = MockModelPatcher("model2")
    
    id1 = registry.register(model1)
    id2 = registry.register(model2)
    
    assert id1 != id2


def test_registry_stats():
    """Test registry statistics for debugging."""
    registry = ModelPatcherRegistry()
    
    # Get initial stats (registry may have state from previous tests due to singleton)
    initial_stats = registry.get_stats()
    initial_register_count = initial_stats["register_count"]
    initial_id_map_hits = initial_stats["id_map_hits"]
    
    model = MockModelPatcher()
    registry.register(model)
    registry.register(model)  # Re-register same
    
    stats = registry.get_stats()
    # Verify increments from our operations
    assert stats["register_count"] == initial_register_count + 1  # One new registration
    assert stats["id_map_hits"] == initial_id_map_hits + 1  # One identity hit


# =============================================================================
# Proxy Tests
# =============================================================================

def test_proxy_pickle_roundtrip():
    """Verify proxy serializes as ID only (<500 bytes)."""
    registry = ModelPatcherRegistry()
    mock_model = MockModelPatcher()
    model_id = registry.register(mock_model)
    
    proxy = ModelPatcherProxy(model_id, registry, manage_lifecycle=False)
    
    # Pickle it
    pickled = pickle.dumps(proxy)
    
    # Verify size is small (no model weights!)
    assert len(pickled) < 500, f"Pickled proxy is {len(pickled)} bytes (should be <500)"
    
    # Unpickle
    unpickled = pickle.loads(pickled)
    
    # Verify it reconstructed correctly
    assert unpickled._instance_id == model_id
    assert isinstance(unpickled, ModelPatcherProxy)


def test_proxy_clone_creates_new_id():
    """Verify clone returns new proxy with new ID."""
    registry = ModelPatcherRegistry()
    mock_model = MockModelPatcher()
    original_id = registry.register(mock_model)
    
    original_proxy = ModelPatcherProxy(original_id, registry, manage_lifecycle=False)
    
    # Set up event loop for async call
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        cloned_proxy = original_proxy.clone()
        
        # Verify new ID
        assert cloned_proxy._instance_id != original_id
        assert cloned_proxy._instance_id.startswith("model_")
        
        # Verify both exist in registry
        original_instance = registry._get_instance(original_id)
        cloned_instance = registry._get_instance(cloned_proxy._instance_id)
        
        assert original_instance is not cloned_instance
        assert cloned_instance._cloned is True
    finally:
        loop.close()


def test_proxy_model_options_getter_setter():
    """
    Test model_options property getter/setter.
    
    Research: Shadow-Reference Protocol Finding 2 (Lazy Patch Application)
    This is CPU-bound metadata, not VRAM weights.
    """
    registry = ModelPatcherRegistry()
    mock_model = MockModelPatcher()
    model_id = registry.register(mock_model)
    
    proxy = ModelPatcherProxy(model_id, registry, manage_lifecycle=False)
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        # Get current options
        options = proxy.model_options
        assert "transformer_options" in options
        
        # Set new options
        new_options = {"transformer_options": {"test": "value"}, "new_key": 42}
        proxy.model_options = new_options
        
        # Verify change persisted (via underlying model)
        assert mock_model.model_options["new_key"] == 42
    finally:
        loop.close()


def test_proxy_get_model_object():
    """Test get_model_object returns nested objects."""
    registry = ModelPatcherRegistry()
    mock_model = MockModelPatcher()
    model_id = registry.register(mock_model)
    
    proxy = ModelPatcherProxy(model_id, registry, manage_lifecycle=False)
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        model_sampling = proxy.get_model_object("model_sampling")
        
        # Should return the nested object
        assert hasattr(model_sampling, "percent_to_sigma")
        assert model_sampling.percent_to_sigma(0.5) == 50
    finally:
        loop.close()


def test_proxy_device_properties():
    """Test load_device and offload_device properties."""
    registry = ModelPatcherRegistry()
    mock_model = MockModelPatcher()
    model_id = registry.register(mock_model)
    
    proxy = ModelPatcherProxy(model_id, registry, manage_lifecycle=False)
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        assert proxy.load_device == "cuda:0"
        assert proxy.offload_device == "cpu"
    finally:
        loop.close()


def test_proxy_size_property():
    """Test size property for memory reporting."""
    registry = ModelPatcherRegistry()
    mock_model = MockModelPatcher()
    model_id = registry.register(mock_model)
    
    proxy = ModelPatcherProxy(model_id, registry, manage_lifecycle=False)
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        size = proxy.size
        assert size == 1024 * 1024 * 100  # 100MB
    finally:
        loop.close()


# =============================================================================
# Property Guards Tests
# =============================================================================

def test_property_guard_model():
    """Verify model property raises helpful AttributeError."""
    registry = ModelPatcherRegistry()
    mock_model = MockModelPatcher()
    model_id = registry.register(mock_model)
    
    proxy = ModelPatcherProxy(model_id, registry, manage_lifecycle=False)
    
    with pytest.raises(AttributeError, match="model.*not supported"):
        _ = proxy.model


def test_property_guard_patches():
    """Verify patches property raises helpful AttributeError."""
    registry = ModelPatcherRegistry()
    mock_model = MockModelPatcher()
    model_id = registry.register(mock_model)
    
    proxy = ModelPatcherProxy(model_id, registry, manage_lifecycle=False)
    
    with pytest.raises(AttributeError, match="patches.*not supported"):
        _ = proxy.patches


def test_method_guard_add_patches():
    """Verify add_patches raises NotImplementedError."""
    registry = ModelPatcherRegistry()
    mock_model = MockModelPatcher()
    model_id = registry.register(mock_model)
    
    proxy = ModelPatcherProxy(model_id, registry, manage_lifecycle=False)
    
    with pytest.raises(NotImplementedError, match="add_patches.*not supported"):
        proxy.add_patches({})


def test_method_guard_patch_model():
    """Verify patch_model raises NotImplementedError."""
    registry = ModelPatcherRegistry()
    mock_model = MockModelPatcher()
    model_id = registry.register(mock_model)
    
    proxy = ModelPatcherProxy(model_id, registry, manage_lifecycle=False)
    
    with pytest.raises(NotImplementedError, match="patch_model.*not supported"):
        proxy.patch_model()


# =============================================================================
# maybe_wrap_model_for_isolation Tests
# =============================================================================

def test_maybe_wrap_model_for_isolation_active():
    """Test wrapper wraps model when isolation active."""
    os.environ["PYISOLATE_ISOLATION_ACTIVE"] = "1"
    os.environ["PYISOLATE_CHILD"] = "0"
    
    try:
        mock_model = MockModelPatcher()
        wrapped = maybe_wrap_model_for_isolation(mock_model)
        
        # Should be wrapped
        assert isinstance(wrapped, ModelPatcherProxy)
        assert wrapped._instance_id.startswith("model_")
    finally:
        os.environ.pop("PYISOLATE_ISOLATION_ACTIVE", None)


def test_maybe_wrap_model_for_isolation_inactive():
    """Test wrapper returns original when isolation inactive."""
    os.environ.pop("PYISOLATE_ISOLATION_ACTIVE", None)
    
    mock_model = MockModelPatcher()
    result = maybe_wrap_model_for_isolation(mock_model)
    
    # Should return original
    assert result is mock_model
    assert not isinstance(result, ModelPatcherProxy)


def test_maybe_wrap_model_for_isolation_in_child():
    """Test wrapper returns original in child process."""
    os.environ["PYISOLATE_ISOLATION_ACTIVE"] = "1"
    os.environ["PYISOLATE_CHILD"] = "1"
    
    try:
        mock_model = MockModelPatcher()
        result = maybe_wrap_model_for_isolation(mock_model)
        
        # Should return original (child doesn't wrap)
        assert result is mock_model
    finally:
        os.environ["PYISOLATE_CHILD"] = "0"
        os.environ.pop("PYISOLATE_ISOLATION_ACTIVE", None)


def test_maybe_wrap_already_wrapped():
    """Test wrapper returns same proxy if already wrapped."""
    os.environ["PYISOLATE_ISOLATION_ACTIVE"] = "1"
    os.environ["PYISOLATE_CHILD"] = "0"
    
    try:
        mock_model = MockModelPatcher()
        wrapped1 = maybe_wrap_model_for_isolation(mock_model)
        wrapped2 = maybe_wrap_model_for_isolation(wrapped1)
        
        # Should return same proxy
        assert wrapped1 is wrapped2
    finally:
        os.environ.pop("PYISOLATE_ISOLATION_ACTIVE", None)


# =============================================================================
# Reconstruction Tests (Lifecycle)
# =============================================================================

def test_reconstruction_lifecycle_host_new():
    """Test reconstruction on host for new object manages lifecycle."""
    os.environ["PYISOLATE_CHILD"] = "0"
    
    proxy = _reconstruct_model_patcher_proxy("model_test", is_new_object=True)
    
    # Should manage lifecycle on host for new objects
    assert proxy._manage_lifecycle is True


def test_reconstruction_lifecycle_host_roundtrip():
    """Test reconstruction on host for round-trip does NOT manage lifecycle."""
    os.environ["PYISOLATE_CHILD"] = "0"
    
    proxy = _reconstruct_model_patcher_proxy("model_test", is_new_object=False)
    
    # Should NOT manage lifecycle (original proxy owns it)
    assert proxy._manage_lifecycle is False


def test_reconstruction_lifecycle_child():
    """Test reconstruction in child never manages lifecycle."""
    os.environ["PYISOLATE_CHILD"] = "1"
    
    try:
        # Both cases should be False in child
        proxy_new = _reconstruct_model_patcher_proxy("model_test", is_new_object=True)
        proxy_roundtrip = _reconstruct_model_patcher_proxy("model_test", is_new_object=False)
        
        assert proxy_new._manage_lifecycle is False
        assert proxy_roundtrip._manage_lifecycle is False
    finally:
        os.environ["PYISOLATE_CHILD"] = "0"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
