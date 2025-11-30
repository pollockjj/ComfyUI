"""
Unit tests for clip_proxy.py (CLIPRegistry + CLIPProxy).

Tests cover:
1. Registry CRUD operations
2. Pickle round-trip (<500 bytes)
3. Clone lifecycle management
4. Property guards raise AttributeError
"""

import os
import pickle
import pytest
from unittest.mock import Mock, MagicMock

# Set environment to simulate host process
os.environ["PYISOLATE_CHILD"] = "0"
os.environ.pop("PYISOLATE_ISOLATION_ACTIVE", None)

from comfy.isolation.clip_proxy import (
    CLIPRegistry,
    CLIPProxy,
    _reconstruct_clip_proxy,
    maybe_wrap_clip_for_isolation
)


class MockCLIP:
    """Mock CLIP object for testing."""
    
    def __init__(self, name="mock_clip"):
        self.name = name
        self._cloned = False
    
    def clone(self):
        """Create a clone with modified name."""
        cloned = MockCLIP(f"{self.name}_clone")
        cloned._cloned = True
        return cloned
    
    def get_ram_usage(self):
        return 1024
    
    def tokenize(self, text, return_word_ids=False, **kwargs):
        return {"tokens": [1, 2, 3]}
    
    def encode(self, text):
        return f"encoded_{text}"
    
    def encode_from_tokens(self, tokens, return_pooled=False, return_dict=False):
        return "encoded_tokens"
    
    def encode_from_tokens_scheduled(
        self, tokens, unprojected=False, add_dict=None, show_pbar=True
    ):
        return [("result", {})]


def test_registry_register_unregister():
    """Test registry registration and cleanup."""
    # Create fresh registry
    registry = CLIPRegistry()
    
    # Register a CLIP instance
    mock_clip = MockCLIP()
    clip_id = registry.register(mock_clip)
    
    # Verify ID format
    assert clip_id.startswith("clip_")
    
    # Verify it's in registry
    retrieved = registry._get_instance(clip_id)
    assert retrieved is mock_clip
    
    # Unregister
    registry.unregister_sync(clip_id)
    
    # Verify it's gone
    with pytest.raises(ValueError, match="not found in registry"):
        registry._get_instance(clip_id)


def test_proxy_pickle_roundtrip():
    """Verify proxy serializes as ID only (<500 bytes)."""
    registry = CLIPRegistry()
    mock_clip = MockCLIP()
    clip_id = registry.register(mock_clip)
    
    # Create proxy (no lifecycle management for test)
    proxy = CLIPProxy(clip_id, registry, manage_lifecycle=False)
    
    # Pickle it
    pickled = pickle.dumps(proxy)
    
    # Verify size
    assert len(pickled) < 500, f"Pickled proxy is {len(pickled)} bytes (should be <500)"
    
    # Unpickle
    unpickled = pickle.loads(pickled)
    
    # Verify it reconstructed correctly
    assert unpickled._instance_id == clip_id
    assert isinstance(unpickled, CLIPProxy)


def test_clone_creates_new_id():
    """Verify clone returns new proxy with new ID."""
    registry = CLIPRegistry()
    mock_clip = MockCLIP()
    original_id = registry.register(mock_clip)
    
    # Create proxy
    original_proxy = CLIPProxy(original_id, registry, manage_lifecycle=False)
    
    # Clone it (mocking the async call)
    import asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        cloned_proxy = original_proxy.clone()
        
        # Verify new ID
        assert cloned_proxy._instance_id != original_id
        assert cloned_proxy._instance_id.startswith("clip_")
        
        # Verify both exist in registry
        original_instance = registry._get_instance(original_id)
        cloned_instance = registry._get_instance(cloned_proxy._instance_id)
        
        assert original_instance is not cloned_instance
        assert cloned_instance._cloned is True
    finally:
        loop.close()


def test_property_guards_raise_errors():
    """Verify property guards raise helpful AttributeError."""
    registry = CLIPRegistry()
    mock_clip = MockCLIP()
    clip_id = registry.register(mock_clip)
    
    proxy = CLIPProxy(clip_id, registry, manage_lifecycle=False)
    
    # Test patcher
    with pytest.raises(AttributeError, match="patcher.*not supported"):
        _ = proxy.patcher
    
    # Test cond_stage_model
    with pytest.raises(AttributeError, match="cond_stage_model.*not supported"):
        _ = proxy.cond_stage_model
    
    # Test layer_idx
    with pytest.raises(AttributeError, match="layer_idx.*not supported"):
        _ = proxy.layer_idx
    
    # Test use_clip_schedule
    with pytest.raises(AttributeError, match="use_clip_schedule.*not supported"):
        _ = proxy.use_clip_schedule


def test_load_model_raises_not_implemented():
    """Verify load_model raises NotImplementedError."""
    registry = CLIPRegistry()
    mock_clip = MockCLIP()
    clip_id = registry.register(mock_clip)
    
    proxy = CLIPProxy(clip_id, registry, manage_lifecycle=False)
    
    with pytest.raises(NotImplementedError, match="load_model.*not supported"):
        proxy.load_model()


def test_maybe_wrap_clip_for_isolation_active():
    """Test wrapper wraps CLIP when isolation active."""
    os.environ["PYISOLATE_ISOLATION_ACTIVE"] = "1"
    
    try:
        mock_clip = MockCLIP()
        wrapped = maybe_wrap_clip_for_isolation(mock_clip)
        
        # Should be wrapped
        assert isinstance(wrapped, CLIPProxy)
        assert wrapped._instance_id.startswith("clip_")
    finally:
        os.environ.pop("PYISOLATE_ISOLATION_ACTIVE", None)


def test_maybe_wrap_clip_for_isolation_inactive():
    """Test wrapper returns original when isolation inactive."""
    os.environ.pop("PYISOLATE_ISOLATION_ACTIVE", None)
    
    mock_clip = MockCLIP()
    result = maybe_wrap_clip_for_isolation(mock_clip)
    
    # Should return original
    assert result is mock_clip
    assert not isinstance(result, CLIPProxy)


def test_reconstruction_lifecycle_host_new():
    """Test reconstruction on host for new object manages lifecycle."""
    os.environ["PYISOLATE_CHILD"] = "0"
    
    proxy = _reconstruct_clip_proxy("clip_test", is_new_object=True)
    
    # Should manage lifecycle on host for new objects
    assert proxy._manage_lifecycle is True


def test_reconstruction_lifecycle_host_roundtrip():
    """Test reconstruction on host for round-trip does NOT manage lifecycle."""
    os.environ["PYISOLATE_CHILD"] = "0"
    
    proxy = _reconstruct_clip_proxy("clip_test", is_new_object=False)
    
    # Should NOT manage lifecycle (original proxy owns it)
    assert proxy._manage_lifecycle is False


def test_reconstruction_lifecycle_child():
    """Test reconstruction in child never manages lifecycle."""
    os.environ["PYISOLATE_CHILD"] = "1"
    
    try:
        # Both cases should be False in child
        proxy_new = _reconstruct_clip_proxy("clip_test", is_new_object=True)
        proxy_roundtrip = _reconstruct_clip_proxy("clip_test", is_new_object=False)
        
        assert proxy_new._manage_lifecycle is False
        assert proxy_roundtrip._manage_lifecycle is False
    finally:
        os.environ["PYISOLATE_CHILD"] = "0"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
