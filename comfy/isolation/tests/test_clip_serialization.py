"""Test CLIP serialization integration."""

import pytest


class MockCLIP:
    """Mock CLIP object for testing serialization."""
    def __init__(self):
        self.value = "test_clip"
    
    def tokenize(self, text):
        return {"tokens": text}


def test_clip_serialization():
    """Test that CLIP objects are serialized to CLIPRef."""
    from pyisolate._internal.model_serialization import serialize_for_isolation
    
    # Create mock CLIP
    clip = MockCLIP()
    # Fake the type name
    clip.__class__.__name__ = "CLIP"
    
    # Serialize
    result = serialize_for_isolation(clip)
    
    # Should be a ref
    assert isinstance(result, dict)
    assert result["__type__"] == "CLIPRef"
    assert "clip_id" in result
    assert isinstance(result["clip_id"], str)
    print(f"✅ CLIP serialized to ref: {result}")


def test_nested_clip_serialization():
    """Test that CLIP objects in nested structures are serialized."""
    from pyisolate._internal.model_serialization import serialize_for_isolation
    
    # Create mock CLIP
    clip = MockCLIP()
    clip.__class__.__name__ = "CLIP"
    
    # Create nested structure
    data = {
        "clip": clip,
        "text": "hello",
        "nested": {"inner_clip": clip, "value": 42}
    }
    
    # Serialize
    result = serialize_for_isolation(data)
    
    # Check structure
    assert result["text"] == "hello"
    assert result["clip"]["__type__"] == "CLIPRef"
    assert result["nested"]["value"] == 42
    assert result["nested"]["inner_clip"]["__type__"] == "CLIPRef"
    
    # Should reuse same ID for same object
    assert result["clip"]["clip_id"] == result["nested"]["inner_clip"]["clip_id"]
    print(f"✅ Nested CLIP serialization preserves identity")


if __name__ == "__main__":
    test_clip_serialization()
    test_nested_clip_serialization()
    print("\n✅ All CLIP serialization tests passed!")
