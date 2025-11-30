"""Test full serialization round-trip through extension wrapper."""

import pytest


class MockCLIP:
    """Mock CLIP with lambda (unpicklable)."""
    def __init__(self):
        self.tokenizer = lambda x: {"tokens": x}  # Lambda = unpicklable!
    
    def tokenize(self, text):
        return self.tokenizer(text)


def test_extension_wrapper_clip_deserialization():
    """Test that extension wrapper can deserialize CLIPRef."""
    from comfy.isolation.extension_wrapper import ComfyNodeExtension
    from comfy.isolation.clip_proxy import CLIPRegistry
    
    # Create mock CLIP and register it
    clip = MockCLIP()
    clip.__class__.__name__ = "CLIP"
    
    registry = CLIPRegistry()
    clip_id = registry.register(clip)
    
    # Create CLIPRef (as serialization layer would)
    clip_ref = {"__type__": "CLIPRef", "clip_id": clip_id}
    
    # Create wrapper and resolve
    wrapper = ComfyNodeExtension()
    inputs = {"clip": clip_ref, "text": "hello", "strength": 1.0}
    resolved = wrapper._resolve_remote_objects(inputs)
    
    # Verify CLIPProxy was created
    from comfy.isolation.clip_proxy import CLIPProxy
    assert isinstance(resolved["clip"], CLIPProxy)
    assert resolved["clip"]._instance_id == clip_id
    assert resolved["text"] == "hello"
    assert resolved["strength"] == 1.0
    
    print(f"✅ Extension wrapper deserialization works! proxy={resolved['clip']}")


def test_extension_wrapper_nested_clip():
    """Test nested CLIP deserialization."""
    from comfy.isolation.extension_wrapper import ComfyNodeExtension
    from comfy.isolation.clip_proxy import CLIPRegistry
    
    # Create and register CLIP
    clip = MockCLIP()
    clip.__class__.__name__ = "CLIP"
    registry = CLIPRegistry()
    clip_id = registry.register(clip)
    
    # Nested structure with CLIPRef
    clip_ref = {"__type__": "CLIPRef", "clip_id": clip_id}
    inputs = {
        "primary": clip_ref,
        "config": {"secondary": clip_ref, "value": 42},
        "list": [clip_ref, "text"],
    }
    
    # Resolve
    wrapper = ComfyNodeExtension()
    resolved = wrapper._resolve_remote_objects(inputs)
    
    # Verify all resolved correctly
    from comfy.isolation.clip_proxy import CLIPProxy
    assert isinstance(resolved["primary"], CLIPProxy)
    assert isinstance(resolved["config"]["secondary"], CLIPProxy)
    assert isinstance(resolved["list"][0], CLIPProxy)
    assert resolved["list"][1] == "text"
    assert resolved["config"]["value"] == 42
    
    print("✅ Nested CLIP deserialization works in extension wrapper!")


if __name__ == "__main__":
    test_extension_wrapper_clip_deserialization()
    test_extension_wrapper_nested_clip()
    print("\n✅ All extension wrapper tests passed!")
