"""End-to-end test for CLIP serialization in execution flow."""

import pytest


class MockCLIP:
    """Mock CLIP with lambda (unpicklable)."""
    def __init__(self):
        self.tokenizer = lambda x: {"tokens": x}  # Lambda = unpicklable!
    
    def tokenize(self, text):
        return self.tokenizer(text)


def test_clip_execution_flow():
    """Test that CLIP objects are serialized during execution."""
    from pyisolate._internal.model_serialization import serialize_for_isolation
    
    # Create mock CLIP
    clip = MockCLIP()
    clip.__class__.__name__ = "CLIP"
    
    try:
        # Simulate node inputs
        inputs = {
            "clip": clip,
            "text": "hello world",
            "strength": 1.0,
        }
        
        # Serialize (this is what _execute does)
        serialized = serialize_for_isolation(inputs)
        
        # Verify CLIP was converted to ref
        assert isinstance(serialized["clip"], dict)
        assert serialized["clip"]["__type__"] == "CLIPRef"
        assert "clip_id" in serialized["clip"]
        
        # Other inputs unchanged
        assert serialized["text"] == "hello world"
        assert serialized["strength"] == 1.0
        
        # Verify CLIPRegistry has it
        from comfy.isolation.clip_proxy import CLIPRegistry
        clip_registry = CLIPRegistry()
        clip_id = serialized["clip"]["clip_id"]
        retrieved = clip_registry._get_instance(clip_id)
        assert retrieved is clip
        
        print(f"✅ CLIP execution flow works! ref={clip_id}")
        
    finally:
        pass


def test_nested_clip_in_execution():
    """Test nested CLIP objects in complex structures."""
    from pyisolate._internal.model_serialization import serialize_for_isolation
    
    # Create mock CLIPs
    clip1 = MockCLIP()
    clip1.__class__.__name__ = "CLIP"
    clip2 = MockCLIP()
    clip2.__class__.__name__ = "CLIP"
    
    try:
        # Complex nested structure
        inputs = {
            "primary_clip": clip1,
            "config": {
                "secondary_clip": clip2,
                "strength": 0.8,
            },
            "list_of_clips": [clip1, clip2],
        }
        
        # Serialize
        serialized = serialize_for_isolation(inputs)
        
        # Verify all CLIPs converted
        assert serialized["primary_clip"]["__type__"] == "CLIPRef"
        assert serialized["config"]["secondary_clip"]["__type__"] == "CLIPRef"
        assert serialized["list_of_clips"][0]["__type__"] == "CLIPRef"
        assert serialized["list_of_clips"][1]["__type__"] == "CLIPRef"
        
        # Identity preserved (clip1 appears twice)
        assert (
            serialized["primary_clip"]["clip_id"]
            == serialized["list_of_clips"][0]["clip_id"]
        )
        
        print("✅ Nested CLIP serialization works in execution!")
        
    finally:
        set_current_registry(None)


if __name__ == "__main__":
    test_clip_execution_flow()
    test_nested_clip_in_execution()
    print("\n✅ All E2E CLIP tests passed!")
