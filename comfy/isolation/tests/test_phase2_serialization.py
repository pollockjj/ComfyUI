#!/usr/bin/env python3
"""
Phase 2 Test: Serialization Layer
Tests ModelPatcher serialization/deserialization with registry.
"""

import sys
sys.path.insert(0, '/home/johnj/ComfyUI')
sys.path.insert(0, '/home/johnj/ComfyUI/user/pyisolate')

from pyisolate._internal.model_serialization import (
    serialize_for_isolation,
    deserialize_from_isolation,
    deserialize_proxy_result,
)


def test_serialization():
    """Test that ModelPatcher serializes to ref and deserializes correctly."""
    print("🧪 Phase 2 Test: Serialization Layer")
    
    # Import dependencies
    try:
        import torch
        from comfy.model_patcher import ModelPatcher
    except ImportError as e:
        print(f"❌ Failed to import dependencies: {e}")
        return False
    
    # Create patcher
    print("  Creating ModelPatcher...")
    model = torch.nn.Linear(10, 10)
    patcher = ModelPatcher(model, torch.device('cpu'), torch.device('cpu'))
    
    # Test 1: Serialize ModelPatcher → ref
    print("  Test 1: Serialize ModelPatcher → ModelPatcherRef...")
    serialized = serialize_for_isolation(patcher)
    
    if not isinstance(serialized, dict):
        print(f"❌ Expected dict, got {type(serialized)}")
        return False
    
    if serialized.get("__type__") != "ModelPatcherRef":
        print(f"❌ Expected __type__=ModelPatcherRef, got {serialized.get('__type__')}")
        return False
    
    model_id = serialized.get("model_id")
    if not model_id:
        print("❌ Missing model_id in serialized data")
        return False
    
    print(f"  ✓ Serialized to: {serialized}")
    
    # Test 2: Deserialize ref → ModelPatcher (host-side)
    print("  Test 2: Deserialize ModelPatcherRef → ModelPatcher (host)...")
    deserialized = deserialize_from_isolation(serialized)
    
    if deserialized is not patcher:
        print(f"❌ Expected same object, got different instance")
        return False
    
    print("  ✓ Deserialized to original ModelPatcher")
    
    # Test 3: Deserialize ref → Proxy (child-side)
    print("  Test 3: Deserialize ModelPatcherRef → ModelPatcherProxy (child)...")
    proxy = deserialize_proxy_result(serialized)
    
    from comfy.isolation.model_patcher_proxy import ModelPatcherProxy
    if not isinstance(proxy, ModelPatcherProxy):
        print(f"❌ Expected ModelPatcherProxy, got {type(proxy)}")
        return False
    
    print(f"  ✓ Deserialized to proxy: {proxy}")
    
    # Test 4: Nested structures
    print("  Test 4: Serialize nested dict with ModelPatcher...")
    nested = {
        "model": patcher,
        "params": {"strength": 1.0},
        "list": [patcher, "text", 123]
    }
    
    serialized_nested = serialize_for_isolation(nested)
    
    if serialized_nested["model"].get("__type__") != "ModelPatcherRef":
        print("❌ Nested ModelPatcher not serialized")
        return False
    
    if serialized_nested["list"][0].get("__type__") != "ModelPatcherRef":
        print("❌ ModelPatcher in list not serialized")
        return False
    
    print("  ✓ Nested structures serialized correctly")
    
    print("✅ Phase 2 Test PASSED: Serialization layer works correctly")
    return True


if __name__ == "__main__":
    success = test_serialization()
    sys.exit(0 if success else 1)
