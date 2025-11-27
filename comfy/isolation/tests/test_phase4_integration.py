#!/usr/bin/env python3
"""
Phase 4 Integration Test: End-to-End MODEL Serialization
Tests that MODEL can cross the process boundary via our hooks.
"""

import sys
import asyncio
sys.path.insert(0, '/home/johnj/ComfyUI')
sys.path.insert(0, '/home/johnj/ComfyUI/user/pyisolate')

async def test_model_integration():
    """Test full MODEL serialization through PyIsolate RPC."""
    print("🧪 Phase 4 Integration Test: MODEL Serialization")
    
    # Import dependencies
    try:
        import torch
        from comfy.model_patcher import ModelPatcher
        from comfy.isolation.model_registry import ScopedModelRegistry, set_current_registry
        from pyisolate._internal.shared import _tensor_to_cpu, _tensor_to_cuda
    except ImportError as e:
        print(f"❌ Failed to import dependencies: {e}")
        return False
    
    # Simulate host-side: Create ModelPatcher and registry
    print("  [Host] Creating ModelPatcher and registry...")
    registry = ScopedModelRegistry()
    set_current_registry(registry)
    
    model = torch.nn.Linear(10, 10)
    patcher = ModelPatcher(model, torch.device('cpu'), torch.device('cpu'))
    
    # Test 1: Serialize via _tensor_to_cpu (sends to child)
    print("  Test 1: Serialize MODEL via _tensor_to_cpu...")
    data_to_send = {"model": patcher, "strength": 1.0}
    
    serialized = _tensor_to_cpu(data_to_send)
    
    if not isinstance(serialized["model"], dict):
        print(f"❌ Expected dict (ModelPatcherRef), got {type(serialized['model'])}")
        return False
    
    if serialized["model"].get("__type__") != "ModelPatcherRef":
        print(f"❌ Expected ModelPatcherRef, got {serialized['model']}")
        return False
    
    model_id = serialized["model"]["model_id"]
    print(f"  ✓ Serialized to ModelPatcherRef: {model_id}")
    
    # Test 2: Deserialize via _tensor_to_cuda (receives in host)
    print("  Test 2: Deserialize ModelPatcherRef back to MODEL (host-side)...")
    
    deserialized = _tensor_to_cuda(serialized)
    
    if deserialized["model"] is not patcher:
        print(f"❌ Expected same ModelPatcher instance, got different object")
        return False
    
    print("  ✓ Deserialized back to original ModelPatcher")
    
    # Test 3: Verify nested structures work
    print("  Test 3: Test nested MODEL in list...")
    nested_data = {
        "models": [patcher, patcher],
        "params": {"strength": 1.5}
    }
    
    serialized_nested = _tensor_to_cpu(nested_data)
    
    if serialized_nested["models"][0].get("__type__") != "ModelPatcherRef":
        print("❌ Nested MODEL not serialized")
        return False
    
    if serialized_nested["models"][1].get("__type__") != "ModelPatcherRef":
        print("❌ Second nested MODEL not serialized")
        return False
    
    # Both should have same model_id (identity preservation)
    if serialized_nested["models"][0]["model_id"] != serialized_nested["models"][1]["model_id"]:
        print("❌ Identity not preserved (same object should have same ID)")
        return False
    
    print(f"  ✓ Nested MODELs serialized with identity preserved")
    
    # Cleanup
    set_current_registry(None)
    
    print("✅ Phase 4 Integration Test PASSED: MODEL serialization works end-to-end")
    return True


if __name__ == "__main__":
    success = asyncio.run(test_model_integration())
    sys.exit(0 if success else 1)
