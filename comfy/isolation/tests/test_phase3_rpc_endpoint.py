#!/usr/bin/env python3
"""
Phase 3 Test: RPC Endpoint
Tests that rpc_execute_model_method handler works correctly.
"""

import sys
import asyncio
sys.path.insert(0, '/home/johnj/ComfyUI')
sys.path.insert(0, '/home/johnj/ComfyUI/user/pyisolate')

from comfy.isolation.model_registry import ScopedModelRegistry, set_current_registry
from comfy.isolation.rpc_handlers import rpc_execute_model_method, ALLOWED_METHODS


async def test_rpc_endpoint():
    """Test that RPC endpoint executes methods correctly."""
    print("🧪 Phase 3 Test: RPC Endpoint")
    
    # Import dependencies
    try:
        import torch
        from comfy.model_patcher import ModelPatcher
    except ImportError as e:
        print(f"❌ Failed to import dependencies: {e}")
        return False
    
    # Create registry and patcher
    print("  Creating ModelPatcher and registry...")
    registry = ScopedModelRegistry()
    set_current_registry(registry)  # Set context
    
    model = torch.nn.Linear(10, 10)
    patcher = ModelPatcher(model, torch.device('cpu'), torch.device('cpu'))
    
    model_id = registry.register(patcher)
    print(f"  Registered as model_id: {model_id}")
    
    # Test 1: Execute whitelisted method
    print("  Test 1: Execute whitelisted method (model_size)...")
    try:
        result = await rpc_execute_model_method(
            model_id=model_id,
            method_name="model_size",
            args=(),
            kwargs={}
        )
        print(f"  ✓ RPC returned: {result}")
        
        if result != patcher.model_size():
            print(f"❌ Size mismatch: RPC={result}, direct={patcher.model_size()}")
            return False
    
    except Exception as e:
        print(f"❌ RPC failed: {e}")
        return False
    
    # Test 2: Reject private method
    print("  Test 2: Reject private method access...")
    try:
        result = await rpc_execute_model_method(
            model_id=model_id,
            method_name="_some_private_method",
            args=(),
            kwargs={}
        )
        print(f"❌ Private method was not rejected!")
        return False
    
    except AttributeError as e:
        if "private method" in str(e):
            print(f"  ✓ Private method correctly rejected")
        else:
            print(f"❌ Wrong error: {e}")
            return False
    
    # Test 3: Reject non-whitelisted method
    print("  Test 3: Reject non-whitelisted method...")
    # Add a fake method to patcher for testing
    patcher.fake_method = lambda: "fake"
    
    try:
        result = await rpc_execute_model_method(
            model_id=model_id,
            method_name="fake_method",
            args=(),
            kwargs={}
        )
        print(f"❌ Non-whitelisted method was not rejected!")
        return False
    
    except AttributeError as e:
        if "not in whitelist" in str(e):
            print(f"  ✓ Non-whitelisted method correctly rejected")
        else:
            print(f"❌ Wrong error: {e}")
            return False
    
    # Test 4: Execute clone() which returns another ModelPatcher
    print("  Test 4: Execute clone() which returns ModelPatcherRef...")
    try:
        result = await rpc_execute_model_method(
            model_id=model_id,
            method_name="clone",
            args=(),
            kwargs={}
        )
        
        if not isinstance(result, dict):
            print(f"❌ Expected dict (ModelPatcherRef), got {type(result)}")
            return False
        
        if result.get("__type__") != "ModelPatcherRef":
            print(f"❌ Expected ModelPatcherRef, got {result}")
            return False
        
        clone_id = result.get("model_id")
        if not clone_id:
            print("❌ Missing model_id in result")
            return False
        
        print(f"  ✓ clone() returned ModelPatcherRef with id: {clone_id}")
        
        # Verify clone is in registry
        clone_patcher = registry.get(clone_id)
        if clone_patcher is None:
            print("❌ Cloned patcher not found in registry")
            return False
        
        print("  ✓ Cloned patcher found in registry")
    
    except Exception as e:
        print(f"❌ clone() RPC failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Cleanup
    set_current_registry(None)
    
    print("✅ Phase 3 Test PASSED: RPC endpoint works correctly")
    return True


if __name__ == "__main__":
    success = asyncio.run(test_rpc_endpoint())
    sys.exit(0 if success else 1)
