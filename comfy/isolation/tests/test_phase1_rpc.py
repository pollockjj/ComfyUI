#!/usr/bin/env python3
"""
Phase 1 Test: Basic RPC Forwarding
Tests that ModelPatcherProxy can forward method calls via mock RPC.
"""

import sys
sys.path.insert(0, '/home/johnj/ComfyUI')

from comfy.isolation.model_registry import ScopedModelRegistry
from comfy.isolation.model_proxy import ModelPatcherProxy


# Mock RPC client for testing
class MockRPCClient:
    def __init__(self, registry):
        self.registry = registry
    
    def call_sync(self, method, model_id, method_name, args, kwargs):
        """Simulate RPC call by executing on host-side object."""
        if method == "rpc_execute_model_method":
            patcher = self.registry.get(model_id)
            if patcher is None:
                raise ValueError(f"ModelPatcher {model_id} not found")
            
            real_method = getattr(patcher, method_name)
            return real_method(*args, **kwargs)
        
        raise NotImplementedError(f"Mock RPC doesn't support method: {method}")


def test_basic_forwarding():
    """Test that proxy forwards simple method calls."""
    print("🧪 Phase 1 Test: Basic RPC Forwarding")
    
    # Import torch and ModelPatcher
    try:
        import torch
        from comfy.model_patcher import ModelPatcher
    except ImportError as e:
        print(f"❌ Failed to import dependencies: {e}")
        return False
    
    # Create real patcher (host-side)
    print("  Creating ModelPatcher...")
    model = torch.nn.Linear(10, 10)
    patcher = ModelPatcher(model, torch.device('cpu'), torch.device('cpu'))
    
    # Register in scoped registry
    print("  Registering in ScopedModelRegistry...")
    registry = ScopedModelRegistry()
    model_id = registry.register(patcher)
    print(f"  Registered as model_id: {model_id}")
    
    # Create proxy (child-side simulation)
    print("  Creating ModelPatcherProxy...")
    rpc_client = MockRPCClient(registry)
    proxy = ModelPatcherProxy(model_id, rpc_client)
    
    # Test RPC call: model_size()
    print("  Testing proxy.model_size() via RPC...")
    proxy_size = proxy.model_size()
    real_size = patcher.model_size()
    
    print(f"  Proxy returned: {proxy_size}")
    print(f"  Real patcher:   {real_size}")
    
    if proxy_size == real_size:
        print("✅ Phase 1 Test PASSED: RPC forwarding works correctly")
        return True
    else:
        print(f"❌ Phase 1 Test FAILED: Size mismatch (proxy={proxy_size}, real={real_size})")
        return False


if __name__ == "__main__":
    success = test_basic_forwarding()
    sys.exit(0 if success else 1)
