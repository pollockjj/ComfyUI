"""ProxiedSingleton for comfy.model_management module (Crystools subset)."""

import logging
import comfy.model_management as mm
from comfy.isolation import LOG_PREFIX

logger = logging.getLogger(__name__)

class ModelManagementProxy:
    """Proxy for model_management module providing device management for isolated nodes.
    
    This is NOT a ProxiedSingleton yet - it's a simple wrapper for testing.
    Crystools needs: get_torch_device, get_torch_device_name
    """
    
    def get_torch_device(self):
        """Get the torch device to use."""
        result = mm.get_torch_device()
        logger.debug(f"{LOG_PREFIX}[ModelManagementProxy] get_torch_device() → {result}")
        return result
    
    def get_torch_device_name(self, device) -> str:
        """Get device name as string."""
        result = mm.get_torch_device_name(device)
        logger.debug(f"{LOG_PREFIX}[ModelManagementProxy] get_torch_device_name({device}) → {result}")
        return result


def run_tests():
    """Run self-tests for ModelManagementProxy (called explicitly after ComfyUI init)."""
    import torch  # Safe to import here - called after ComfyUI torch init
    
    proxy = ModelManagementProxy()
    passed = 0
    failed = 0
    
    # Test 1: get_torch_device returns torch.device
    try:
        device = proxy.get_torch_device()
        assert isinstance(device, torch.device), f"Expected torch.device, got {type(device)}"
        logger.info(f"{LOG_PREFIX}[Test] ✅ ModelManagementProxy.get_torch_device() → {device}")
        passed += 1
    except Exception as e:
        logger.error(f"{LOG_PREFIX}[Test] ❌ ModelManagementProxy.get_torch_device() failed: {e}")
        failed += 1
    
    # Test 2: get_torch_device_name returns string
    try:
        device = proxy.get_torch_device()
        device_name = proxy.get_torch_device_name(device)
        assert isinstance(device_name, str), f"Expected str, got {type(device_name)}"
        assert len(device_name) > 0, "Device name is empty"
        logger.info(f"{LOG_PREFIX}[Test] ✅ ModelManagementProxy.get_torch_device_name() → {device_name}")
        passed += 1
    except Exception as e:
        logger.error(f"{LOG_PREFIX}[Test] ❌ ModelManagementProxy.get_torch_device_name() failed: {e}")
        failed += 1
    
    # Summary
    total = passed + failed
    logger.info(f"{LOG_PREFIX}[Test] ModelManagementProxy: {passed}/{total} tests passed")
    
    if failed > 0:
        raise RuntimeError(f"ModelManagementProxy self-tests failed: {failed}/{total}")
