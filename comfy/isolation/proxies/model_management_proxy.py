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

