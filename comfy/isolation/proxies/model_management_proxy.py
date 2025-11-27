"""Proxy for comfy.model_management module."""

import logging
import comfy.model_management as mm
from comfy.isolation import LOG_PREFIX

logger = logging.getLogger(__name__)

class ModelManagementProxy:
    """Proxy for comfy.model_management providing device management, memory control, and dtype selection for isolated nodes."""
    
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
    
    def soft_empty_cache(self, force=False):
        """Empty CUDA/MPS/XPU cache to free memory."""
        logger.debug(f"{LOG_PREFIX}[ModelManagementProxy] soft_empty_cache(force={force})")
        mm.soft_empty_cache(force=force)
    
    def unet_dtype(self, device=None, model_params=0, supported_dtypes=None, weight_dtype=None):
        """Get optimal dtype for UNet."""
        if supported_dtypes is None:
            supported_dtypes = [mm.torch.float16, mm.torch.bfloat16, mm.torch.float32]
        result = mm.unet_dtype(device, model_params, supported_dtypes, weight_dtype)
        logger.debug(f"{LOG_PREFIX}[ModelManagementProxy] unet_dtype({device}, {model_params}) → {result}")
        return result
    
    def unet_offload_device(self):
        """Get device to offload UNet to."""
        result = mm.unet_offload_device()
        logger.debug(f"{LOG_PREFIX}[ModelManagementProxy] unet_offload_device() → {result}")
        return result
    
    def free_memory(self, memory_required, device, keep_loaded=None):
        """Free memory by unloading models."""
        if keep_loaded is None:
            keep_loaded = []
        logger.debug(f"{LOG_PREFIX}[ModelManagementProxy] free_memory({memory_required}, {device})")
        return mm.free_memory(memory_required, device, keep_loaded)
    
    def load_models_gpu(self, models, memory_required=0, force_patch_weights=False, minimum_memory_required=None, force_full_load=False):
        """Load models to GPU."""
        logger.debug(f"{LOG_PREFIX}[ModelManagementProxy] load_models_gpu({len(models)} models)")
        return mm.load_models_gpu(models, memory_required, force_patch_weights, minimum_memory_required, force_full_load)
    
    def load_model_gpu(self, model):
        """Load single model to GPU."""
        logger.debug(f"{LOG_PREFIX}[ModelManagementProxy] load_model_gpu()")
        return mm.load_model_gpu(model)
    
    def should_use_fp16(self, device=None, model_params=0, prioritize_performance=True, manual_cast=False):
        """Check if FP16 should be used."""
        result = mm.should_use_fp16(device, model_params, prioritize_performance, manual_cast)
        logger.debug(f"{LOG_PREFIX}[ModelManagementProxy] should_use_fp16({device}) → {result}")
        return result
    
    def should_use_bf16(self, device=None, model_params=0, prioritize_performance=True, manual_cast=False):
        """Check if BF16 should be used."""
        result = mm.should_use_bf16(device, model_params, prioritize_performance, manual_cast)
        logger.debug(f"{LOG_PREFIX}[ModelManagementProxy] should_use_bf16({device}) → {result}")
        return result
    
    def text_encoder_offload_device(self):
        """Get device to offload text encoder to."""
        result = mm.text_encoder_offload_device()
        logger.debug(f"{LOG_PREFIX}[ModelManagementProxy] text_encoder_offload_device() → {result}")
        return result
    
    def intermediate_device(self):
        """Get intermediate computation device."""
        result = mm.intermediate_device()
        logger.debug(f"{LOG_PREFIX}[ModelManagementProxy] intermediate_device() → {result}")
        return result
    
    def module_size(self, module):
        """Calculate module size in bytes."""
        result = mm.module_size(module)
        logger.debug(f"{LOG_PREFIX}[ModelManagementProxy] module_size() → {result}")
        return result
    
    @property
    def OOM_EXCEPTION(self):
        """Out of memory exception class."""
        return mm.OOM_EXCEPTION
    
    @property
    def XFORMERS_IS_AVAILABLE(self):
        """Check if xformers is available."""
        return mm.XFORMERS_IS_AVAILABLE

