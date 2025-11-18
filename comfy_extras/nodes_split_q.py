"""
Split-Q Parallel Attention Override Node

Provides a ComfyUI node that allows manual control of Split-Q multi-GPU execution.
Can override CLI --use-split-q-multigpu flag and specify custom device pairs.
"""

import logging
import torch


class SplitQOverride:
    """
    Manual override for Split-Q multi-GPU configuration.
    Allows enabling/disabling and custom device selection.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        # Detect available CUDA devices
        available_devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        if not available_devices:
            available_devices = ["cuda:0"]  # Fallback
        
        return {
            "required": {
                "model": ("MODEL",),
                "enable_split_q": ("BOOLEAN", {"default": False}),
                "device_primary": (available_devices, {"default": "cuda:0"}),
                "device_secondary": (available_devices, {"default": "cuda:1" if len(available_devices) > 1 else "cuda:0"}),
            }
        }
    
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "configure_split_q"
    CATEGORY = "advanced/split_q"
    
    def configure_split_q(self, model, enable_split_q, device_primary, device_secondary):
        """
        Sets Split-Q configuration in transformer_options.
        Can override CLI flag and specify custom devices.
        """
        clone = model.clone()
        clone.model_options = dict(clone.model_options)
        clone.model_options.setdefault("transformer_options", {})["split_q_override"] = {
            "enable": enable_split_q,
            "device_primary": device_primary,
            "device_secondary": device_secondary,
        }
        
        logging.info("⚡ [split-q][Override] enable=%s, devices=%s/%s", 
                     enable_split_q, device_primary, device_secondary)
        
        return (clone,)


NODE_CLASS_MAPPINGS = {
    "SplitQOverride": SplitQOverride,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SplitQOverride": "Split-Q Multi-GPU Override",
}
