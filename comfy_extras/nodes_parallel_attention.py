"""Parallel Attention configuration and control nodes."""

import torch
import logging

LOG_PREFIX = "⚡ [Parallel-Attention][Config Node]"


def get_device_list():
    """Get list of available devices for parallel attention."""
    devs = ["cpu"]
    
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        devs += [f"cuda:{i}" for i in range(device_count)]
    
    return devs


class ParallelAttentionConfig:
    """Configure parallel attention options for a model.
    
    Requires --use-parallel-attention CLI flag to be set.
    Extends base config attached by comfy/sd.py.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        devices = get_device_list()
        return {
            "required": {
                "model": ("MODEL",),
                "enable_fsdp2": ("BOOLEAN", {"default": True}),
                "device_1": (devices, {"default": "cuda:0"}),
                "device_2": (devices, {"default": "cuda:1"}),
                "backend": (["auto", "nccl", "gloo"], {"default": "auto"}),
            }
        }
    
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "configure"
    CATEGORY = "parallel_attention"
    
    def configure(self, model, enable_fsdp2, device_1, device_2, backend):
        """Configure parallel attention and spawn workers."""
        if not hasattr(model.model, 'parallel_config'):
            raise RuntimeError(
                f"{LOG_PREFIX} --use-parallel-attention flag not set. "
                "Start ComfyUI with: python main.py --use-parallel-attention"
            )
        
        if not model.model.parallel_config.get('enabled'):
            raise RuntimeError(f"{LOG_PREFIX} Parallel config disabled")
        
        # Set hardware parameters from node inputs
        model.model.parallel_config['enable_fsdp2'] = enable_fsdp2
        model.model.parallel_config['device_1'] = device_1
        model.model.parallel_config['device_2'] = device_2
        model.model.parallel_config['world_size'] = 2
        model.model.parallel_config['backend'] = backend if backend != "auto" else ("nccl" if torch.cuda.is_available() else "gloo")
        model.model.parallel_config['phase'] = "0.4"
        model.model.parallel_config['message'] = "Phase 0.4: Worker initialization"
        
        # Phase 0.4: Spawn workers if FSDP2 enabled
        if enable_fsdp2:
            from comfy.parallel_attention import FSDP2Executor
            
            # Check if executor already exists
            if hasattr(model, 'parallel_executor') and model.parallel_executor is not None:
                logging.info(f"{LOG_PREFIX} Executor already exists, reusing")
            else:
                # Log initialization start
                logging.info(f"{LOG_PREFIX} Initializing parallel attention engine")
                logging.info(f"{LOG_PREFIX} Configuration: world_size={model.model.parallel_config['world_size']}, backend={model.model.parallel_config['backend']}")
                logging.info(f"{LOG_PREFIX} Devices: {device_1}, {device_2}")
                
                # Spawn workers
                executor = FSDP2Executor(
                    world_size=model.model.parallel_config['world_size'],
                    backend=model.model.parallel_config['backend']
                )
                
                # Attach to model
                model.parallel_executor = executor
                
                logging.info(f"{LOG_PREFIX} Workers spawned and attached to model")
                logging.info(f"{LOG_PREFIX} Executor: world_size={executor.world_size}, backend={executor.backend}")
        
        logging.info(f"{LOG_PREFIX} Node config applied")
        logging.info(f"{LOG_PREFIX} Config: {model.model.parallel_config}")
        
        return (model,)


NODE_CLASS_MAPPINGS = {
    "ParallelAttentionConfig": ParallelAttentionConfig,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ParallelAttentionConfig": "Parallel Attention Config",
}