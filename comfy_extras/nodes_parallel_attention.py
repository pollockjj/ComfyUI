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
        # Check for context
        if not hasattr(model, 'parallel_attention'):
            raise RuntimeError(
                f"{LOG_PREFIX} No parallel_attention context found. "
                "Ensure --use-parallel-attention flag is set."
            )
        
        ctx = model.parallel_attention
        
        if not ctx.enabled:
            raise RuntimeError(f"{LOG_PREFIX} Context disabled (no policy for {ctx.model_type})")
        
        # Phase B: Worker Initialization
        if enable_fsdp2 and ctx.executor is None:
            from comfy.parallel_attention import FSDP2Executor
            
            # Determine backend
            actual_backend = backend if backend != "auto" else ("nccl" if torch.cuda.is_available() else "gloo")
            
            logging.info(f"{LOG_PREFIX} Spawning workers for {ctx.model_type}")
            logging.info(f"{LOG_PREFIX} Devices: {device_1}, {device_2}")
            logging.info(f"{LOG_PREFIX} Backend: {actual_backend}")
            
            # Spawn workers
            executor = FSDP2Executor(world_size=2, backend=actual_backend)
            
            # Populate context
            ctx.executor = executor
            ctx.world_size = 2
            ctx.backend = actual_backend
            ctx.phase = "workers_initialized"
            
            logging.info(f"{LOG_PREFIX} Workers spawned")
            ctx.log_state(LOG_PREFIX)
        elif ctx.executor is not None:
            logging.info(f"{LOG_PREFIX} Workers already initialized, reusing")
        
        return (model,)


NODE_CLASS_MAPPINGS = {
    "ParallelAttentionConfig": ParallelAttentionConfig,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ParallelAttentionConfig": "Parallel Attention Config",
}