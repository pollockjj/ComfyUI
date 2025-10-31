"""FSDP2ModelPatcher - Relay forward calls to distributed workers.

Extends ModelPatcher to relay apply_model() calls to FSDP2 workers.
Maintains 100% compatibility with ComfyUI's existing workflow system.

Based on Raylight FSDPModelPatcher pattern, adapted for multiprocessing.
"""

from comfy.model_patcher import ModelPatcher
import torch
import logging

LOG_PREFIX = "⚡ [Parallel-Attention]"


class FSDP2ModelPatcher(ModelPatcher):
    """ModelPatcher that relays forward calls to FSDP2 workers.
    
    Extends ModelPatcher to support distributed inference while maintaining
    full compatibility with ComfyUI's sampling and node system.
    
    Pattern: Raylight FSDPModelPatcher + multiprocessing executor
    """
    
    def __init__(self, model, load_device, offload_device, size=0, 
                 weight_inplace_update=False, executor=None):
        """Initialize FSDP2ModelPatcher.
        
        Args:
            model: Model instance (on parent process)
            load_device: Device to load to
            offload_device: Device to offload to
            size: Model size in bytes
            weight_inplace_update: Whether to update weights in place
            executor: MultiprocExecutor with workers
        """
        super().__init__(
            model=model,
            load_device=load_device,
            offload_device=offload_device,
            size=size,
            weight_inplace_update=weight_inplace_update
        )
        self.executor = executor
        self.is_fsdp2 = executor is not None
        
        logging.info(f"{LOG_PREFIX} [FSDP2ModelPatcher] Created with executor: {self.is_fsdp2}")
    
    def apply_model(self, x, t, c_concat=None, c_crossattn=None, control=None, 
                   transformer_options={}, **kwargs):
        """Apply model forward pass.
        
        If FSDP2 executor attached, relay to workers.
        Otherwise, call parent implementation.
        
        Args:
            x: Latent input
            t: Timestep
            c_concat: Concat conditioning
            c_crossattn: Cross-attention conditioning
            control: ControlNet conditioning
            transformer_options: Additional options
            **kwargs: Additional arguments
            
        Returns:
            Model output (denoised latent)
        """
        if self.is_fsdp2 and self.executor is not None:
            # Relay to FSDP2 workers
            logging.debug(f"{LOG_PREFIX} [FSDP2ModelPatcher] Relaying apply_model to workers")
            
            # Prepare arguments for worker forward
            # Workers expect: (x, timestep, context, y, guidance, ...)
            context = c_crossattn.get("c_crossattn", None) if isinstance(c_crossattn, dict) else c_crossattn
            y = c_concat.get("y", None) if isinstance(c_concat, dict) else None
            guidance = kwargs.get("guidance", None)
            
            result = self.executor.execute_collective(
                "forward",
                {
                    "args": (x, t, context, y, guidance),
                    "kwargs": {}
                }
            )
            
            if result.get("status") == "success":
                output = result.get("output")
                logging.debug(f"{LOG_PREFIX} [FSDP2ModelPatcher] Received output from workers")
                return output
            else:
                error = result.get("error", "Unknown error")
                raise RuntimeError(f"Worker forward failed: {error}")
        else:
            # No FSDP2, use parent implementation
            return super().apply_model(x, t, c_concat, c_crossattn, control, 
                                      transformer_options, **kwargs)
    
    def clone(self, *args, **kwargs):
        """Clone the model patcher.
        
        Maintains executor reference in cloned instance.
        """
        n = super().clone(*args, **kwargs)
        n.__class__ = FSDP2ModelPatcher
        n.executor = self.executor
        n.is_fsdp2 = self.is_fsdp2
        return n
    
    def model_memory_required(self, device):
        """Report sharded memory size to ComfyUI scheduler.
        
        Returns:
            Memory required per GPU (sharded size, not full size)
        """
        if self.is_fsdp2 and self.executor is not None:
            # Return sharded size (full size / world_size)
            world_size = self.executor.world_size
            full_size = super().model_memory_required(device)
            sharded_size = full_size // world_size
            logging.debug(f"{LOG_PREFIX} [FSDP2ModelPatcher] Memory required: {sharded_size / (1024**3):.2f}GB (sharded)")
            return sharded_size
        else:
            return super().model_memory_required(device)
