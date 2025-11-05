"""Ring-Attention wrapper using ComfyUI's patches/patches_replace mechanism.

This module implements the wrapper that injects Ring-Attention patches into
the Flux forward_orig execution flow using ComfyUI's native patching system.

Architecture:
    - Intercepts apply_model via model_function_wrapper seam
    - Injects ring_context into transformer_options
    - Attaches three patch functions at strategic points
    - Dispatches to workers with patches active

Reference: Research report "ComfyUI Patches for Ring-Attention.md"
"""

import torch
import torch.distributed as dist
import logging
from typing import Callable, Dict, Any, Optional

from .ring_patches import RingAttentionPatches

LOG_PREFIX = "⚡ [RingAttention]"


class RingAttentionWrapper:
    """Ring-Attention wrapper using patches/patches_replace mechanism.
    
    This wrapper intercepts the apply_model call and injects Ring-Attention
    patches into transformer_options before dispatching to workers.
    
    Pattern: "Chained Patch" strategy using three patch functions to manage
    sequence parallelism split/gather operations.
    """
    
    def __init__(
        self, 
        executor, 
        world_size: int,
        depth_double: int = 19,
        depth_single: int = 38
    ):
        """Initialize Ring-Attention wrapper.
        
        Args:
            executor: FSDP2Executor with worker pool
            world_size: Number of GPUs for sequence parallelism
            depth_double: Number of double_blocks (19 for Flux-Dev)
            depth_single: Number of single_blocks (38 for Flux-Dev)
        """
        self.executor = executor
        self.world_size = world_size
        
        # Create patch functions
        self.patches = RingAttentionPatches(
            depth_double=depth_double,
            depth_single=depth_single
        )
        
        # Step tracking for it/s measurement
        self.step_count = 0
        self.step_times = []
    
    def __call__(
        self, 
        apply_model_func: Callable, 
        kwargs: Dict[str, Any]
    ) -> torch.Tensor:
        """Intercept apply_model, inject patches, dispatch to workers.
        
        This function implements the "injection phase" of the Chained Patch
        strategy by populating transformer_options["patches"] and
        transformer_options["patches_replace"] before the model executes.
        
        Args:
            apply_model_func: Original model.apply_model function (unused in Phase 2.1)
            kwargs: {
                "input": x (latent tensor),
                "timestep": timestep tensor,
                "c": conditioning dict,
                "cond_or_uncond": CFG batch mask (Phase 2.2)
            }
        
        Returns:
            Model output tensor for this step
        """
        # Extract apply_model arguments
        x = kwargs["input"]
        timestep = kwargs["timestep"]
        c = kwargs["c"]
        
        # Start timing this step
        import time
        step_start = time.time()
        self.step_count += 1
        
        # Initialize session logger on first call
        from comfy.parallel_attention.session_logger import SessionLogger
        session_logger = SessionLogger.get_instance()
        if not session_logger.is_active():
            session_logger.start_session()
        
        logging.debug(
            f"{LOG_PREFIX} Intercepted: x.shape={x.shape}, t={timestep}"
        )
        
        # Get or create transformer_options
        transformer_options = c.get("transformer_options", {})
        
        # Create ring_context for workers
        # Workers will populate rank and sp_group when they initialize
        ring_context = {
            "rank": None,  # Will be set by worker
            "world_size": self.world_size,
            "sp_group": None,  # Will be set by worker
            "txt_length": None,  # Will be set by patch_last_double_block
            "pre_single_block_tensor": None  # Bridge the gap at line 182
        }
        
        transformer_options["ring_context"] = ring_context
        c["transformer_options"] = transformer_options
        
        # Serialize for multiprocess dispatch
        logging.warning(f"🚨🚨🚨 [RING] CPU TRANSFER: x.shape={x.shape}, device={x.device} → CPU")
        x_cpu = x.cpu()
        logging.warning(f"🚨🚨🚨 [RING] CPU TRANSFER: timestep.shape={timestep.shape}, device={timestep.device} → CPU")
        timestep_cpu = timestep.cpu()
        worker_args = {
            "x": x_cpu,
            "timestep": timestep_cpu,
            "c": self._serialize_conditioning(c),
            "ring_enabled": True,
            "ring_context": ring_context
        }
        
        result = self.executor.execute_collective(
            "apply_model_step", 
            worker_args
        )
        
        # Return output on original device
        output = result["output"].to(x.device)
        
        # Calculate step timing
        step_end = time.time()
        step_duration = step_end - step_start
        self.step_times.append(step_duration)
        
        # Log per-step timing to session
        avg_it_s = sum(self.step_times) / len(self.step_times)
        session_logger.log(
            f"⚡ [RingAttention] Step {self.step_count}: {step_duration:.2f}s/it (avg: {avg_it_s:.2f}s/it)"
        )
        
        logging.debug(f"{LOG_PREFIX} Returned: output.shape={output.shape}")
        
        return output
    
    def _serialize_conditioning(self, c: dict) -> dict:
        """Move conditioning tensors to CPU for multiprocess serialization.
        
        This is the same pattern as UnifiedParallelWrapper. Recursively
        serializes tensors, dicts, and lists.
        
        Args:
            c: Conditioning dictionary (may contain nested structures)
        
        Returns:
            Serialized conditioning with all tensors on CPU
        """
        result = {}
        for key, value in c.items():
            if isinstance(value, torch.Tensor):
                result[key] = value.cpu()
            elif isinstance(value, dict):
                result[key] = self._serialize_conditioning(value)
            elif isinstance(value, list):
                result[key] = [
                    v.cpu() if isinstance(v, torch.Tensor) else v 
                    for v in value
                ]
            else:
                result[key] = value
        return result
