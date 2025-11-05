"""Unified wrapper for per-step distributed execution (Phase 2).

Implements "Smart Parent/Dumb Worker" pattern.
Stateless wrapper intercepts apply_model() per-step from standard KSampler.
"""

import torch
import logging
from typing import Any, Dict, Callable

LOG_PREFIX = "⚡ [UnifiedWrapper]"


class UnifiedParallelWrapper:
    """Stateless per-step wrapper for FSDP2 distributed inference.
    
    Intercepts apply_model() calls from ComfyUI samplers, dispatches to workers
    as stateless RPC. Enables standard KSampler compatibility.
    
    Pattern: Smart Parent (runs sampling loop) / Dumb Worker (RPC server)
    """
    
    def __init__(self, executor, model_type: str, enable_cfg_split: bool = False):
        """Initialize wrapper.
        
        Args:
            executor: FSDP2Executor with worker pool
            model_type: Model type (flux, wan, etc.)
            enable_cfg_split: Enable CFG-Split parallelism (Phase 2.2)
        """
        self.executor = executor
        self.model_type = model_type
        self.enable_cfg_split = enable_cfg_split
        
        logging.info(f"{LOG_PREFIX} Initialized for {model_type}")
        logging.info(f"{LOG_PREFIX} CFG-Split: {'enabled' if enable_cfg_split else 'disabled'}")
    
    def __call__(self, apply_model_func: Callable, kwargs: Dict[str, Any]) -> torch.Tensor:
        """Intercept apply_model call, dispatch to workers.
        
        Called per-step by ComfyUI samplers. Stateless - no session tracking.
        
        Args:
            apply_model_func: Original apply_model function (unused in Phase 2.1)
            kwargs: {
                "input": x,                    # Latent tensor
                "timestep": timestep,          # Timestep tensor
                "c": conditioning_dict,        # c_concat, c_crossattn, etc.
                "cond_or_uncond": batch_mask   # CFG batch mask (Phase 2.2)
            }
        
        Returns:
            torch.Tensor: Model output for this step
        """
        # Extract apply_model arguments from kwargs dict
        x = kwargs["input"]
        timestep = kwargs["timestep"]
        c = kwargs["c"]
        
        logging.debug(f"{LOG_PREFIX} Intercepted: x.shape={x.shape}, t={timestep}")
        
        # Phase 2.1: No CFG-Split yet, just dispatch to workers
        # Phase 2.2: Will split cond/uncond here using cond_or_uncond mask
        
        # Prepare args for workers (serialize tensors to CPU for multiprocess)
        logging.warning(f"🚨🚨🚨 CPU TRANSFER: x.shape={x.shape}, device={x.device} → CPU")
        x_cpu = x.cpu()
        logging.warning(f"🚨🚨🚨 CPU TRANSFER: timestep.shape={timestep.shape}, device={timestep.device} → CPU")
        timestep_cpu = timestep.cpu()
        worker_args = {
            "x": x_cpu,
            "timestep": timestep_cpu,
            "c": self._serialize_conditioning(c)
        }
        
        # Dispatch to workers (blocking RPC call)
        result = self.executor.execute_collective("apply_model_step", worker_args)
        
        # Extract output from rank 0 worker
        output = result["output"]
        
        # Move back to original device (samplers expect this)
        if hasattr(x, 'device'):
            output = output.to(x.device)
        
        logging.debug(f"{LOG_PREFIX} Returned: output.shape={output.shape}")
        
        return output
    
    def _serialize_conditioning(self, c: dict) -> dict:
        """Move conditioning tensors to CPU for serialization.
        
        Args:
            c: Conditioning dict (c_concat, c_crossattn, etc.)
        
        Returns:
            dict: Same structure with all tensors on CPU
        """
        result = {}
        for key, value in c.items():
            if isinstance(value, torch.Tensor):
                logging.warning(f"🚨🚨🚨 CPU TRANSFER: conditioning[{key}].shape={value.shape}, device={value.device} → CPU")
                result[key] = value.cpu()
            elif isinstance(value, dict):
                result[key] = self._serialize_conditioning(value)
            elif isinstance(value, list):
                result[key] = [v.cpu() if isinstance(v, torch.Tensor) else v for v in value]
            else:
                result[key] = value
        return result
