"""model_function_wrapper for FSDP2 distributed inference.

Manages torch.distributed ProcessGroup lifecycle and dispatches
apply_model calls to FSDP2 workers on a per-step basis.

Pattern: Environment Manager (from core_plan.md)
"""

import logging
import torch
import torch.distributed as dist

LOG_PREFIX = "⚡ [DistributedWrapper]"


class DistributedEnvironmentWrapper:
    """Wrapper for standard KSampler to dispatch to FSDP2 workers.
    
    Lifecycle:
    1. Each step: RPC to workers for apply_model
    2. Cleanup: Log statistics
    
    Note: Parent process does NOT manage ProcessGroup.
    Workers already have their own ProcessGroup initialized during spawn.
    This wrapper just dispatches RPC calls to workers per-step.
    
    Usage:
        wrapper = DistributedEnvironmentWrapper(executor, world_size=2, backend="nccl")
        model.set_model_unet_function_wrapper(wrapper)
        # Standard KSampler now works with FSDP2
    """
    
    def __init__(self, executor, world_size=2, backend="nccl"):
        """Initialize wrapper.
        
        Args:
            executor: FSDP2Executor instance (manages worker processes)
            world_size: Number of GPUs (2 for current implementation)
            backend: torch.distributed backend (for documentation only)
        """
        self.executor = executor
        self.world_size = world_size
        self.backend = backend
        self.step_count = 0
        
        # Timing instrumentation
        self.total_rpc_time = 0.0
        self.rpc_calls = 0
    
    def __call__(self, apply_model_func, kwargs):
        """Called once per sampling step by ComfyUI sampler.
        
        This is the model_function_wrapper signature.
        
        Args:
            apply_model_func: model.apply_model (ignored - workers use their own)
            kwargs: {
                "input": noisy latent tensor [B, C, H, W],
                "timestep": timestep tensor [B] or [B, 1],
                "c": conditioning dict {
                    "c_crossattn": context embeddings,
                    "y": pooled embeddings (optional),
                    "guidance": guidance scale (optional),
                    ...
                },
                "cond_or_uncond": batch indices (optional, for CFG)
            }
        
        Returns:
            torch.Tensor: predicted noise (same shape as input)
        """
        import time
        
        # Parent process doesn't need ProcessGroup - workers already have it
        # Just dispatch via RPC
        
        self.step_count += 1
        
        # Time RPC call
        rpc_start = time.perf_counter()
        
        # Serialize kwargs for RPC
        step_args = self._serialize_kwargs(kwargs)
        
        logging.debug(f"{LOG_PREFIX} Step {self.step_count}: dispatching to workers")
        
        # Execute on workers via RPC
        result = self.executor.execute_collective("apply_model_step", step_args)
        
        rpc_elapsed = (time.perf_counter() - rpc_start) * 1000  # ms
        self.total_rpc_time += rpc_elapsed
        self.rpc_calls += 1
        
        logging.debug(
            f"{LOG_PREFIX} Step {self.step_count} complete: {rpc_elapsed:.2f}ms RPC "
            f"(avg: {self.total_rpc_time/self.rpc_calls:.2f}ms)"
        )
        
        # Check for errors (worker returns {"output": tensor} or {"error": str})
        if "error" in result:
            error = result["error"]
            logging.error(f"{LOG_PREFIX} Worker failed: {error}")
            self.cleanup()
            raise RuntimeError(f"{LOG_PREFIX} Worker failed: {error}")
        
        if "output" not in result:
            logging.error(f"{LOG_PREFIX} Worker returned invalid result: {list(result.keys())}")
            self.cleanup()
            raise RuntimeError(f"{LOG_PREFIX} Worker returned no output")
        
        # Extract output tensor (from rank 0 worker)
        output = result["output"]
        
        # Move to parent device (cuda:0)
        if not output.is_cuda or output.device.index != 0:
            output = output.to("cuda:0")
        
        return output
    
    def _serialize_kwargs(self, kwargs):
        """Serialize kwargs for RPC to workers.
        
        Moves tensors to CPU for pickling.
        
        Args:
            kwargs: apply_model kwargs dict
        
        Returns:
            Serialized dict ready for RPC
        """
        step_args = {
            "x": kwargs["input"].cpu(),  # Worker expects "x" not "input"
            "timestep": kwargs["timestep"].cpu(),
            "c": self._serialize_conditioning(kwargs["c"]),
        }
        
        # Optional: cond_or_uncond (for CFG batching)
        if "cond_or_uncond" in kwargs:
            cond_or_uncond = kwargs["cond_or_uncond"]
            if isinstance(cond_or_uncond, torch.Tensor):
                step_args["cond_or_uncond"] = cond_or_uncond.cpu()
            else:
                step_args["cond_or_uncond"] = cond_or_uncond
        
        return step_args
    
    def _serialize_conditioning(self, c):
        """Move conditioning tensors to CPU for serialization.
        
        Args:
            c: Conditioning dict (Flux format)
        
        Returns:
            CPU-serialized conditioning dict
        """
        c_cpu = {}
        for k, v in c.items():
            if isinstance(v, torch.Tensor):
                c_cpu[k] = v.cpu()
            elif isinstance(v, dict):
                # Nested dict (e.g., transformer_options)
                c_cpu[k] = self._serialize_conditioning(v)
            else:
                c_cpu[k] = v
        return c_cpu
    
    def cleanup(self):
        """Log cleanup statistics.
        
        Parent process doesn't manage ProcessGroup - workers do.
        This just logs RPC statistics.
        """
        if self.rpc_calls > 0:
            logging.info(
                f"{LOG_PREFIX} Session complete: {self.rpc_calls} RPC calls, "
                f"avg {self.total_rpc_time/self.rpc_calls:.2f}ms/call, "
                f"total {self.total_rpc_time:.0f}ms overhead"
            )
        
        # Reset counters
        self.step_count = 0
        self.total_rpc_time = 0.0
        self.rpc_calls = 0
    
    def __del__(self):
        """Ensure cleanup on garbage collection."""
        self.cleanup()
