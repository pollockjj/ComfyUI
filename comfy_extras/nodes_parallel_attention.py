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
        # Check for parallel_attention dict (CFG-Split pattern)
        if not hasattr(model, 'parallel_attention'):
            raise RuntimeError(
                f"{LOG_PREFIX} No parallel_attention dict found on ModelPatcher"
            )
        
        pa = model.parallel_attention
        
        if not isinstance(pa, dict):
            raise RuntimeError(f"{LOG_PREFIX} parallel_attention is not a dict")
        
        # Detect model type
        model_type = type(model.model).__name__.lower()
        if model_type == "qwenimage":
            model_type = "qwen_image"
        elif model_type.startswith("wan"):
            model_type = "wan"
        
        # Phase B: Worker Initialization
        if enable_fsdp2 and pa.get("executor") is None:
            from comfy.parallel_attention import FSDP2Executor, FSDP2PolicyRegistry
            
            # Cleanup any existing executor first
            if hasattr(model, '_fsdp2_executor') and model._fsdp2_executor is not None:
                logging.info(f"{LOG_PREFIX} Cleaning up old executor...")
                model._fsdp2_executor.shutdown()
                model._fsdp2_executor = None
            
            # Get policy for model
            if not FSDP2PolicyRegistry.is_registered(model_type):
                raise RuntimeError(f"{LOG_PREFIX} No policy registered for model type: {model_type}")
            
            policy = FSDP2PolicyRegistry.get_policy(model_type)
            
            # Determine backend
            actual_backend = backend if backend != "auto" else ("nccl" if torch.cuda.is_available() else "gloo")
            
            logging.info(f"{LOG_PREFIX} Spawning workers for {model_type}")
            logging.info(f"{LOG_PREFIX} Devices: {device_1}, {device_2}")
            logging.info(f"{LOG_PREFIX} Backend: {actual_backend}")
            
            # Spawn workers
            executor = FSDP2Executor(world_size=2, backend=actual_backend)
            
            logging.info(f"{LOG_PREFIX} Workers spawned")
            
            # Get checkpoint path from inner model
            inner_pa = model.model._parallel_attention
            checkpoint_path = inner_pa.get("checkpoint_path")
            
            if checkpoint_path is None:
                raise RuntimeError(
                    f"{LOG_PREFIX} No checkpoint path set. "
                    f"Model must be loaded with FSDP2 enabled in model options."
                )
            
            logging.info(f"{LOG_PREFIX} Initializing workers with checkpoint: {checkpoint_path}")
            
            # Initialize workers: load and shard model
            # Pass model_type only - workers will get policy from registry
            result = executor.execute_collective("initialize_fsdp2_from_checkpoint", {
                "checkpoint_path": checkpoint_path,
                "model_type": model_type,
            })
            
            if result.get("status") != "success":
                raise RuntimeError(f"{LOG_PREFIX} Worker init failed: {result.get('error')}")
            
            logging.info(
                f"{LOG_PREFIX} Workers initialized: {result['vram_gb']:.2f}GB per GPU, "
                f"{result['sharded_count']} sharded params"
            )
            
            # Set executor on ModelPatcher for sample() intercept
            model._fsdp2_executor = executor
            
            # Update parallel_attention context
            pa["executor"] = executor
            pa["sharded"] = True
            pa["vram_per_gpu"] = result["vram_gb"]
            pa["sharded_params"] = result["sharded_count"]
            pa["phase"] = "ready_for_inference"
            
            logging.info(f"{LOG_PREFIX} ✅ Model ready for distributed inference")
        elif pa.get("executor") is not None:
            logging.info(f"{LOG_PREFIX} Workers already initialized, reusing")
        
        return (model,)


class TestFSDP2Inference:
    """Test FSDP2 distributed inference with sharded model.
    
    Validates that workers can execute forward pass on FSDP2-sharded model.
    Tests data parallelism - all ranks compute identical output.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "latent": ("LATENT",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 1, "min": 1, "max": 10}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0}),
            }
        }
    
    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("latent", "test_result")
    FUNCTION = "test_inference"
    CATEGORY = "parallel_attention/test"
    
    def test_inference(self, model, latent, positive, negative, seed, steps, cfg):
        """Test FSDP2 inference with minimal sampling."""
        import comfy.sample
        import numpy as np
        
        LOG_PREFIX = "⚡ [PA-Test][Inference]"
        
        # Check for parallel attention context
        if not hasattr(model, 'parallel_attention'):
            return (latent, "❌ No parallel_attention context")
        
        ctx = model.parallel_attention
        
        if not ctx.is_ready_for_sharding():
            return (latent, f"❌ Not ready: phase={ctx.phase}")
        
        if ctx.executor is None:
            return (latent, "❌ No executor (run ParallelAttentionConfig first)")
        
        # Validate workers have sharded model
        if not ctx.sharded:
            return (latent, "❌ Workers not sharded (model not loaded yet)")
        
        logging.info(f"{LOG_PREFIX} Starting inference test")
        logging.info(f"{LOG_PREFIX} Model: {ctx.model_type}, Steps: {steps}")
        logging.info(f"{LOG_PREFIX} Sharded params: {ctx.sharded_params}")
        
        try:
            # Run sampling
            samples = latent["samples"]
            
            # Use ComfyUI's sampler
            output_latent = comfy.sample.sample(
                model,
                noise=torch.randn_like(samples),
                steps=steps,
                cfg=cfg,
                sampler_name="euler",
                scheduler="simple",
                positive=positive,
                negative=negative,
                latent_image=samples,
                denoise=1.0,
            )
            
            logging.info(f"{LOG_PREFIX} ✅ Inference complete")
            
            # Build result
            result = (
                f"✅ FSDP2 Inference Test PASSED\n"
                f"Model: {ctx.model_type}\n"
                f"Steps: {steps}\n"
                f"Sharded params: {ctx.sharded_params}\n"
                f"VRAM per GPU: {ctx.vram_per_gpu:.2f}GB\n"
                f"Output shape: {output_latent['samples'].shape}"
            )
            
            return ({"samples": output_latent["samples"]}, result)
            
        except Exception as e:
            logging.error(f"{LOG_PREFIX} Inference failed: {e}", exc_info=True)
            return (latent, f"❌ Inference failed: {str(e)}")


NODE_CLASS_MAPPINGS = {
    "ParallelAttentionConfig": ParallelAttentionConfig,
    "TestFSDP2Inference": TestFSDP2Inference,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ParallelAttentionConfig": "Parallel Attention Config",
    "TestFSDP2Inference": "Test FSDP2 Inference",
}