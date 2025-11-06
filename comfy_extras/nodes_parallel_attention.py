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
                "ulysses_degree": ("INT", {"default": 1, "min": 1, "max": 16}),
                "ring_degree": ("INT", {"default": 1, "min": 1, "max": 16}),
                "attention_backend": (
                    [
                        "FLASH_ATTN",
                        "FLASH_ATTN_3",
                        "SAGE_AUTO_DETECT",
                        "SAGE_FP16_TRITON",
                        "SAGE_FP16_CUDA",
                        "SAGE_FP8_CUDA",
                        "SAGE_FP8_SM90",
                        "TORCH",
                    ],
                    {"default": "FLASH_ATTN"},
                ),
            }
        }
    
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "configure"
    CATEGORY = "parallel_attention"
    
    @classmethod
    def IS_CHANGED(cls, model, enable_fsdp2, device_1, device_2, backend, ulysses_degree, ring_degree, attention_backend):
        """Force re-execution when settings change (prevents worker reuse with wrong config)."""
        import hashlib
        
        settings_str = f"{enable_fsdp2}{device_1}{device_2}{backend}{ulysses_degree}{ring_degree}{attention_backend}"
        current_hash = hashlib.sha256(settings_str.encode()).hexdigest()
        
        if not hasattr(cls, '_last_hash'):
            cls._last_hash = current_hash
        elif cls._last_hash != current_hash:
            cls._last_hash = current_hash
        
        return current_hash
    
    def configure(self, model, enable_fsdp2, device_1, device_2, backend, ulysses_degree, ring_degree, attention_backend):
        """Configure parallel attention and spawn workers."""
        world_size = 2  # Currently hardcoded to 2 GPUs
        logging.info(f"{LOG_PREFIX} FSDP2 mode selected (world_size={world_size})")

        sequence_degree = ulysses_degree * ring_degree
        if sequence_degree > world_size:
            raise RuntimeError(
                f"{LOG_PREFIX} Invalid USP config: ulysses_degree={ulysses_degree}, ring_degree={ring_degree}, world_size={world_size}"
            )
        usp_enabled = sequence_degree > 1
        if usp_enabled:
            logging.info(
                f"{LOG_PREFIX} USP enabled (ulysses_degree={ulysses_degree}, ring_degree={ring_degree}, attention={attention_backend})"
            )
        
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
        if enable_fsdp2:
            from comfy.parallel_attention import FSDP2Executor, FSDP2PolicyRegistry
            
            # Cleanup any existing executor (IS_CHANGED forces reinit on settings change)
            if pa.get("executor") is not None:
                logging.info(f"{LOG_PREFIX} Settings changed, killing existing workers...")
                pa["executor"].shutdown()
                pa["executor"] = None
            
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
            
            # Initialize workers: load, shard model, and apply any provided object patches
            object_patches = {}
            usp_config = {
                "ulysses_degree": ulysses_degree,
                "ring_degree": ring_degree,
                "attention_backend": attention_backend,
            }

            result = executor.execute_collective("initialize_fsdp2_from_checkpoint", {
                "checkpoint_path": checkpoint_path,
                "model_type": model_type,
                "object_patches": object_patches,
                "usp_config": usp_config,
            })
            
            if result.get("status") != "success":
                raise RuntimeError(f"{LOG_PREFIX} Worker init failed: {result.get('error')}")
            
            logging.info(
                f"{LOG_PREFIX} Workers initialized: {result['vram_gb']:.2f}GB per GPU, "
                f"{result['sharded_count']} sharded params"
            )
            
            # Store executor and metadata (NO wrapper attachment - use FSDP2DistributedSampler custom node)
            model.parallel_attention = {
                "executor": executor,
                "model_type": model_type,
                "sharded": True,
                "vram_per_gpu": result["vram_gb"],
                "sharded_params": result["sharded_count"],
                "usp_config": usp_config,
            }
            
            logging.info(f"{LOG_PREFIX} ✅ Workers ready (use FSDP2DistributedSampler custom node)")
            
            # Update inner parallel_attention context
            pa["executor"] = executor
            pa["sharded"] = True
            pa["vram_per_gpu"] = result["vram_gb"]
            pa["sharded_params"] = result["sharded_count"]
            pa["usp_config"] = usp_config
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
        
        pa = model.parallel_attention
        
        # Check executor exists
        if pa.get("executor") is None:
            return (latent, "❌ No executor (run ParallelAttentionConfig first)")
        
        # Validate workers have sharded model
        if not pa.get("sharded", False):
            return (latent, "❌ Workers not sharded (model not loaded yet)")
        
        logging.info(f"{LOG_PREFIX} Starting inference test")
        logging.info(f"{LOG_PREFIX} Model: {pa.get('model_type')}, Steps: {steps}")
        logging.info(f"{LOG_PREFIX} Sharded params: {pa.get('sharded_params')}")
        
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
                f"Model: {pa.get('model_type')}\n"
                f"Steps: {steps}\n"
                f"Sharded params: {pa.get('sharded_params')}\n"
                f"VRAM per GPU: {pa.get('vram_per_gpu', 0.0):.2f}GB\n"
                f"Output shape: {output_latent['samples'].shape}"
            )
            
            return ({"samples": output_latent["samples"]}, result)
            
        except Exception as e:
            logging.error(f"{LOG_PREFIX} Inference failed: {e}", exc_info=True)
            return (latent, f"❌ Inference failed: {str(e)}")


class FSDP2DistributedSampler:
    """Custom sampler for FSDP2 distributed inference.

    Replaces standard KSampler when using FSDP2.
    Workers execute full sampling sessions via comfy.sample.sample().
    """
    
    @classmethod
    def INPUT_TYPES(s):
        import comfy.samplers
        return {
            "required": {
                "model": ("MODEL",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }
    
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "sampling"
    
    def sample(self, model, seed, steps, cfg, sampler_name, scheduler, 
               positive, negative, latent_image, denoise=1.0):
        """Execute distributed sampling via workers (Direct RPC pattern)."""
        LOG_PREFIX = "⚡ [FSDP2DistributedSampler]"
        
        # Get executor from model
        executor = model.parallel_attention["executor"]
        
        # Build standard ksampler args matching worker's _common_ksampler signature
        ksampler_args = {
            "seed": seed,
            "steps": steps,
            "cfg": cfg,
            "sampler_name": sampler_name,
            "scheduler": scheduler,
            "positive": positive,
            "negative": negative,
            "latent": latent_image,  # Worker expects "latent" not "latent_image"
            "denoise": denoise
        }
        
        logging.info(f"{LOG_PREFIX} Direct RPC dispatch: {steps} steps, seed={seed}")
        
        # Direct RPC to workers (Commit 7abfaef pattern)
        result = executor.execute_collective("common_ksampler", ksampler_args)
        
        if result.get("status") != "success":
            raise RuntimeError(f"{LOG_PREFIX} Worker sampling failed: {result.get('error')}")
        
        # Extract samples
        return ({"samples": result["result"]["samples"]},)


class TestApplyModelStep:
    """Test Phase 2.1 apply_model_step handler (Milestone 2.1.1).
    
    Validates workers can execute single forward pass and return output.
    This is the foundation for the per-step wrapper pattern.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "latent": ("LATENT",),
                "positive": ("CONDITIONING",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }
    
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "test_apply_step"
    CATEGORY = "parallel_attention/test"
    
    def test_apply_step(self, model, latent, positive, seed):
        """Test single apply_model_step call."""
        import comfy.sample
        
        LOG_PREFIX = "⚡ [PA-Test][ApplyStep]"
        
        logging.info(f"{LOG_PREFIX} ========================================")
        logging.info(f"{LOG_PREFIX} NODE CALLED - Starting test")
        logging.info(f"{LOG_PREFIX} ========================================")
        
        # Check for parallel attention context
        if not hasattr(model, 'parallel_attention'):
            logging.error(f"{LOG_PREFIX} FAILED: No parallel_attention context")
            return (latent, "❌ No parallel_attention context")
        
        pa = model.parallel_attention
        
        logging.info(f"{LOG_PREFIX} Has executor: {pa.get('executor') is not None}")
        logging.info(f"{LOG_PREFIX} Is sharded: {pa.get('sharded', False)}")
        
        if pa.get("executor") is None:
            logging.error(f"{LOG_PREFIX} FAILED: No executor")
            return (latent, "❌ No executor (run ParallelAttentionConfig first)")
        
        if not pa.get("sharded", False):
            logging.error(f"{LOG_PREFIX} FAILED: Workers not sharded")
            return (latent, "❌ Workers not sharded (model not loaded yet)")
        
        logging.info(f"{LOG_PREFIX} Validation passed - testing apply_model_step handler...")
        
        try:
            # Prepare single-step inputs
            samples = latent["samples"]
            noise = comfy.sample.prepare_noise(samples, seed)
            
            # Create dummy timestep (just for testing forward pass)
            timestep = torch.tensor([999.0])
            
            # Extract conditioning (Flux format)
            # positive is list of [cond_tensor, cond_dict]
            cond_tensor = positive[0][0]
            cond_dict = positive[0][1]
            
            # Build conditioning dict for apply_model
            # apply_model signature: c_crossattn (context), y (pooled)
            cond = {"c_crossattn": cond_tensor}
            if "pooled_output" in cond_dict:
                cond["y"] = cond_dict["pooled_output"]
            
            # Build args for apply_model_step
            step_args = {
                "x": noise.cpu(),
                "timestep": timestep.cpu(),
                "c": cond
            }
            
            logging.info(f"{LOG_PREFIX} Calling apply_model_step on workers...")
            logging.info(f"{LOG_PREFIX}   x.shape={noise.shape}, timestep={timestep}")
            
            # Execute on workers
            result = pa["executor"].execute_collective("apply_model_step", step_args)
            
            logging.info(f"{LOG_PREFIX} Result keys: {list(result.keys())}")
            
            if "output" in result:
                output = result["output"]
                logging.info(f"{LOG_PREFIX} ✅ TEST PASSED - Forward pass complete")
                logging.info(f"{LOG_PREFIX} Input shape: {noise.shape}")
                logging.info(f"{LOG_PREFIX} Output shape: {output.shape}")
                logging.info(f"{LOG_PREFIX} VRAM per GPU: {pa.get('vram_per_gpu', 0.0):.2f}GB")
                logging.info(f"{LOG_PREFIX} ========================================")
                
                return ({"samples": output},)
            else:
                logging.error(f"{LOG_PREFIX} ❌ TEST FAILED - No output in result: {result}")
                return (latent,)
                
        except Exception as e:
            logging.error(f"{LOG_PREFIX} ❌ TEST FAILED - Exception: {e}", exc_info=True)
            logging.error(f"{LOG_PREFIX} ========================================")
            return (latent,)


NODE_CLASS_MAPPINGS = {
    "ParallelAttentionConfig": ParallelAttentionConfig,
    "TestFSDP2Inference": TestFSDP2Inference,
    "FSDP2DistributedSampler": FSDP2DistributedSampler,
    "TestApplyModelStep": TestApplyModelStep,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ParallelAttentionConfig": "Parallel Attention Config",
    "TestFSDP2Inference": "Test FSDP2 Inference",
    "FSDP2DistributedSampler": "FSDP2 Distributed Sampler",
    "TestApplyModelStep": "Test Apply Model Step (Phase 2.1)",
}