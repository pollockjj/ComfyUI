"""Parallel Attention UNET Loader with FSDP sharding.

Drop-in replacement for UNETLoader that automatically:
- Detects available GPUs
- Initializes distributed workers
- Loads model with FSDP sharding across 2 GPUs
"""

import torch
import logging
import folder_paths
import comfy.sd
import comfy.model_management

LOG_PREFIX = "⚡ [Parallel-Attention]"


class UnetLoaderParallelAttention:
    """UNET Loader - 100% standard ComfyUI loading.
    
    Identical to UNETLoader. Calls comfy.sd.load_diffusion_model()
    with standard lifecycle. When --use-parallel-attention flag is set,
    meta device copy is created automatically in comfy/sd.py.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "unet_name": (folder_paths.get_filename_list("diffusion_models"),),
            }
        }
    
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_unet_parallel"
    CATEGORY = "parallel_attention"
    
    def load_unet_parallel(self, unet_name):
        """Load UNET using 100% standard ComfyUI loading pipeline.
        
        This is IDENTICAL to UNETLoader - calls comfy.sd.load_diffusion_model()
        with no model_options, using the complete standard lifecycle.
        """
        unet_path = folder_paths.get_full_path_or_raise("diffusion_models", unet_name)
        model = comfy.sd.load_diffusion_model(unet_path, model_options={})
        
        logging.info(f"{LOG_PREFIX} [Loader] Loaded: {type(model.model).__name__}")
        logging.info(f"{LOG_PREFIX} [Loader] Device: {model.load_device}")
        logging.info(f"{LOG_PREFIX} [Loader] Size: {model.model_size() / (1024**3):.2f}GB")
        
        return (model,)


"""Test node for distributed runtime."""

import logging
import folder_paths

from comfy.parallel_attention.phase_one_unit_tests import run_phase1a_tests

LOG_PREFIX = "⚡ [Parallel-Attention]"

class ParallelAttentionUnitTests:
    """Phase 1A: Core Salvage Unit Tests.
    
    Tests ONLY the 4 perfect salvaged files:
    - FSDP2Executor (DeviceMesh-based executor)
    - parallel_state (DeviceMesh management)
    - FSDP2PolicyRegistry (policy system)
    - ShardingConfig/BlockConfig (data structures)
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Phase 1A: Core Salvage Tests
                "test_executor_spawn": ("BOOLEAN", {"default": False}),
                "test_devicemesh_init": ("BOOLEAN", {"default": False}),
                "test_worker_communication": ("BOOLEAN", {"default": False}),
                "test_policy_registry": ("BOOLEAN", {"default": False}),
                
                # Convenience
                "run_all_phase1a": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    FUNCTION = "run_tests"
    CATEGORY = "parallel_attention"
    
    def run_tests(self, test_executor_spawn, test_devicemesh_init,
                  test_worker_communication, test_policy_registry,
                  run_all_phase1a):
        """Run Phase 1A unit tests for core salvaged infrastructure.
        
        Tests ONLY the 4 perfect files with NO modifications.
        """
        
        logging.info(f"{LOG_PREFIX} [Test] ══════════════════════════════════════════════════════")
        logging.info(f"{LOG_PREFIX} [Test] PHASE 1A: CORE SALVAGE UNIT TESTS")
        logging.info(f"{LOG_PREFIX} [Test] ══════════════════════════════════════════════════════")
        
        summary = run_phase1a_tests(
            test_executor_spawn=test_executor_spawn,
            test_devicemesh_init=test_devicemesh_init,
            test_worker_communication=test_worker_communication,
            test_policy_registry=test_policy_registry,
            run_all_phase1a=run_all_phase1a,
        )

        return (summary,)

NODE_CLASS_MAPPINGS = {
    "UnetLoaderParallelAttention": UnetLoaderParallelAttention,
    "ParallelAttentionUnitTests": ParallelAttentionUnitTests
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UnetLoaderParallelAttention": "UNET Loader (Parallel Attention)",
    "ParallelAttentionUnitTests": "Parallel Attention Unit Tests"
}