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
    """UNET Loader with Parallel Attention (FSDP sharding across 2 GPUs).
    
    Automatically initializes distributed environment and loads model
    with FSDP sharding if 2+ GPUs available. Otherwise falls back to
    standard loading.
    
    Device selection allows choosing which 2 GPUs to use for sharding.
    Backend is automatically selected (NCCL with GLOO fallback).
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
    
    def __init__(self):
        self.executor = None
        self.is_initialized = False
    
    def _initialize_distributed(self):
        """Initialize distributed environment."""
        from comfy.parallel_attention.executor import MultiprocExecutor
        
        logging.info(f"{LOG_PREFIX} [Loader] Initializing distributed on 2 GPUs")
        
        # World size is always 2 for parallel attention
        world_size = 2
        
        # Backend is auto-selected (NCCL with GLOO fallback)
        backend = "auto"
        
        # Create executor - devices auto-assigned as cuda:0 and cuda:1
        self.executor = MultiprocExecutor(world_size=world_size, backend=backend)
        
        self.is_initialized = True
        logging.info(f"{LOG_PREFIX} [Loader] Distributed executor initialized")
    
    def load_unet_parallel(self, unet_name):
        """Load UNET with FSDP sharding across 2 GPUs.
        
        Args:
            unet_name: Model filename
        
        Returns:
            Tuple of (FSDPModelPatcher,) ready for use
        """
        # Check if we have 2+ GPUs
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            raise RuntimeError(
                f"{LOG_PREFIX} [Loader] Parallel Attention requires 2+ CUDA devices. "
                f"Found: {torch.cuda.device_count() if torch.cuda.is_available() else 0}"
            )
        
        # Initialize distributed if not already done
        if not self.is_initialized:
            self._initialize_distributed()
        
        # Build model options
        model_options = {
            'fsdp': {
                'enabled': True,
                'cpu_offload': False
            }
        }
        
        unet_path = folder_paths.get_full_path_or_raise("diffusion_models", unet_name)
        
        logging.info(f"{LOG_PREFIX} [Loader] Loading {unet_name} with FSDP across 2 GPUs")
        
        # Load model in workers (where torch.distributed is initialized)
        results = self.executor.execute_collective("load_fsdp_model", {
            "unet_path": unet_path,
            "model_options": model_options
        })
        
        # Check if loading succeeded
        if not results.get("success", False):
            error = results.get("error", "Unknown error")
            raise RuntimeError(f"{LOG_PREFIX} [Loader] Model loading failed: {error}")
        
        logging.info(
            f"{LOG_PREFIX} [Loader] Model loaded successfully: "
            f"type={results.get('model_type', 'Unknown')}, "
            f"fsdp={results.get('is_fsdp', False)}"
        )
        
        # For now, we can't return the actual model object from workers
        # This is a limitation - the model lives in worker processes
        # We need to design a different pattern for this
        
        # TODO: Design pattern for accessing worker-resident models
        # Options:
        # 1. Model proxy object that forwards calls to workers
        # 2. Pull model back to main process (defeats purpose of FSDP)
        # 3. Run entire sampling in workers (requires bigger refactor)
        
        raise NotImplementedError(
            f"{LOG_PREFIX} [Loader] Model loading successful in workers, but "
            "ComfyUI execution model requires models in main process. "
            "This loader demonstrates FSDP loading works - full integration requires "
            "architectural changes to run sampling in worker processes."
        )
    
    def __del__(self):
        """Cleanup: shutdown executor when node is destroyed."""
        if self.executor is not None and self.is_initialized:
            logging.info(f"{LOG_PREFIX} [Loader] Shutting down distributed executor")
            self.executor.shutdown()
            self.executor = None
            self.is_initialized = False


NODE_CLASS_MAPPINGS = {
    "UnetLoaderParallelAttention": UnetLoaderParallelAttention
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UnetLoaderParallelAttention": "UNET Loader (Parallel Attention)"
}
"""Test node for distributed runtime."""

import torch
import torch.multiprocessing as mp
import logging
import folder_paths

LOG_PREFIX = "⚡ [Parallel-Attention]"

class TestDistributedRuntime:
    """Test node for distributed multiprocess executor."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "world_size": ("INT", {"default": 2, "min": 1, "max": 8}),
                "backend": (["auto", "nccl", "gloo"],),
                "test_type": (["basic", "devicemesh", "fsdp_policy", "fsdp_load", "all"],),
            },
            "optional": {
                "unet_name": (folder_paths.get_filename_list("diffusion_models"),),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    FUNCTION = "test_executor"
    CATEGORY = "parallel_attention"
    
    def test_executor(self, world_size, backend, test_type, unet_name=None):
        """Test the distributed executor."""
        
        # Set multiprocessing start method
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass  # Already set
        
        from comfy.parallel_attention import MultiprocExecutor
        
        logging.info(f"{LOG_PREFIX} [Test] Starting test: world_size={world_size}, backend={backend}, test_type={test_type}")
        
        # Handle FSDP load test separately (returns MODEL output)
        if test_type == "fsdp_load":
            if unet_name is None:
                return ("SKIP: fsdp_load test requires unet_name to be provided",)
            
            try:
                logging.info(f"{LOG_PREFIX} [Test] Test 6: FSDP Model Loading")
                logging.info(f"{LOG_PREFIX} [Test] ╔═══════════════════════════════════════════════════════╗")
                logging.info(f"{LOG_PREFIX} [Test] ║           FSDP Model Loading Test                     ║")
                logging.info(f"{LOG_PREFIX} [Test] ╚═══════════════════════════════════════════════════════╝")
                
                # Start executor
                executor = MultiprocExecutor(world_size=world_size, backend=backend)
                
                import comfy.sd
                import os
                
                unet_path = folder_paths.get_full_path("diffusion_models", unet_name)
                
                if not os.path.exists(unet_path):
                    executor.shutdown()
                    return (f"FAIL: Model file not found: {unet_path}",)
                
                logging.info(f"{LOG_PREFIX} [Test] Loading {unet_name} with FSDP")
                
                # Load with FSDP enabled - must run in workers where torch.distributed is initialized
                model_options = {
                    'fsdp': {
                        'enabled': True,
                        'cpu_offload': False
                    }
                }
                
                # Call load_fsdp_model on workers
                results = executor.run_on_all_workers(
                    method="load_fsdp_model",
                    unet_path=unet_path,
                    model_options=model_options
                )
                
                # Check results from all workers
                all_success = all(r.get("success", False) for r in results)
                
                if not all_success:
                    errors = [r.get("error", "Unknown error") for r in results if not r.get("success", False)]
                    executor.shutdown()
                    return (f"FAIL: Model loading failed: {errors[0]}",)
                
                # Get model info from first worker
                model_info = results[0]
                
                logging.info(f"{LOG_PREFIX} [Test] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
                logging.info(f"{LOG_PREFIX} [Test] FSDP Load test passed:")
                logging.info(f"{LOG_PREFIX} [Test]   Model: {unet_name}")
                logging.info(f"{LOG_PREFIX} [Test]   Type: {model_info.get('model_type', 'Unknown')}")
                logging.info(f"{LOG_PREFIX} [Test]   FSDP wrapped: {model_info.get('is_fsdp', False)}")
                logging.info(f"{LOG_PREFIX} [Test]   Workers: {len(results)}")
                
                executor.shutdown()
                return (f"PASS: FSDP model loaded successfully on {len(results)} workers",)
                
                logging.info(f"{LOG_PREFIX} [Test] FSDP Load test passed:")
                logging.info(f"{LOG_PREFIX} [Test]   Model: {unet_name}")
                logging.info(f"{LOG_PREFIX} [Test]   Detected type: {detected_type}")
                logging.info(f"{LOG_PREFIX} [Test]   Policy retrieved: {policy_retrieved}")
                logging.info(f"{LOG_PREFIX} [Test]   FSDP imports OK: {fsdp_imports_ok}")
                logging.info(f"{LOG_PREFIX} [Test]   Params: {result.get('params', 0)/1e9:.2f}B")
                
                # Shutdown executor
                executor.shutdown()
                
                success_msg = (
                    f"{LOG_PREFIX} [Test] PASS: FSDP Load test passed!\n"
                    f"{LOG_PREFIX} [Test] Model: {unet_name}\n"
                    f"{LOG_PREFIX} [Test] Detected type: {detected_type}\n"
                    f"{LOG_PREFIX} [Test] Policy retrieved: {policy_retrieved}\n"
                    f"{LOG_PREFIX} [Test] FSDP imports: OK\n"
                    f"{LOG_PREFIX} [Test] Workers tested: {world_size}\n"
                    f"{LOG_PREFIX} [Test] Model size: {result.get('params', 0)/1e9:.2f}B params"
                )
                
                logging.info(success_msg)
                return (success_msg,)
                
            except Exception as e:
                import traceback
                error_msg = f"FAIL: {type(e).__name__}: {e}\n{traceback.format_exc()}"
                logging.error(f"{LOG_PREFIX} [Test] {error_msg}")
                if 'executor' in locals():
                    executor.shutdown()
                return (error_msg,)
        
        # Original tests (no MODEL output)
        try:
            # Test 1: Spawn and echo (always run)
            logging.info(f"{LOG_PREFIX} [Test] Test 1: Spawn workers and echo RPC")
            executor = MultiprocExecutor(world_size=world_size, backend=backend)
            
            # Test echo
            test_message = "hello from comfy"
            result = executor.execute_collective("echo", {"message": test_message})
            
            if result != test_message:
                executor.shutdown()
                return (f"FAIL: Echo test failed. Expected '{test_message}', got '{result}'",)
            
            logging.info(f"{LOG_PREFIX} [Test] Echo test passed: '{result}'")
            
            # Test 2: Multiple RPCs (if basic or all)
            if test_type in ["basic", "all"]:
                logging.info(f"{LOG_PREFIX} [Test] Test 2: Multiple sequential RPCs")
                for i in range(5):
                    message = f"message_{i}"
                    result = executor.execute_collective("echo", {"message": message})
                    if result != message:
                        executor.shutdown()
                        return (f"FAIL: RPC {i} failed",)
                
                logging.info(f"{LOG_PREFIX} [Test] Multiple RPC test passed")
            
            # Test 3: Collective operation (if basic or all, and CUDA available)
            if test_type in ["basic", "all"]:
                if torch.cuda.is_available() and executor.backend == "nccl":
                    logging.info(f"{LOG_PREFIX} [Test] Test 3: torch.distributed collective (all_reduce)")
                    logging.info(f"{LOG_PREFIX} [Test] ─────────────────────────────────────────────────────")
                    result = executor.execute_collective("allreduce_test", {})
                    logging.info(f"{LOG_PREFIX} [Test] ─────────────────────────────────────────────────────")
                    expected = sum(range(world_size))
                    
                    if result != expected:
                        executor.shutdown()
                        return (f"FAIL: all_reduce failed. Expected {expected}, got {result}",)
                    
                    logging.info(f"{LOG_PREFIX} [Test] Collective test passed: result={result}")
                else:
                    logging.info(f"{LOG_PREFIX} [Test] Test 3: Skipped (CUDA not available or not using NCCL)")
            
            # Test 4: DeviceMesh integration (if devicemesh or all)
            if test_type in ["devicemesh", "all"]:
                if torch.cuda.is_available() and executor.backend == "nccl":
                    logging.info(f"{LOG_PREFIX} [Test] Test 4: DeviceMesh topology and SP collective")
                    logging.info(f"{LOG_PREFIX} [Test] ═════════════════════════════════════════════════════")
                    result = executor.execute_collective("devicemesh_test", {})
                    logging.info(f"{LOG_PREFIX} [Test] ═════════════════════════════════════════════════════")
                    
                    # Validate mesh structure
                    mesh_shape = result.get("mesh_shape", [])
                    expected_shape = [1, world_size]  # [dp_size=1, sp_size=world_size]
                    
                    if mesh_shape != expected_shape:
                        executor.shutdown()
                        return (f"FAIL: Mesh shape mismatch. Expected {expected_shape}, got {mesh_shape}",)
                    
                    # Validate all_gather result
                    gathered = result.get("gathered", [])
                    expected_gathered = [float(i) for i in range(world_size)]
                    
                    if gathered != expected_gathered:
                        executor.shutdown()
                        return (f"FAIL: SP all_gather mismatch. Expected {expected_gathered}, got {gathered}",)
                    
                    logging.info(f"{LOG_PREFIX} [Test] DeviceMesh test passed:")
                    logging.info(f"{LOG_PREFIX} [Test]   Mesh shape: {mesh_shape} (dp=1, sp={world_size})")
                    logging.info(f"{LOG_PREFIX} [Test]   SP rank {result['sp_rank']}/{result['sp_size']}, DP rank {result['dp_rank']}/{result['dp_size']}")
                    logging.info(f"{LOG_PREFIX} [Test]   SP all_gather result: {gathered}")
                else:
                    logging.info(f"{LOG_PREFIX} [Test] Test 4: Skipped (CUDA not available or not using NCCL)")
            
            # Test 5: FSDP Policy Registry (if fsdp_policy or all)
            if test_type in ["fsdp_policy", "all"] or (test_type == "all" and unet_name):
                logging.info(f"{LOG_PREFIX} [Test] Test 5: FSDP Policy Registry")
                logging.info(f"{LOG_PREFIX} [Test] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
                result = executor.execute_collective("test_fsdp_policy", {"model_name": "flux"})
                logging.info(f"{LOG_PREFIX} [Test] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
                
                # Validate flux policy registered
                if not result.get("is_registered", False):
                    executor.shutdown()
                    return (f"FAIL: Flux policy not registered. Available: {result.get('available_policies', [])}",)
                
                # Validate policy is callable
                if not result.get("policy_callable", False):
                    executor.shutdown()
                    return (f"FAIL: Flux policy not callable. Type: {result.get('policy_type')}",)
                
                # Check expected policies registered
                available = result.get("available_policies", [])
                expected_policies = ["flux", "qwen_image", "wan"]
                
                missing = [p for p in expected_policies if p not in available]
                if missing:
                    executor.shutdown()
                    return (f"FAIL: Missing policies: {missing}. Available: {available}",)
                
                logging.info(f"{LOG_PREFIX} [Test] FSDP Policy test passed:")
                logging.info(f"{LOG_PREFIX} [Test]   Model: {result['model_name']}")
                logging.info(f"{LOG_PREFIX} [Test]   Registered: {result['is_registered']}")
                logging.info(f"{LOG_PREFIX} [Test]   Available policies: {available}")
                logging.info(f"{LOG_PREFIX} [Test]   Policy type: {result['policy_type']}")
                logging.info(f"{LOG_PREFIX} [Test]   Policy callable: {result['policy_callable']}")
            
            # Test 6: FSDP Load (if all AND unet_name provided)
            if test_type == "all" and unet_name:
                logging.info(f"{LOG_PREFIX} [Test] Test 6: FSDP Model Loading")
                logging.info(f"{LOG_PREFIX} [Test] ╔═══════════════════════════════════════════════════════╗")
                logging.info(f"{LOG_PREFIX} [Test] ║           FSDP Model Loading Test                     ║")
                logging.info(f"{LOG_PREFIX} [Test] ╚═══════════════════════════════════════════════════════╝")
                
                import os
                
                unet_path = folder_paths.get_full_path("diffusion_models", unet_name)
                
                if not os.path.exists(unet_path):
                    executor.shutdown()
                    return (f"FAIL: Model file not found: {unet_path}",)
                
                logging.info(f"{LOG_PREFIX} [Test] Loading {unet_name} with FSDP on workers")
                
                # Load model in workers (where torch.distributed is initialized)
                result = executor.execute_collective("load_fsdp_model", {
                    "unet_path": unet_path,
                    "model_options": {
                        'fsdp': {
                            'enabled': True,
                            'cpu_offload': False
                        }
                    }
                })
                
                success = result.get("success", False)
                error = result.get("error", None)
                
                if not success:
                    executor.shutdown()
                    return (f"FAIL: FSDP model loading failed: {error}",)
                
                logging.info(f"{LOG_PREFIX} [Test] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
                logging.info(f"{LOG_PREFIX} [Test] FSDP Load test passed:")
                logging.info(f"{LOG_PREFIX} [Test]   Model: {unet_name}")
                logging.info(f"{LOG_PREFIX} [Test]   FSDP wrapped: {result.get('is_fsdp_wrapped', False)}")
                logging.info(f"{LOG_PREFIX} [Test]   Shard factor: {result.get('shard_factor', 0)}")
            
            # Shutdown
            logging.info(f"{LOG_PREFIX} [Test] Shutting down executor")
            executor.shutdown()
            
            success_msg = (
                f"{LOG_PREFIX} [Test] PASS: All tests passed!\n"
                f"{LOG_PREFIX} [Test] world_size={world_size}\n"
                f"{LOG_PREFIX} [Test] backend={backend}\n"
                f"{LOG_PREFIX} [Test] test_type={test_type}\n"
                f"{LOG_PREFIX} [Test] CUDA available: {torch.cuda.is_available()}"
            )
            logging.info(success_msg)
            
            return (success_msg,)
            
        except Exception as e:
            import traceback
            error_msg = f"FAIL: {type(e).__name__}: {e}\n{traceback.format_exc()}"
            logging.error(f"{LOG_PREFIX} [Test] {error_msg}")
            return (error_msg,)

NODE_CLASS_MAPPINGS = {
    "UnetLoaderParallelAttention": UnetLoaderParallelAttention,
    "TestDistributedRuntime": TestDistributedRuntime
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UnetLoaderParallelAttention": "UNET Loader (Parallel Attention)",
    "TestDistributedRuntime": "Test Distributed Runtime"
}


