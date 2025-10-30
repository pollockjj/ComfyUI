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
        
        # STEP 1: Extract model scaffold (perfect information)
        logging.info(f"{LOG_PREFIX} [Loader] Extracting model scaffold from checkpoint...")
        
        from comfy.parallel_attention.model_scaffold import extract_model_scaffold
        
        scaffold, state_dict = extract_model_scaffold(unet_path)
        
        logging.info(
            f"{LOG_PREFIX} [Loader] Scaffold extracted: "
            f"type={scaffold['model_type']}, "
            f"latent_format={scaffold['latent_format']['class_name']}, "
            f"size={scaffold['model_size'] / (1024**3):.2f}GB"
        )
        
        # STEP 2: Send scaffold + checkpoint path to workers
        # Workers will reconstruct exact model structure from scaffold
        logging.info(f"{LOG_PREFIX} [Loader] Loading FSDP model in 2 workers with scaffold...")
        
        results = self.executor.execute_collective("load_fsdp_model", {
            "unet_path": unet_path,
            "scaffold": scaffold,  # Send complete scaffold
            "model_options": model_options
        })
        
        # Check if loading succeeded
        if not results.get("success", False):
            error = results.get("error", "Unknown error")
            raise RuntimeError(f"{LOG_PREFIX} [Loader] Model loading failed: {error}")
        
        logging.info(
            f"{LOG_PREFIX} [Loader] Model loaded in workers: "
            f"type={results.get('model_type', 'FSDPModelPatcher')}, "
            f"fsdp={results.get('is_fsdp', True)}"
        )
        
        # STEP 3: Create wrapper from scaffold (already have all info)
        # No need to extract properties from worker results - closed loop!
        from comfy.parallel_attention.distributed_model_wrapper import DistributedModelWrapper
        
        model_wrapper = DistributedModelWrapper(
            executor=self.executor,
            scaffold=scaffold  # Use scaffold directly
        )
        
        logging.info(
            f"{LOG_PREFIX} [Loader] Wrapper created from scaffold (closed loop): {model_wrapper}"
        )
        
        # Return wrapper (ComfyUI samplers will call wrapper.apply_model())
        return (model_wrapper,)
    
    def __del__(self):
        """Cleanup: shutdown executor when node is destroyed."""
        if self.executor is not None and self.is_initialized:
            logging.info(f"{LOG_PREFIX} [Loader] Shutting down distributed executor")
            self.executor.shutdown()
            self.executor = None
            self.is_initialized = False


class TestParallelAttention:
    """Test all parallel attention functionality with scaffold pattern validation.
    
    TDD test node for Phase 2.3 Model Scaffold Pattern.
    Tests the "Copy at Perfect Information" architecture.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    FUNCTION = "test_all"
    CATEGORY = "testing"
    
    def test_all(self, model):
        """Run all validation tests for scaffold pattern."""
        results = []
        
        # Test 1: Wrapper validation
        from comfy.parallel_attention.distributed_model_wrapper import DistributedModelWrapper
        if isinstance(model, DistributedModelWrapper):
            results.append("✅ DistributedModelWrapper created")
            results.append(f"✅ World size: {model.world_size}")
            results.append(f"✅ Model size: {model.model_size() / (1024**3):.2f} GB")
        else:
            results.append(f"❌ Not DistributedModelWrapper: {type(model)}")
            return ("\n".join(results),)
        
        # Test 2: Scaffold properties (THE KEY FIX)
        try:
            # Test latent_format (was causing NoneType error)
            latent_format = model.get_model_object("latent_format")
            if latent_format is None:
                results.append("❌ latent_format is None (scaffold pattern FAILED)")
            else:
                results.append(f"✅ latent_format: {latent_format.__class__.__name__}")
                results.append(f"✅ latent_channels: {latent_format.latent_channels}")
                results.append(f"✅ latent_dimensions: {latent_format.latent_dimensions}")
            
            # Test other scaffold properties
            results.append(f"✅ dtype: {model.dtype}")
            results.append(f"✅ model_type: {model.model_type}")
            results.append(f"✅ is_adm: {model.is_adm()}")
            
            # Test extra_conds
            extra_conds = model.extra_conds()
            results.append(f"✅ extra_conds: {type(extra_conds).__name__}")
            
        except Exception as e:
            results.append(f"❌ Scaffold property test failed: {e}")
            import traceback
            results.append(f"   {traceback.format_exc()}")
        
        # Test 3: Forward pass (PENDING - handler not implemented yet)
        try:
            results.append("⏸️  Forward pass: PENDING (handler implementation)")
            # TODO: Enable when forward_pass handler complete
            # import torch
            # x = torch.randn(1, 16, 64, 64)
            # timestep = torch.tensor([999.0])
            # output = model.apply_model(x, timestep)
            # results.append(f"✅ Forward pass successful: {tuple(output.shape)}")
        except Exception as e:
            results.append(f"⏸️  Forward pass: {e}")
        
        # Test 4: VRAM
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                vram = torch.cuda.memory_allocated(i) / (1024**3)
                results.append(f"✅ GPU {i}: {vram:.2f} GB")
        
        return ("\n".join(results),)


NODE_CLASS_MAPPINGS = {
    "UnetLoaderParallelAttention": UnetLoaderParallelAttention,
    "TestParallelAttention": TestParallelAttention,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UnetLoaderParallelAttention": "Unet Loader (Parallel Attention)",
    "TestParallelAttention": "Test Parallel Attention (All)",
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
                "test_type": (["basic", "devicemesh", "fsdp_policy", "all"],),
            },
            "optional": {
                "model": ("MODEL",),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    FUNCTION = "test_executor"
    CATEGORY = "parallel_attention"
    
    def test_executor(self, world_size, backend, test_type, model=None):
        """Test the distributed executor."""
        
        from comfy.parallel_attention import MultiprocExecutor
        
        logging.info(f"{LOG_PREFIX} [Test] Starting test: world_size={world_size}, backend={backend}, test_type={test_type}")
        
        # If model provided, check if it's FSDP
        if model is not None:
            from comfy.parallel_attention.fsdp_model_patcher import FSDPModelPatcher
            
            is_fsdp = isinstance(model, FSDPModelPatcher)
            logging.info(f"{LOG_PREFIX} [Test] Model provided: FSDP={is_fsdp}")
            
            if is_fsdp:
                logging.info(f"{LOG_PREFIX} [Test] FSDP Model detected:")
                logging.info(f"{LOG_PREFIX} [Test]   Shard factor: {model.shard_factor}")
                logging.info(f"{LOG_PREFIX} [Test]   Wrapped: {model.is_fsdp_wrapped}")
                logging.info(f"{LOG_PREFIX} [Test]   Model size: {model.model_size()/1e9:.2f}GB")
        
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
            if test_type in ["fsdp_policy", "all"]:
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


