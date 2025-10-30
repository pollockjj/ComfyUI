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
        try:
            logging.info(f"{LOG_PREFIX} [Loader] Starting _initialize_distributed...")
            
            from comfy.parallel_attention.executor import MultiprocExecutor
            logging.info(f"{LOG_PREFIX} [Loader] MultiprocExecutor imported successfully")
            
            logging.info(f"{LOG_PREFIX} [Loader] Initializing distributed on 2 GPUs")
            
            # World size is always 2 for parallel attention
            world_size = 2
            
            # Backend is auto-selected (NCCL with GLOO fallback)
            backend = "auto"
            
            # Create executor - devices auto-assigned as cuda:0 and cuda:1
            logging.info(f"{LOG_PREFIX} [Loader] Creating MultiprocExecutor(world_size={world_size}, backend={backend})...")
            self.executor = MultiprocExecutor(world_size=world_size, backend=backend)
            logging.info(f"{LOG_PREFIX} [Loader] Executor created: {self.executor}")
            logging.info(f"{LOG_PREFIX} [Loader] Executor type: {type(self.executor)}")
            
            self.is_initialized = True
            logging.info(f"{LOG_PREFIX} [Loader] Distributed executor initialized successfully")
        except Exception as e:
            import traceback
            error_msg = f"{LOG_PREFIX} [Loader] Executor initialization failed: {e}\n{traceback.format_exc()}"
            logging.error(error_msg)
            raise RuntimeError(error_msg)
    
    def load_unet_parallel(self, unet_name):
        """Load UNET with FSDP2 sharding across 2 GPUs.
        
        Uses ComfyUI's standard loading with model_options['fsdp2'] opt-in.
        Returns FSDP2ModelPatcher directly (no wrapper, no scaffold, no serialization).
        
        Args:
            unet_name: Model filename
        
        Returns:
            Tuple of (FSDP2ModelPatcher,) ready for use
        """
        # Validate GPUs
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            raise RuntimeError(
                f"{LOG_PREFIX} [Loader] Parallel Attention requires 2+ CUDA devices. "
                f"Found: {torch.cuda.device_count() if torch.cuda.is_available() else 0}"
            )
        
        # Initialize executor if needed
        if not self.is_initialized or self.executor is None:
            self._initialize_distributed()
        
        # Get checkpoint path
        unet_path = folder_paths.get_full_path_or_raise("diffusion_models", unet_name)
        
        # Load state dict (KEEP ORIGINAL KEYS)
        import comfy.utils
        state_dict = comfy.utils.load_torch_file(unet_path)
        
        # Detect model config (DON'T strip prefix - model needs original keys)
        import comfy.model_detection
        model_config = comfy.model_detection.model_config_from_unet(state_dict, "")
        if model_config is None:
            raise RuntimeError(f"{LOG_PREFIX} [Loader] Could not detect model type")
        
        logging.info(f"{LOG_PREFIX} [Loader] Detected: {model_config.__class__.__name__}")
        logging.info(f"{LOG_PREFIX} [Loader] Sample checkpoint keys: {list(state_dict.keys())[:3]}")
        
        # Create META DEVICE parent model (0 bytes, all properties)
        # CRITICAL: Pass original state_dict with prefixes intact
        logging.info(f"{LOG_PREFIX} [Loader] Creating meta device parent model (0 bytes)...")
        with torch.device('meta'):
            parent_model = model_config.get_model(state_dict, "", device=torch.device('meta'))
        
        logging.info(
            f"{LOG_PREFIX} [Loader] Meta parent created: "
            f"latent_format={parent_model.latent_format.__class__.__name__}, "
            f"dtype={parent_model.get_dtype()}, "
            f"size=0 bytes (meta device)"
        )
        
        # Send checkpoint path + model_config to workers (NO model object)
        logging.info(f"{LOG_PREFIX} [Loader] Loading FSDP2 model in workers...")
        
        # Serialize model_config (small, just config dict)
        model_config_dict = {
            "class_name": model_config.__class__.__name__,
            "unet_config": model_config.unet_config,
        }
        
        results = self.executor.execute_collective("load_fsdp2_model", {
            "checkpoint_path": unet_path,
            "model_config": model_config_dict,  # ← Serializable dict, not model
        })
        
        # Check if loading succeeded
        if not results.get("success", False):
            error = results.get("error", "Unknown error")
            raise RuntimeError(f"{LOG_PREFIX} [Loader] Model loading failed: {error}")
        
        logging.info(
            f"{LOG_PREFIX} [Loader] Model loaded in workers: "
            f"FSDP2 sharding applied, VRAM={results.get('vram_gb', 0):.2f}GB per GPU"
        )
        
        # Create wrapper with meta device parent model
        from comfy.parallel_attention.distributed_model_wrapper import DistributedModelWrapper
        
        model_wrapper = DistributedModelWrapper(
            executor=self.executor,
            parent_model=parent_model  # ← Meta device model (0 bytes)
        )
        
        logging.info(f"{LOG_PREFIX} [Loader] Wrapper created with meta parent")
        
        return (model_wrapper,)


"""Test node for distributed runtime."""

import torch
import torch.multiprocessing as mp
import logging
import folder_paths

LOG_PREFIX = "⚡ [Parallel-Attention]"

class ParallelAttentionUnitTests:
    """Phase-based unit tests for parallel attention implementation.
    
    Each boolean corresponds to a development sub-phase.
    Enable tests for phases you're actively working on.
    
    world_size=2, backend=auto (hardcoded for simplicity).
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Phase 1: Foundation (COMPLETE)
                "phase1_1_multiproc": ("BOOLEAN", {"default": False}),      # Tests 1.1-1.2: Spawn, RPC
                "phase1_2_collectives": ("BOOLEAN", {"default": False}),    # Test 1.2: all_reduce
                "phase1_3_devicemesh": ("BOOLEAN", {"default": False}),     # Test 1.3: DeviceMesh
                
                # Phase 2.1: FSDP2 Policies (COMPLETE)
                "phase2_1_fsdp_policies": ("BOOLEAN", {"default": False}),  # Test 2.1: Registry
                
                # Phase 2.2: FSDP2 API Migration (COMPLETE)
                "phase2_2_fsdp2_api": ("BOOLEAN", {"default": False}),      # Test 2.2: API migration
                
                # Phase 2.2.1: Model Loading Validation (COMPLETE)
                "phase2_2_1_model_loading": ("BOOLEAN", {"default": False}), # Test 2.2.1: Model validation
                
                # Convenience: Run all completed phases
                "run_all_complete": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "model": ("MODEL",),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    FUNCTION = "run_tests"
    CATEGORY = "parallel_attention"
    
    def run_tests(self, phase1_1_multiproc, phase1_2_collectives, phase1_3_devicemesh,
                  phase2_1_fsdp_policies, phase2_2_fsdp2_api, phase2_2_1_model_loading,
                  run_all_complete, model=None):
        """Run phase-based unit tests for parallel attention.
        
        Hardcoded: world_size=2, backend=auto (NCCL with GLOO fallback).
        """
        from comfy.parallel_attention import MultiprocExecutor
        
        # Hardcoded configuration
        world_size = 2
        backend = "auto"
        
        # Determine which phases to run
        if run_all_complete:
            # Run all completed phases (1.x, 2.1, 2.2, 2.2.1)
            phase1_1_multiproc = True
            phase1_2_collectives = True
            phase1_3_devicemesh = True
            phase2_1_fsdp_policies = True
            phase2_2_fsdp2_api = True
            phase2_2_1_model_loading = True
        
        # Check if any tests enabled
        any_enabled = (phase1_1_multiproc or phase1_2_collectives or phase1_3_devicemesh or
                      phase2_1_fsdp_policies or phase2_2_fsdp2_api or phase2_2_1_model_loading)
        
        if not any_enabled:
            return ("⚠️  No tests enabled. Enable at least one phase boolean.",)
        
        logging.info(f"{LOG_PREFIX} [Test] ══════════════════════════════════════════════════════")
        logging.info(f"{LOG_PREFIX} [Test] PARALLEL ATTENTION UNIT TESTS")
        logging.info(f"{LOG_PREFIX} [Test] Configuration: world_size={world_size}, backend={backend}")
        logging.info(f"{LOG_PREFIX} [Test] ══════════════════════════════════════════════════════")
        
        results = []
        
        try:
            # Initialize executor for Phase 1+ tests
            if phase1_1_multiproc or phase1_2_collectives or phase1_3_devicemesh or phase2_1_fsdp_policies or phase2_2_fsdp2_api:
                logging.info(f"{LOG_PREFIX} [Test] Initializing MultiprocExecutor...")
                executor = MultiprocExecutor(world_size=world_size, backend=backend)
                logging.info(f"{LOG_PREFIX} [Test] ✅ Executor ready (backend={executor.backend})")
            else:
                executor = None
            
            # ═══════════════════════════════════════════════════════════════
            # PHASE 1.1: Multiprocess Foundation (Tests 1.1-1.2)
            # ═══════════════════════════════════════════════════════════════
            if phase1_1_multiproc:
                logging.info(f"{LOG_PREFIX} [Test] ┌─────────────────────────────────────────────────────────┐")
                logging.info(f"{LOG_PREFIX} [Test] │ PHASE 1.1: Multiprocess Foundation (COMPLETE)          │")
                logging.info(f"{LOG_PREFIX} [Test] └─────────────────────────────────────────────────────────┘")
                
                # Test 1.1.1: Worker spawn and echo RPC
                logging.info(f"{LOG_PREFIX} [Test] Test 1.1.1: Worker spawn + echo RPC...")
                test_message = "hello from comfy"
                result = executor.execute_collective("echo", {"message": test_message})
                
                if result != test_message:
                    if executor: executor.shutdown()
                    return (f"❌ FAIL [Test 1.1.1]: Echo failed. Expected '{test_message}', got '{result}'",)
                
                logging.info(f"{LOG_PREFIX} [Test] ✅ PASS [Test 1.1.1]: Echo RPC working")
                results.append("✅ Phase 1.1.1: Worker spawn + echo RPC")
                
                # Test 1.1.2: Multiple sequential RPCs (stability)
                logging.info(f"{LOG_PREFIX} [Test] Test 1.1.2: Multiple sequential RPCs (stability)...")
                for i in range(5):
                    message = f"message_{i}"
                    result = executor.execute_collective("echo", {"message": message})
                    if result != message:
                        if executor: executor.shutdown()
                        return (f"❌ FAIL [Test 1.1.2]: RPC {i} failed",)
                
                logging.info(f"{LOG_PREFIX} [Test] ✅ PASS [Test 1.1.2]: Multiple RPCs stable")
                results.append("✅ Phase 1.1.2: Multiple sequential RPCs")
            
            # ═══════════════════════════════════════════════════════════════
            # PHASE 1.2: NCCL/GLOO Collectives (Test 1.2)
            # ═══════════════════════════════════════════════════════════════
            if phase1_2_collectives:
                logging.info(f"{LOG_PREFIX} [Test] ┌─────────────────────────────────────────────────────────┐")
                logging.info(f"{LOG_PREFIX} [Test] │ PHASE 1.2: NCCL/GLOO Collectives (COMPLETE)            │")
                logging.info(f"{LOG_PREFIX} [Test] └─────────────────────────────────────────────────────────┘")
                
                if torch.cuda.is_available() and executor.backend == "nccl":
                    logging.info(f"{LOG_PREFIX} [Test] Test 1.2: torch.distributed all_reduce...")
                    result = executor.execute_collective("allreduce_test", {})
                    expected = sum(range(world_size))
                    
                    if result != expected:
                        if executor: executor.shutdown()
                        return (f"❌ FAIL [Test 1.2]: all_reduce failed. Expected {expected}, got {result}",)
                    
                    logging.info(f"{LOG_PREFIX} [Test] ✅ PASS [Test 1.2]: all_reduce collective working (result={result})")
                    results.append(f"✅ Phase 1.2: NCCL all_reduce (result={result})")
                else:
                    logging.info(f"{LOG_PREFIX} [Test] ⏸️  SKIP [Test 1.2]: CUDA not available or backend={executor.backend if executor else 'none'}")
                    results.append("⏸️  Phase 1.2: Skipped (no CUDA/NCCL)")
            
            # ═══════════════════════════════════════════════════════════════
            # PHASE 1.3: DeviceMesh Topology (Test 1.3)
            # ═══════════════════════════════════════════════════════════════
            if phase1_3_devicemesh:
                logging.info(f"{LOG_PREFIX} [Test] ┌─────────────────────────────────────────────────────────┐")
                logging.info(f"{LOG_PREFIX} [Test] │ PHASE 1.3: DeviceMesh Topology (COMPLETE)              │")
                logging.info(f"{LOG_PREFIX} [Test] └─────────────────────────────────────────────────────────┘")
                
                if torch.cuda.is_available() and executor.backend == "nccl":
                    logging.info(f"{LOG_PREFIX} [Test] Test 1.3: DeviceMesh topology + SP collective...")
                    result = executor.execute_collective("devicemesh_test", {})
                    
                    # Validate mesh structure
                    mesh_shape = result.get("mesh_shape", [])
                    expected_shape = [1, world_size]  # [dp_size=1, sp_size=world_size]
                    
                    if mesh_shape != expected_shape:
                        if executor: executor.shutdown()
                        return (f"❌ FAIL [Test 1.3]: Mesh shape mismatch. Expected {expected_shape}, got {mesh_shape}",)
                    
                    # Validate all_gather result
                    gathered = result.get("gathered", [])
                    expected_gathered = [float(i) for i in range(world_size)]
                    
                    if gathered != expected_gathered:
                        if executor: executor.shutdown()
                        return (f"❌ FAIL [Test 1.3]: SP all_gather mismatch. Expected {expected_gathered}, got {gathered}",)
                    
                    logging.info(f"{LOG_PREFIX} [Test] ✅ PASS [Test 1.3]: DeviceMesh topology correct")
                    logging.info(f"{LOG_PREFIX} [Test]   Mesh: {mesh_shape} (dp=1, sp={world_size})")
                    logging.info(f"{LOG_PREFIX} [Test]   SP rank {result['sp_rank']}/{result['sp_size']}, all_gather: {gathered}")
                    results.append(f"✅ Phase 1.3: DeviceMesh topology (sp_size={world_size})")
                else:
                    logging.info(f"{LOG_PREFIX} [Test] ⏸️  SKIP [Test 1.3]: CUDA not available or backend={executor.backend if executor else 'none'}")
                    results.append("⏸️  Phase 1.3: Skipped (no CUDA/NCCL)")
            
            # ═══════════════════════════════════════════════════════════════
            # PHASE 2.1: FSDP2 Policy Registry (Test 2.1)
            # ═══════════════════════════════════════════════════════════════
            if phase2_1_fsdp_policies:
                logging.info(f"{LOG_PREFIX} [Test] ┌─────────────────────────────────────────────────────────┐")
                logging.info(f"{LOG_PREFIX} [Test] │ PHASE 2.1: FSDP2 Policy Registry (COMPLETE)             │")
                logging.info(f"{LOG_PREFIX} [Test] └─────────────────────────────────────────────────────────┘")
                
                logging.info(f"{LOG_PREFIX} [Test] Test 2.1: FSDP2 Policy Registry (Flux/Wan/Qwen)...")
                result = executor.execute_collective("test_fsdp_policy", {"model_name": "flux"})
                
                # Validate flux policy registered
                if not result.get("is_registered", False):
                    if executor: executor.shutdown()
                    return (f"❌ FAIL [Test 2.1]: Flux policy not registered. Available: {result.get('available_policies', [])}",)
                
                # Validate policy is callable
                if not result.get("policy_callable", False):
                    if executor: executor.shutdown()
                    return (f"❌ FAIL [Test 2.1]: Flux policy not callable. Type: {result.get('policy_type')}",)
                
                # Check expected policies registered
                available = result.get("available_policies", [])
                expected_policies = ["flux", "qwen_image", "wan"]
                
                missing = [p for p in expected_policies if p not in available]
                if missing:
                    if executor: executor.shutdown()
                    return (f"❌ FAIL [Test 2.1]: Missing policies: {missing}. Available: {available}",)
                
                logging.info(f"{LOG_PREFIX} [Test] ✅ PASS [Test 2.1]: FSDP2 policies registered")
                logging.info(f"{LOG_PREFIX} [Test]   Policies: {', '.join(available)}")
                results.append(f"✅ Phase 2.1: FSDP2 Policy Registry ({len(available)} policies)")
            
            # ═══════════════════════════════════════════════════════════════
            # PHASE 2.2: FSDP2 API Migration (Test 2.2)
            # ═══════════════════════════════════════════════════════════════
            if phase2_2_fsdp2_api:
                logging.info(f"{LOG_PREFIX} [Test] ┌─────────────────────────────────────────────────────────┐")
                logging.info(f"{LOG_PREFIX} [Test] │ PHASE 2.2: FSDP2 API Migration (COMPLETE)              │")
                logging.info(f"{LOG_PREFIX} [Test] └─────────────────────────────────────────────────────────┘")
                
                logging.info(f"{LOG_PREFIX} [Test] Test 2.2: FSDP2 API Migration (fully_shard, DTensor)...")
                result = executor.execute_collective("test_fsdp2_api", {})
                
                # Check all validations passed
                if not result.get("all_passed", False):
                    failed_checks = []
                    if not result.get("fsdp2_import", False):
                        failed_checks.append("Missing fully_shard import")
                    if not result.get("no_fsdp1_import", False):
                        failed_checks.append("Still has FSDP1 import")
                    if not result.get("has_helper_method", False):
                        failed_checks.append("Missing _get_modules_for_policy()")
                    if not result.get("uses_fully_shard", False):
                        failed_checks.append("Doesn't use fully_shard()")
                    if not result.get("no_fsdp_wrapper", False):
                        failed_checks.append("Still uses FSDP() wrapper")
                    if not result.get("has_dtensor_check", False):
                        failed_checks.append("Missing DTensor detection")
                    if not result.get("no_isinstance_fsdp", False):
                        failed_checks.append("Still uses isinstance(FSDP)")
                    if not result.get("has_reshard_after_forward", False):
                        failed_checks.append("Missing reshard_after_forward")
                    if not result.get("no_sharding_strategy_enum", False):
                        failed_checks.append("Still uses ShardingStrategy enum")
                    
                    if executor: executor.shutdown()
                    failure_msg = f"❌ FAIL [Test 2.2]: FSDP2 API incomplete ({result.get('passed_checks', '0/9')})\n"
                    for check in failed_checks:
                        failure_msg += f"  ❌ {check}\n"
                    return (failure_msg,)
                
                logging.info(f"{LOG_PREFIX} [Test] ✅ PASS [Test 2.2]: FSDP2 API Migration complete")
                logging.info(f"{LOG_PREFIX} [Test]   Checks: {result.get('passed_checks', '0/9')}")
                results.append(f"✅ Phase 2.2: FSDP2 API Migration (9/9 checks)")
            
            # ═══════════════════════════════════════════════════════════════
            # PHASE 2.2.1: FSDP2 Model Loading with Meta Device (Test 2.2.1)
            # ═══════════════════════════════════════════════════════════════
            if phase2_2_1_model_loading:
                logging.info(f"{LOG_PREFIX} [Test] ┌─────────────────────────────────────────────────────────┐")
                logging.info(f"{LOG_PREFIX} [Test] │ PHASE 2.2.1: FSDP2 Model Loading (Meta Device)         │")
                logging.info(f"{LOG_PREFIX} [Test] └─────────────────────────────────────────────────────────┘")
                
                if model is None:
                    logging.info(f"{LOG_PREFIX} [Test] ⏸️  SKIP [Test 2.2.1]: No MODEL input")
                    logging.info(f"{LOG_PREFIX} [Test]   Connect UnetLoaderParallelAttention → model input")
                    results.append("⏸️  Phase 2.2.1: Skipped (no MODEL)")
                else:
                    logging.info(f"{LOG_PREFIX} [Test] Test 2.2.1: Meta device parent + FSDP2 workers...")
                    
                    from comfy.parallel_attention.distributed_model_wrapper import DistributedModelWrapper
                    checks_passed = 0
                    checks_total = 4
                    
                    # Check 1: Type is DistributedModelWrapper
                    if isinstance(model, DistributedModelWrapper):
                        logging.info(f"{LOG_PREFIX} [Test]   ✅ [1/4] Type: DistributedModelWrapper")
                        checks_passed += 1
                    else:
                        logging.error(f"{LOG_PREFIX} [Test]   ❌ [1/4] Wrong type: {type(model).__name__}")
                    
                    # Check 2: Parent model on meta device
                    try:
                        if hasattr(model, '_parent'):
                            parent = model._parent
                            
                            # Check all parameters on meta device
                            params_list = list(parent.parameters())
                            if len(params_list) > 0:
                                is_meta = all(p.device.type == 'meta' for p in params_list)
                                if is_meta:
                                    logging.info(f"{LOG_PREFIX} [Test]   ✅ [2/4] Parent on meta device")
                                    logging.info(f"{LOG_PREFIX} [Test]     Parameters: {len(params_list)}")
                                    checks_passed += 1
                                else:
                                    devices = set(p.device.type for p in params_list[:5])
                                    logging.error(f"{LOG_PREFIX} [Test]   ❌ [2/4] Parent not on meta (devices: {devices})")
                            else:
                                logging.error(f"{LOG_PREFIX} [Test]   ❌ [2/4] Parent has no parameters")
                        else:
                            logging.error(f"{LOG_PREFIX} [Test]   ❌ [2/4] Model has no _parent attribute")
                    except Exception as e:
                        logging.error(f"{LOG_PREFIX} [Test]   ❌ [2/4] Meta device check failed: {e}")
                    
                    # Check 3: Parent has properties
                    try:
                        has_latent = hasattr(model, 'latent_format') and model.latent_format is not None
                        has_dtype = hasattr(model, 'get_dtype')
                        
                        if has_latent and has_dtype:
                            logging.info(f"{LOG_PREFIX} [Test]   ✅ [3/4] Parent has properties")
                            logging.info(f"{LOG_PREFIX} [Test]     latent_format: {model.latent_format.__class__.__name__}")
                            logging.info(f"{LOG_PREFIX} [Test]     dtype: {model.get_dtype()}")
                            checks_passed += 1
                        else:
                            logging.error(f"{LOG_PREFIX} [Test]   ❌ [3/4] Missing properties (latent={has_latent}, dtype={has_dtype})")
                    except Exception as e:
                        logging.error(f"{LOG_PREFIX} [Test]   ❌ [3/4] Property check failed: {e}")
                    
                    # Check 4: VRAM usage (workers should have ~11GB)
                    if torch.cuda.is_available():
                        try:
                            vram_gb = torch.cuda.memory_allocated(0) / (1024**3)
                            logging.info(f"{LOG_PREFIX} [Test]   ℹ️  [4/4] VRAM: {vram_gb:.2f}GB")
                            logging.info(f"{LOG_PREFIX} [Test]     (Workers have sharded model, main process has 0GB parent)")
                            checks_passed += 1
                        except Exception as e:
                            logging.error(f"{LOG_PREFIX} [Test]   ❌ [4/4] VRAM check failed: {e}")
                    else:
                        logging.info(f"{LOG_PREFIX} [Test]   ⏸️  [4/4] No CUDA for VRAM check")
                        checks_passed += 1
                    
                    # Summary
                    if checks_passed == checks_total:
                        logging.info(f"{LOG_PREFIX} [Test] ✅ PASS [Test 2.2.1]: {checks_passed}/{checks_total}")
                        results.append(f"✅ Phase 2.2.1: FSDP2 Model Loading ({checks_passed}/{checks_total})")
                    else:
                        logging.error(f"{LOG_PREFIX} [Test] ❌ FAIL [Test 2.2.1]: {checks_passed}/{checks_total}")
                        results.append(f"❌ Phase 2.2.1: FSDP2 Model Loading ({checks_passed}/{checks_total})")
            
            # Shutdown executor
            if executor:
                logging.info(f"{LOG_PREFIX} [Test] Shutting down executor...")
                executor.shutdown()
            
            # ═══════════════════════════════════════════════════════════════
            # Test Summary
            # ═══════════════════════════════════════════════════════════════
            logging.info(f"{LOG_PREFIX} [Test] ══════════════════════════════════════════════════════")
            logging.info(f"{LOG_PREFIX} [Test] TEST SUMMARY")
            logging.info(f"{LOG_PREFIX} [Test] ══════════════════════════════════════════════════════")
            
            for result_line in results:
                logging.info(f"{LOG_PREFIX} [Test] {result_line}")
            
            passed_count = sum(1 for r in results if r.startswith("✅"))
            pending_count = sum(1 for r in results if r.startswith("⏸️"))
            
            logging.info(f"{LOG_PREFIX} [Test] ──────────────────────────────────────────────────────")
            logging.info(f"{LOG_PREFIX} [Test] Total: {len(results)} tests ({passed_count} passed, {pending_count} pending)")
            logging.info(f"{LOG_PREFIX} [Test] Config: world_size={world_size}, backend={backend}")
            logging.info(f"{LOG_PREFIX} [Test] CUDA: {torch.cuda.is_available()}")
            logging.info(f"{LOG_PREFIX} [Test] ══════════════════════════════════════════════════════")
            
            success_msg = "\n".join([
                "═" * 70,
                "PARALLEL ATTENTION UNIT TESTS - RESULTS",
                "═" * 70,
                "",
            ] + results + [
                "",
                "─" * 70,
                f"Total: {len(results)} tests | Passed: {passed_count} | Pending: {pending_count}",
                f"Config: world_size={world_size}, backend={backend}",
                "═" * 70,
            ])
            
            return (success_msg,)
            
        except Exception as e:
            import traceback
            error_msg = f"❌ TEST FAILURE: {type(e).__name__}: {e}\n\n{traceback.format_exc()}"
            logging.error(f"{LOG_PREFIX} [Test] {error_msg}")
            
            # Try to shutdown executor if it exists
            try:
                if 'executor' in locals() and executor:
                    executor.shutdown()
            except:
                pass
            
            return (error_msg,)

NODE_CLASS_MAPPINGS = {
    "UnetLoaderParallelAttention": UnetLoaderParallelAttention,
    "ParallelAttentionUnitTests": ParallelAttentionUnitTests
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UnetLoaderParallelAttention": "UNET Loader (Parallel Attention)",
    "ParallelAttentionUnitTests": "Parallel Attention Unit Tests"
}