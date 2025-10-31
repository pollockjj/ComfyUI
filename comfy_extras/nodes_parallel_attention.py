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
                # Phase 2.2.1.1: Meta Device Ground Truth
                "phase2_2_1_1_meta_ground_truth": ("BOOLEAN", {"default": False}),
                
                # Phase 2.2.1.1: Copy-Exact Standard Loader
                "phase2_2_1_1_copy_exact_loader": ("BOOLEAN", {"default": False}),
                
                # Phase 2.5: Worker Spawn at Flag Detection
                "phase2_5_worker_spawn": ("BOOLEAN", {"default": False}),
                
                # Phase 2.6: Checkpoint Path to Workers
                "phase2_6_checkpoint_path": ("BOOLEAN", {"default": False}),
                
                # Phase 2.7: FSDP2 Sharding
                "phase2_7_fsdp2_sharding": ("BOOLEAN", {"default": False}),
                
                # Phase 2.7.2: Deep Validation (ACTIVE)
                "phase2_7_2_deep_validation": ("BOOLEAN", {"default": False}),
                
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
    
    def run_tests(self, phase2_2_1_1_meta_ground_truth, phase2_2_1_1_copy_exact_loader,
                  phase2_5_worker_spawn, phase2_6_checkpoint_path, phase2_7_fsdp2_sharding,
                  phase2_7_2_deep_validation, run_all_complete, model=None):
        """Run phase-based unit tests for parallel attention.
        
        Hardcoded: world_size=2, backend=auto (NCCL with GLOO fallback).
        """
        from comfy.parallel_attention import MultiprocExecutor
        
        # Hardcoded configuration
        world_size = 2
        backend = "auto"
        
        # Determine which phases to run
        if run_all_complete:
            # Run all completed phases
            phase2_2_1_1_meta_ground_truth = True
            phase2_2_1_1_copy_exact_loader = True
            phase2_5_worker_spawn = True
            phase2_6_checkpoint_path = True
        
        # Check if any tests enabled
        any_enabled = (phase2_2_1_1_meta_ground_truth or phase2_2_1_1_copy_exact_loader or
                      phase2_5_worker_spawn or phase2_6_checkpoint_path or phase2_7_fsdp2_sharding)
        
        if not any_enabled:
            return ("⚠️  No tests enabled. Enable at least one phase boolean.",)
        
        logging.info(f"{LOG_PREFIX} [Test] ══════════════════════════════════════════════════════")
        logging.info(f"{LOG_PREFIX} [Test] PARALLEL ATTENTION UNIT TESTS")
        logging.info(f"{LOG_PREFIX} [Test] Configuration: world_size={world_size}, backend={backend}")
        logging.info(f"{LOG_PREFIX} [Test] ══════════════════════════════════════════════════════")
        
        results = []
        
        try:
            # Use executor from model (Phase 2.5+)
            if model is not None and hasattr(model, 'parallel_executor') and model.parallel_executor is not None:
                executor = model.parallel_executor
                logging.info(f"{LOG_PREFIX} [Test] Using existing executor from model (backend={executor.backend})")
            else:
                executor = None
            
            # ═══════════════════════════════════════════════════════════════
            # PHASE 2.2.1.1: Meta Device Ground Truth
            # ═══════════════════════════════════════════════════════════════
            if phase2_2_1_1_meta_ground_truth:
                logging.info(f"{LOG_PREFIX} [Test] ┌─────────────────────────────────────────────────────────┐")
                logging.info(f"{LOG_PREFIX} [Test] │ PHASE 1.1: Multiprocess Foundation (COMPLETE)          │")
            # PHASE 2.2.1.1: Meta Device Ground Truth (Test 2.2.1.1)
            # ═══════════════════════════════════════════════════════════════
            if phase2_2_1_1_meta_ground_truth:
                logging.info(f"{LOG_PREFIX} [Test] ┌─────────────────────────────────────────────────────────┐")
                logging.info(f"{LOG_PREFIX} [Test] │ PHASE 2.2.1.1: Meta Device Ground Truth                │")
                logging.info(f"{LOG_PREFIX} [Test] └─────────────────────────────────────────────────────────┘")
                
                if model is None:
                    logging.info(f"{LOG_PREFIX} [Test] ⏸️  SKIP: No MODEL input")
                    results.append("⏸️  Phase 2.2.1.1: Skipped (no MODEL)")
                else:
                    checks_passed = 0
                    checks_total = 12
                    
                    # Check 1: Has meta_model attribute
                    has_meta = hasattr(model, 'meta_model') and model.meta_model is not None
                    logging.info(f"{LOG_PREFIX} [Test]   {'✅' if has_meta else '❌'} [1/12] Has meta_model attribute")
                    if has_meta: checks_passed += 1
                    
                    if not has_meta:
                        logging.error(f"{LOG_PREFIX} [Test]   ❌ Meta model not found. Start ComfyUI with --use-parallel-attention")
                        results.append(f"❌ Phase 2.2.1.1: No meta model ({checks_passed}/{checks_total})")
                    else:
                        meta = model.meta_model
                        real = model.model
                        
                        # Check 2: Meta model on meta device
                        try:
                            first_param = next(meta.parameters())
                            is_meta_device = first_param.device.type == 'meta'
                            logging.info(f"{LOG_PREFIX} [Test]   {'✅' if is_meta_device else '❌'} [2/12] Meta model on 'meta' device")
                            if is_meta_device: checks_passed += 1
                        except Exception as e:
                            logging.error(f"{LOG_PREFIX} [Test]   ❌ [2/12] Device check failed: {e}")
                        
                        # Check 3: Meta reports same size as actual, but uses 0 memory
                        try:
                            real_reported = sum(p.numel() * p.element_size() for p in real.parameters())
                            meta_reported = sum(p.numel() * p.element_size() for p in meta.parameters())
                            sizes_match = real_reported == meta_reported
                            
                            # Meta device = 0 actual allocation despite reported size
                            is_meta = all(p.device.type == 'meta' for p in meta.parameters())
                            
                            if sizes_match and is_meta:
                                logging.info(f"{LOG_PREFIX} [Test]   ✅ [3/12] Sizes match: {real_reported / (1024**3):.2f}GB reported, meta device uses 0 actual")
                                checks_passed += 1
                            else:
                                logging.error(f"{LOG_PREFIX} [Test]   ❌ [3/12] Size mismatch or not meta: real={real_reported / (1024**3):.2f}GB, meta={meta_reported / (1024**3):.2f}GB, is_meta={is_meta}")
                        except Exception as e:
                            logging.error(f"{LOG_PREFIX} [Test]   ❌ [3/12] Size comparison failed: {e}")
                        
                        # Check 4: Same class type
                        same_type = type(meta).__name__ == type(real).__name__
                        logging.info(f"{LOG_PREFIX} [Test]   {'✅' if same_type else '❌'} [4/12] Same type: {type(meta).__name__} == {type(real).__name__}")
                        if same_type: checks_passed += 1
                        
                        # Check 5: latent_format matches
                        try:
                            meta_latent = meta.latent_format.__class__.__name__
                            real_latent = real.latent_format.__class__.__name__
                            matches = meta_latent == real_latent
                            logging.info(f"{LOG_PREFIX} [Test]   {'✅' if matches else '❌'} [5/12] latent_format: {meta_latent} == {real_latent}")
                            if matches: checks_passed += 1
                        except Exception as e:
                            logging.error(f"{LOG_PREFIX} [Test]   ❌ [5/12] latent_format check failed: {e}")
                        
                        # Check 6: model_type matches
                        try:
                            meta_type = meta.model_type
                            real_type = real.model_type
                            matches = meta_type == real_type
                            logging.info(f"{LOG_PREFIX} [Test]   {'✅' if matches else '❌'} [6/12] model_type: {meta_type} == {real_type}")
                            if matches: checks_passed += 1
                        except Exception as e:
                            logging.error(f"{LOG_PREFIX} [Test]   ❌ [6/12] model_type check failed: {e}")
                        
                        # Check 7: model_config matches
                        try:
                            meta_config = meta.model_config.__class__.__name__
                            real_config = real.model_config.__class__.__name__
                            matches = meta_config == real_config
                            logging.info(f"{LOG_PREFIX} [Test]   {'✅' if matches else '❌'} [7/12] model_config: {meta_config} == {real_config}")
                            if matches: checks_passed += 1
                        except Exception as e:
                            logging.error(f"{LOG_PREFIX} [Test]   ❌ [7/12] model_config check failed: {e}")
                        
                        # Check 8: diffusion_model exists on both
                        try:
                            has_meta_diff = hasattr(meta, 'diffusion_model')
                            has_real_diff = hasattr(real, 'diffusion_model')
                            both_have = has_meta_diff and has_real_diff
                            logging.info(f"{LOG_PREFIX} [Test]   {'✅' if both_have else '❌'} [8/12] Both have diffusion_model")
                            if both_have: checks_passed += 1
                        except Exception as e:
                            logging.error(f"{LOG_PREFIX} [Test]   ❌ [8/12] diffusion_model check failed: {e}")
                        
                        # Check 9: Same diffusion_model type
                        try:
                            if hasattr(meta, 'diffusion_model') and hasattr(real, 'diffusion_model'):
                                meta_diff_type = type(meta.diffusion_model).__name__
                                real_diff_type = type(real.diffusion_model).__name__
                                matches = meta_diff_type == real_diff_type
                                logging.info(f"{LOG_PREFIX} [Test]   {'✅' if matches else '❌'} [9/12] diffusion_model type: {meta_diff_type} == {real_diff_type}")
                                if matches: checks_passed += 1
                            else:
                                logging.error(f"{LOG_PREFIX} [Test]   ❌ [9/12] Missing diffusion_model")
                        except Exception as e:
                            logging.error(f"{LOG_PREFIX} [Test]   ❌ [9/12] diffusion_model type check failed: {e}")
                        
                        # Check 10: Same number of parameters
                        try:
                            meta_params = sum(1 for _ in meta.parameters())
                            real_params = sum(1 for _ in real.parameters())
                            matches = meta_params == real_params
                            logging.info(f"{LOG_PREFIX} [Test]   {'✅' if matches else '❌'} [10/12] Parameter count: {meta_params} == {real_params}")
                            if matches: checks_passed += 1
                        except Exception as e:
                            logging.error(f"{LOG_PREFIX} [Test]   ❌ [10/12] Parameter count check failed: {e}")
                        
                        # Check 11: Same module structure
                        try:
                            meta_modules = set(name for name, _ in meta.named_modules())
                            real_modules = set(name for name, _ in real.named_modules())
                            matches = meta_modules == real_modules
                            diff = len(meta_modules.symmetric_difference(real_modules))
                            logging.info(f"{LOG_PREFIX} [Test]   {'✅' if matches else '❌'} [11/12] Module structure matches (diff: {diff})")
                            if matches: checks_passed += 1
                        except Exception as e:
                            logging.error(f"{LOG_PREFIX} [Test]   ❌ [11/12] Module structure check failed: {e}")
                        
                        # Check 12: Ground truth validation (architecture-aware)
                        try:
                            # Check if this is FSDP2ModelPatcher (Phase 2.3 architecture)
                            from comfy.model_patcher_fsdp2 import FSDP2ModelPatcher
                            is_fsdp2 = isinstance(model, FSDP2ModelPatcher)
                            
                            real_first_param = next(real.parameters())
                            meta_first_param = next(meta.parameters())
                            
                            if is_fsdp2:
                                # FSDP2 Architecture: Meta model IS the interface
                                # Both model.model and model.meta_model should be on meta device
                                # Workers hold the actual sharded weights
                                real_is_meta = real_first_param.device.type == 'meta'
                                meta_is_meta = meta_first_param.device.type == 'meta'
                                correct = real_is_meta and meta_is_meta
                                logging.info(f"{LOG_PREFIX} [Test]   {'✅' if correct else '❌'} [12/12] FSDP2: model.model on meta (interface), model.meta_model on meta (reference)")
                            else:
                                # Standard Architecture: Real has weights, meta doesn't
                                real_has_data = real_first_param.device.type != 'meta'
                                meta_no_data = meta_first_param.device.type == 'meta'
                                correct = real_has_data and meta_no_data
                                logging.info(f"{LOG_PREFIX} [Test]   {'✅' if correct else '❌'} [12/12] Standard: Real has weights ({real_first_param.device}), meta doesn't (meta)")
                            
                            if correct: checks_passed += 1
                        except Exception as e:
                            logging.error(f"{LOG_PREFIX} [Test]   ❌ [12/12] Ground truth check failed: {e}")
                        
                        # Summary
                        if checks_passed == checks_total:
                            logging.info(f"{LOG_PREFIX} [Test] ✅ PASS [Test 2.2.1.1]: {checks_passed}/{checks_total}")
                            results.append(f"✅ Phase 2.2.1.1: Meta Ground Truth ({checks_passed}/{checks_total})")
                        else:
                            logging.error(f"{LOG_PREFIX} [Test] ❌ FAIL [Test 2.2.1.1]: {checks_passed}/{checks_total}")
                            results.append(f"❌ Phase 2.2.1.1: Meta Ground Truth ({checks_passed}/{checks_total})")
            
            # ═══════════════════════════════════════════════════════════════
            # PHASE 2.2.1.1: Copy-Exact Standard Loader (Test 2.2.1.1)
            # ═══════════════════════════════════════════════════════════════
            if phase2_2_1_1_copy_exact_loader:
                logging.info(f"{LOG_PREFIX} [Test] ┌─────────────────────────────────────────────────────────┐")
                logging.info(f"{LOG_PREFIX} [Test] │ PHASE 2.2.1.1: Copy-Exact Standard Loader              │")
                logging.info(f"{LOG_PREFIX} [Test] └─────────────────────────────────────────────────────────┘")
                
                if model is None:
                    logging.info(f"{LOG_PREFIX} [Test] ⏸️  SKIP: No MODEL input")
                    results.append("⏸️  Phase 2.2.1.1 Copy-Exact: Skipped (no MODEL)")
                else:
                    from comfy.model_patcher import ModelPatcher
                    checks_passed = 0
                    checks_total = 3
                    
                    # Check 1: Is it a ModelPatcher?
                    is_patcher = isinstance(model, ModelPatcher)
                    logging.info(f"{LOG_PREFIX} [Test]   {'✅' if is_patcher else '❌'} [1/3] isinstance(ModelPatcher): {is_patcher}")
                    if is_patcher: checks_passed += 1
                    
                    # Check 2: Can call model_size()?
                    try:
                        size = model.model_size()
                        logging.info(f"{LOG_PREFIX} [Test]   ✅ [2/3] model_size() works: {size / (1024**3):.2f}GB")
                        checks_passed += 1
                    except Exception as e:
                        logging.error(f"{LOG_PREFIX} [Test]   ❌ [2/3] model_size() failed: {e}")
                    
                    # Check 3: Can call load()?
                    try:
                        model.load(model.load_device)
                        logging.info(f"{LOG_PREFIX} [Test]   ✅ [3/3] load() works")
                        checks_passed += 1
                    except Exception as e:
                        logging.error(f"{LOG_PREFIX} [Test]   ❌ [3/3] load() failed: {e}")
                    
                    # Summary
                    if checks_passed == checks_total:
                        logging.info(f"{LOG_PREFIX} [Test] ✅ PASS: {checks_passed}/{checks_total}")
                        results.append(f"✅ Phase 2.2.1.1: Standard Loader ({checks_passed}/{checks_total})")
                    else:
                        logging.error(f"{LOG_PREFIX} [Test] ❌ FAIL: {checks_passed}/{checks_total}")
                        results.append(f"❌ Phase 2.2.1.1: Standard Loader ({checks_passed}/{checks_total})")
            
            # ═══════════════════════════════════════════════════════════════
            # PHASE 2.5: Worker Spawn at Flag Detection (Test 2.5)
            # ═══════════════════════════════════════════════════════════════
            if phase2_5_worker_spawn:
                logging.info(f"{LOG_PREFIX} [Test] ┌─────────────────────────────────────────────────────────┐")
                logging.info(f"{LOG_PREFIX} [Test] │ PHASE 2.5: Worker Spawn at Flag Detection              │")
                logging.info(f"{LOG_PREFIX} [Test] └─────────────────────────────────────────────────────────┘")
                
                if model is None:
                    logging.info(f"{LOG_PREFIX} [Test] ⏸️  SKIP: No MODEL input")
                    results.append("⏸️  Phase 2.5: Skipped (no MODEL)")
                else:
                    checks_passed = 0
                    checks_total = 5
                    
                    # Check 1: Has parallel_executor attribute
                    has_executor_attr = hasattr(model, 'parallel_executor') and model.parallel_executor is not None
                    logging.info(f"{LOG_PREFIX} [Test]   {'✅' if has_executor_attr else '❌'} [1/5] Has parallel_executor attribute")
                    if has_executor_attr: checks_passed += 1
                    
                    if not has_executor_attr:
                        logging.error(f"{LOG_PREFIX} [Test]   ❌ Executor not found. Start ComfyUI with --use-parallel-attention")
                        results.append(f"❌ Phase 2.5: No executor ({checks_passed}/{checks_total})")
                    else:
                        test_executor = model.parallel_executor
                        
                        # Check 2: Executor has correct world_size
                        try:
                            exec_world_size = getattr(test_executor, 'world_size', None)
                            correct_size = exec_world_size == 2
                            logging.info(f"{LOG_PREFIX} [Test]   {'✅' if correct_size else '❌'} [2/5] world_size: {exec_world_size} == 2")
                            if correct_size: checks_passed += 1
                        except Exception as e:
                            logging.error(f"{LOG_PREFIX} [Test]   ❌ [2/5] world_size check failed: {e}")
                        
                        # Check 3: Backend selected (NCCL or GLOO)
                        try:
                            exec_backend = getattr(test_executor, 'backend', None)
                            valid_backend = exec_backend in ['nccl', 'gloo']
                            logging.info(f"{LOG_PREFIX} [Test]   {'✅' if valid_backend else '❌'} [3/5] backend: {exec_backend} in ['nccl', 'gloo']")
                            if valid_backend: checks_passed += 1
                        except Exception as e:
                            logging.error(f"{LOG_PREFIX} [Test]   ❌ [3/5] backend check failed: {e}")
                        
                        # Check 4: Workers are alive (echo test)
                        try:
                            test_msg = "phase2_5_test"
                            result = test_executor.execute_collective("echo", {"message": test_msg})
                            workers_alive = result == test_msg
                            logging.info(f"{LOG_PREFIX} [Test]   {'✅' if workers_alive else '❌'} [4/5] Workers alive (echo test)")
                            if workers_alive: checks_passed += 1
                        except Exception as e:
                            logging.error(f"{LOG_PREFIX} [Test]   ❌ [4/5] Echo test failed: {e}")
                        
                        # Check 5: Meta model still created correctly
                        has_meta = hasattr(model, 'meta_model') and model.meta_model is not None
                        if has_meta:
                            try:
                                first_param = next(model.meta_model.parameters())
                                is_meta = first_param.device.type == 'meta'
                                logging.info(f"{LOG_PREFIX} [Test]   {'✅' if is_meta else '❌'} [5/5] Meta model on 'meta' device")
                                if is_meta: checks_passed += 1
                            except Exception as e:
                                logging.error(f"{LOG_PREFIX} [Test]   ❌ [5/5] Meta check failed: {e}")
                        else:
                            logging.error(f"{LOG_PREFIX} [Test]   ❌ [5/5] No meta_model attribute")
                        
                        # Summary
                        if checks_passed == checks_total:
                            logging.info(f"{LOG_PREFIX} [Test] ✅ PASS: {checks_passed}/{checks_total}")
                            results.append(f"✅ Phase 2.5: Worker Spawn ({checks_passed}/{checks_total})")
                        else:
                            logging.error(f"{LOG_PREFIX} [Test] ❌ FAIL: {checks_passed}/{checks_total}")
                            results.append(f"❌ Phase 2.5: Worker Spawn ({checks_passed}/{checks_total})")
            
            # ═══════════════════════════════════════════════════════════════
            # PHASE 2.6: Checkpoint Path to Workers (Test 2.6)
            # ═══════════════════════════════════════════════════════════════
            if phase2_6_checkpoint_path:
                logging.info(f"{LOG_PREFIX} [Test] ┌─────────────────────────────────────────────────────────┐")
                logging.info(f"{LOG_PREFIX} [Test] │ PHASE 2.6: Checkpoint Path to Workers                  │")
                logging.info(f"{LOG_PREFIX} [Test] └─────────────────────────────────────────────────────────┘")
                
                if model is None:
                    logging.info(f"{LOG_PREFIX} [Test] ⏸️  SKIP: No MODEL input")
                    results.append("⏸️  Phase 2.6: Skipped (no MODEL)")
                else:
                    checks_passed = 0
                    checks_total = 4
                    
                    # Check 1: ModelPatcher has checkpoint_path
                    has_checkpoint_path = hasattr(model, 'checkpoint_path') and model.checkpoint_path is not None
                    logging.info(f"{LOG_PREFIX} [Test]   {'✅' if has_checkpoint_path else '❌'} [1/4] Has checkpoint_path attribute")
                    if has_checkpoint_path: checks_passed += 1
                    
                    if not has_checkpoint_path:
                        logging.error(f"{LOG_PREFIX} [Test]   ❌ checkpoint_path not found")
                        results.append(f"❌ Phase 2.6: No checkpoint path ({checks_passed}/{checks_total})")
                    else:
                        checkpoint_path = model.checkpoint_path
                        logging.info(f"{LOG_PREFIX} [Test]   Checkpoint path: {checkpoint_path}")
                        
                        # Check 2: Has executor to send command
                        has_executor_attr = hasattr(model, 'parallel_executor') and model.parallel_executor is not None
                        logging.info(f"{LOG_PREFIX} [Test]   {'✅' if has_executor_attr else '❌'} [2/4] Has executor to send command")
                        if has_executor_attr: checks_passed += 1
                        
                        if not has_executor_attr:
                            logging.error(f"{LOG_PREFIX} [Test]   ❌ No executor")
                            results.append(f"❌ Phase 2.6: No executor ({checks_passed}/{checks_total})")
                        else:
                            test_executor = model.parallel_executor
                            
                            # Check 3: Workers can load checkpoint
                            try:
                                result = test_executor.execute_collective("load_checkpoint", {
                                    "checkpoint_path": checkpoint_path
                                })
                                
                                worker_loaded = result.get("status") == "success"
                                key_count = result.get("key_count", 0)
                                total_size_gb = result.get("total_size_gb", 0)
                                
                                logging.info(f"{LOG_PREFIX} [Test]   {'✅' if worker_loaded else '❌'} [3/4] Workers loaded checkpoint: {key_count} keys, {total_size_gb:.2f}GB")
                                if worker_loaded: checks_passed += 1
                                
                            except Exception as e:
                                logging.error(f"{LOG_PREFIX} [Test]   ❌ [3/4] Worker load failed: {e}")
                                import traceback
                                traceback.print_exc()
                            
                            # Check 4: Key count matches (verify workers got actual checkpoint)
                            try:
                                keys_match = key_count > 700  # Flux has 780 keys, Wan ~similar
                                
                                logging.info(f"{LOG_PREFIX} [Test]   {'✅' if keys_match else '❌'} [4/4] Worker has state_dict keys: {key_count}")
                                if keys_match: checks_passed += 1
                                
                            except Exception as e:
                                logging.error(f"{LOG_PREFIX} [Test]   ❌ [4/4] Key count check failed: {e}")
                            
                            # Summary
                            if checks_passed == checks_total:
                                logging.info(f"{LOG_PREFIX} [Test] ✅ PASS: {checks_passed}/{checks_total}")
                                results.append(f"✅ Phase 2.6: Checkpoint Path ({checks_passed}/{checks_total})")
                            else:
                                logging.error(f"{LOG_PREFIX} [Test] ❌ FAIL: {checks_passed}/{checks_total}")
                                results.append(f"❌ Phase 2.6: Checkpoint Path ({checks_passed}/{checks_total})")
            
            # ═══════════════════════════════════════════════════════════════
            # PHASE 2.7: FSDP2 Sharding (Test 2.7)
            # ═══════════════════════════════════════════════════════════════
            if phase2_7_fsdp2_sharding:
                logging.info(f"{LOG_PREFIX} [Test] ┌─────────────────────────────────────────────────────────┐")
                logging.info(f"{LOG_PREFIX} [Test] │ PHASE 2.7: FSDP2 Sharding                              │")
                logging.info(f"{LOG_PREFIX} [Test] └─────────────────────────────────────────────────────────┘")
                
                if model is None:
                    logging.info(f"{LOG_PREFIX} [Test] ⏸️  SKIP: No MODEL input")
                    results.append("⏸️  Phase 2.7: Skipped (no MODEL)")
                else:
                    checks_passed = 0
                    checks_total = 5
                    
                    # Check 1: Has meta_model
                    has_meta = hasattr(model, 'meta_model') and model.meta_model is not None
                    logging.info(f"{LOG_PREFIX} [Test]   {'✅' if has_meta else '❌'} [1/5] Has meta_model")
                    if has_meta: checks_passed += 1
                    
                    # Check 2: Has checkpoint_path
                    has_checkpoint = hasattr(model, 'checkpoint_path') and model.checkpoint_path is not None
                    logging.info(f"{LOG_PREFIX} [Test]   {'✅' if has_checkpoint else '❌'} [2/5] Has checkpoint_path")
                    if has_checkpoint: checks_passed += 1
                    
                    # Check 3: Has executor
                    has_executor = hasattr(model, 'parallel_executor') and model.parallel_executor is not None
                    logging.info(f"{LOG_PREFIX} [Test]   {'✅' if has_executor else '❌'} [3/5] Has executor")
                    if has_executor: checks_passed += 1
                    
                    if has_meta and has_checkpoint and has_executor:
                        # Check 4: Workers apply FSDP2 sharding
                        try:
                            # Detect model type from class name
                            model_class_name = model.model.__class__.__name__
                            if "Flux" in model_class_name:
                                model_type = "flux"
                            elif "Wan" in model_class_name or "WAN" in model_class_name:
                                model_type = "wan"
                            elif "Qwen" in model_class_name:
                                model_type = "qwen_image"
                            else:
                                raise ValueError(f"Unknown model type: {model_class_name}")
                            
                            logging.info(f"{LOG_PREFIX} [Test]   Detected model type: {model_type} (from {model_class_name})")
                            
                            # Clean meta_model for pickling (remove unpicklable attributes)
                            import copy
                            clean_meta = copy.deepcopy(model.meta_model)
                            if hasattr(clean_meta, 'model_sampling'):
                                delattr(clean_meta, 'model_sampling')
                            if hasattr(clean_meta, 'latent_format'):
                                delattr(clean_meta, 'latent_format')
                            
                            # Workers use FastVideo iterator pattern (stream from file)
                            result = model.parallel_executor.execute_collective(
                                "initialize_fsdp2_from_checkpoint",
                                {
                                    "checkpoint_path": model.checkpoint_path,
                                    "model_type": model_type,
                                    "meta_model": clean_meta
                                }
                            )
                            
                            sharding_success = result.get("status") == "success"
                            vram_gb = result.get("vram_gb", 0)
                            
                            logging.info(f"{LOG_PREFIX} [Test]   {'✅' if sharding_success else '❌'} [4/5] FSDP2 sharding applied: {vram_gb:.2f}GB VRAM")
                            if sharding_success: checks_passed += 1
                            
                        except Exception as e:
                            logging.error(f"{LOG_PREFIX} [Test]   ❌ [4/5] FSDP2 sharding failed: {e}")
                            import traceback
                            traceback.print_exc()
                        
                        # Check 5: VRAM reduced (≤12GB per worker)
                        try:
                            vram_target = 12.0  # GB
                            vram_reduced = vram_gb <= vram_target
                            logging.info(f"{LOG_PREFIX} [Test]   {'✅' if vram_reduced else '❌'} [5/5] VRAM ≤ {vram_target}GB: {vram_gb:.2f}GB")
                            if vram_reduced: checks_passed += 1
                        except Exception as e:
                            logging.error(f"{LOG_PREFIX} [Test]   ❌ [5/5] VRAM check failed: {e}")
                    
                    # Summary
                    if checks_passed == checks_total:
                        logging.info(f"{LOG_PREFIX} [Test] ✅ PASS: {checks_passed}/{checks_total}")
                        results.append(f"✅ Phase 2.7: FSDP2 Sharding ({checks_passed}/{checks_total})")
                    else:
                        logging.error(f"{LOG_PREFIX} [Test] ❌ FAIL: {checks_passed}/{checks_total}")
                        results.append(f"❌ Phase 2.7: FSDP2 Sharding ({checks_passed}/{checks_total})")
            
            # Phase 2.7.2: Deep Validation Tests
            if phase2_7_2_deep_validation:
                logging.info(f"{LOG_PREFIX} [Test] ┌─────────────────────────────────────────────────────────┐")
                logging.info(f"{LOG_PREFIX} [Test] │ PHASE 2.7.2: Deep Validation                           │")
                logging.info(f"{LOG_PREFIX} [Test] └─────────────────────────────────────────────────────────┘")
                
                if model is None:
                    logging.info(f"{LOG_PREFIX} [Test] ⏸️  SKIP: No MODEL input")
                    results.append("⏸️  Phase 2.7.2: Skipped (no MODEL)")
                else:
                    checks_passed = 0
                    checks_total = 4
                    
                    # Test 1: Ignored param replicated
                    try:
                        result = model.parallel_executor.execute_collective(
                            "check_param_sharding",
                            {"param_name": "diffusion_model.img_in.weight"}
                        )
                        
                        is_replicated = not result.get("is_sharded", True)
                        size_mb = result.get("size_mb", 0)
                        
                        logging.info(f"{LOG_PREFIX} [Test]   {'✅' if is_replicated else '❌'} [1/4] img_in.weight replicated: {size_mb:.1f}MB (not sharded)")
                        if is_replicated: checks_passed += 1
                        
                    except Exception as e:
                        logging.error(f"{LOG_PREFIX} [Test]   ❌ [1/4] Ignored param check failed: {e}")
                        import traceback
                        traceback.print_exc()
                    
                    # Test 2: Sharded param distributed
                    try:
                        result = model.parallel_executor.execute_collective(
                            "check_param_sharding",
                            {"param_name": "diffusion_model.double_blocks.0.img_attn.qkv.weight"}
                        )
                        
                        is_sharded = result.get("is_sharded", False)
                        local_shape = result.get("local_shape", ())
                        global_shape = result.get("global_shape", ())
                        
                        logging.info(f"{LOG_PREFIX} [Test]   {'✅' if is_sharded else '❌'} [2/4] double_blocks.0 sharded: local={local_shape} global={global_shape}")
                        if is_sharded: checks_passed += 1
                        
                    except Exception as e:
                        logging.error(f"{LOG_PREFIX} [Test]   ❌ [2/4] Sharded param check failed: {e}")
                        import traceback
                        traceback.print_exc()
                    
                    # Test 3: VRAM breakdown
                    try:
                        result = model.parallel_executor.execute_collective(
                            "get_vram_breakdown",
                            {}
                        )
                        
                        sharded_vram = result.get("sharded_vram_gb", 0)
                        replicated_vram = result.get("replicated_vram_gb", 0)
                        sharded_count = result.get("sharded_count", 0)
                        replicated_count = result.get("replicated_count", 0)
                        
                        breakdown_ok = sharded_vram > 8 and replicated_vram < 3
                        
                        logging.info(f"{LOG_PREFIX} [Test]   {'✅' if breakdown_ok else '❌'} [3/4] VRAM: {sharded_vram:.2f}GB sharded ({sharded_count}), {replicated_vram:.2f}GB replicated ({replicated_count})")
                        if breakdown_ok: checks_passed += 1
                        
                    except Exception as e:
                        logging.error(f"{LOG_PREFIX} [Test]   ❌ [3/4] VRAM breakdown failed: {e}")
                        import traceback
                        traceback.print_exc()
                    
                    # Test 4: Sharding strategy
                    try:
                        result = model.parallel_executor.execute_collective(
                            "validate_sharding_strategy",
                            {}
                        )
                        
                        double_wrapped = result.get("double_blocks_wrapped", 0)
                        single_wrapped = result.get("single_blocks_wrapped", 0)
                        total_fsdp = result.get("total_fsdp_modules", 0)
                        
                        strategy_ok = double_wrapped == 19 and single_wrapped == 38
                        
                        logging.info(f"{LOG_PREFIX} [Test]   {'✅' if strategy_ok else '❌'} [4/4] Strategy: {double_wrapped} double + {single_wrapped} single = {total_fsdp} FSDP modules")
                        if strategy_ok: checks_passed += 1
                        
                    except Exception as e:
                        logging.error(f"{LOG_PREFIX} [Test]   ❌ [4/4] Strategy validation failed: {e}")
                        import traceback
                        traceback.print_exc()
                    
                    # Summary
                    if checks_passed == checks_total:
                        logging.info(f"{LOG_PREFIX} [Test] ✅ PASS: {checks_passed}/{checks_total}")
                        results.append(f"✅ Phase 2.7.2: Deep Validation ({checks_passed}/{checks_total})")
                    else:
                        logging.error(f"{LOG_PREFIX} [Test] ❌ FAIL: {checks_passed}/{checks_total}")
                        results.append(f"❌ Phase 2.7.2: Deep Validation ({checks_passed}/{checks_total})")
            
            # Shutdown executor ONLY if we created it (not from model)
            if executor and not (model is not None and hasattr(model, 'parallel_executor') and model.parallel_executor is executor):
                logging.info(f"{LOG_PREFIX} [Test] Shutting down test executor...")
                executor.shutdown()
            elif executor:
                logging.info(f"{LOG_PREFIX} [Test] Keeping model's executor alive (created in sd.py)")
            
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
            
            # Try to shutdown executor if it exists AND we created it
            try:
                if 'executor' in locals() and executor and not (model is not None and hasattr(model, 'parallel_executor') and model.parallel_executor is executor):
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