"""Worker process event loop."""

import torch
import torch.distributed as dist
import logging
import datetime
import os
from comfy.parallel_attention.worker_context import WorkerContext

# Logging prefix for visibility
LOG_PREFIX = "⚡ [Parallel-Attention]"

def worker_main(rank: int, 
                world_size: int, 
                init_method: str, 
                backend: str, 
                pipe):
    """Worker process event loop."""
    
    # Set device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    
    try:
        logging.info(f"{LOG_PREFIX} [Worker-{rank}] Initializing: rank={rank}/{world_size}, device={device}, backend={backend}")
        
        # Set NCCL environment variables for stability
        if backend == "nccl":
            os.environ.setdefault('TORCH_NCCL_ASYNC_ERROR_HANDLING', '1')
            os.environ.setdefault('NCCL_TIMEOUT_MS', '60000')
            os.environ.setdefault('NCCL_DEBUG', 'WARN')
            os.environ.setdefault('NCCL_IB_DISABLE', '1')
        
        # Initialize torch.distributed
        dist.init_process_group(
            backend=backend,
            init_method=init_method,
            rank=rank,
            world_size=world_size,
            timeout=datetime.timedelta(seconds=60)
        )
        
        logging.info(f"{LOG_PREFIX} [Worker-{rank}] torch.distributed initialized: world_size={dist.get_world_size()}, rank={dist.get_rank()}")
        
        # Initialize DeviceMesh for parallel state
        from comfy.parallel_attention.parallel_state import initialize_parallel_state
        
        # Default: SP=world_size, DP=1 (pure sequence parallel)
        # Can be configured via environment or kwargs in future
        sp_size = world_size
        dp_size = 1
        initialize_parallel_state(sp_size=sp_size, dp_size=dp_size)
        
        logging.info(f"{LOG_PREFIX} [Worker-{rank}] DeviceMesh initialized")
        
        # Signal ready to main process
        pipe.send({"status": "ready"})
        
        # Event loop
        while True:
            if not pipe.poll(timeout=None):
                continue
                
            request = pipe.recv()
            method = request.get("method")
            
            if method == "shutdown":
                logging.info(f"{LOG_PREFIX} [Worker-{rank}] Received shutdown signal")
                break
            
            elif method == "echo":
                result = request["kwargs"]["message"]
                pipe.send({"status": "success", "result": result})
            
            elif method == "allreduce_test":
                tensor = torch.tensor([rank], dtype=torch.float32, device=device)
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                result = tensor.item()
                pipe.send({"status": "success", "result": result})
            
            elif method == "devicemesh_test":
                from comfy.parallel_attention.parallel_state import (
                    get_device_mesh, get_sp_group, get_sp_rank, get_sp_size,
                    get_dp_rank, get_dp_size
                )
                
                # Get mesh info
                mesh = get_device_mesh()
                sp_rank = get_sp_rank()
                sp_size = get_sp_size()
                dp_rank = get_dp_rank()
                dp_size = get_dp_size()
                sp_group = get_sp_group()
                
                logging.info(f"{LOG_PREFIX} [Worker-{rank}] DeviceMesh test: mesh={mesh}, sp_rank={sp_rank}/{sp_size}, dp_rank={dp_rank}/{dp_size}")
                
                # Test all_gather on SP group
                tensor = torch.tensor([float(sp_rank)], dtype=torch.float32, device=device)
                gathered = [torch.zeros_like(tensor) for _ in range(sp_size)]
                dist.all_gather(gathered, tensor, group=sp_group)
                gathered_list = [t.item() for t in gathered]
                
                logging.info(f"{LOG_PREFIX} [Worker-{rank}] SP all_gather result: {gathered_list}")
                
                result = {
                    "rank": rank,
                    "sp_rank": sp_rank,
                    "sp_size": sp_size,
                    "dp_rank": dp_rank,
                    "dp_size": dp_size,
                    "mesh_shape": list(mesh.mesh.shape),
                    "gathered": gathered_list
                }
                pipe.send({"status": "success", "result": result})
            
            elif method == "test_fsdp_policy":
                from comfy.parallel_attention.fsdp2_policies import FSDP2PolicyRegistry
                
                # Get policy name from kwargs
                model_name = request["kwargs"].get("model_name", "flux")
                
                logging.info(f"{LOG_PREFIX} [Worker-{rank}] Testing FSDP2 policy: {model_name}")
                
                # Test registry operations
                is_registered = FSDP2PolicyRegistry.is_registered(model_name)
                available_policies = FSDP2PolicyRegistry.list_registered()
                
                logging.info(f"{LOG_PREFIX} [Worker-{rank}] Policy registered: {is_registered}")
                logging.info(f"{LOG_PREFIX} [Worker-{rank}] Available policies: {available_policies}")
                
                if is_registered:
                    # Get and instantiate policy
                    policy_fn = FSDP2PolicyRegistry.get_policy(model_name)
                    policy = policy_fn()
                    
                    logging.info(f"{LOG_PREFIX} [Worker-{rank}] Policy retrieved successfully")
                    logging.info(f"{LOG_PREFIX} [Worker-{rank}] Policy type: {type(policy).__name__}")
                    
                    result = {
                        "rank": rank,
                        "model_name": model_name,
                        "is_registered": is_registered,
                        "available_policies": available_policies,
                        "policy_callable": callable(policy),
                        "policy_type": type(policy).__name__
                    }
                else:
                    result = {
                        "rank": rank,
                        "model_name": model_name,
                        "is_registered": is_registered,
                        "available_policies": available_policies,
                        "policy_callable": False,
                        "policy_type": None
                    }
                
                pipe.send({"status": "success", "result": result})
            
            elif method == "test_fsdp2_api":
                # NEW TEST: Validate FSDP2 API migration
                logging.info(f"{LOG_PREFIX} [Worker-{rank}] ╔══════════════════════════════════════════════════════════╗")
                logging.info(f"{LOG_PREFIX} [Worker-{rank}] ║         FSDP2 API MIGRATION VALIDATION TEST              ║")
                logging.info(f"{LOG_PREFIX} [Worker-{rank}] ╚══════════════════════════════════════════════════════════╝")
                
                validation_results = {}
                
                # Test 1: Check FSDP2ModelPatcher has FSDP2 imports
                try:
                    from comfy.parallel_attention.fsdp2_model_patcher import FSDP2ModelPatcher
                    import inspect
                    
                    # Check imports in module
                    import comfy.parallel_attention.fsdp2_model_patcher as patcher_module
                    source = inspect.getsource(patcher_module)
                    
                    has_fully_shard_import = 'from torch.distributed.fsdp import fully_shard' in source
                    has_old_fsdp_import = 'from torch.distributed.fsdp import FullyShardedDataParallel as FSDP' in source
                    
                    validation_results["fsdp2_import"] = has_fully_shard_import
                    validation_results["no_fsdp1_import"] = not has_old_fsdp_import
                    
                    logging.info(f"{LOG_PREFIX} [Worker-{rank}] ✓ Import check: fully_shard={has_fully_shard_import}, no_FSDP1={not has_old_fsdp_import}")
                except Exception as e:
                    validation_results["fsdp2_import"] = False
                    validation_results["no_fsdp1_import"] = False
                    logging.error(f"{LOG_PREFIX} [Worker-{rank}] ✗ Import check failed: {e}")
                
                # Test 2: Check _get_modules_for_policy method exists
                try:
                    has_helper_method = hasattr(FSDP2ModelPatcher, '_get_modules_for_policy')
                    validation_results["has_helper_method"] = has_helper_method
                    logging.info(f"{LOG_PREFIX} [Worker-{rank}] ✓ Helper method exists: {has_helper_method}")
                except Exception as e:
                    validation_results["has_helper_method"] = False
                    logging.error(f"{LOG_PREFIX} [Worker-{rank}] ✗ Helper method check failed: {e}")
                
                # Test 3: Check _wrap_with_fsdp uses fully_shard (not FSDP wrapper)
                try:
                    method_source = inspect.getsource(FSDP2ModelPatcher._wrap_with_fsdp)
                    uses_fully_shard = 'fully_shard(' in method_source
                    uses_old_fsdp = 'FSDP(' in method_source and 'self.model = FSDP' in method_source
                    
                    validation_results["uses_fully_shard"] = uses_fully_shard
                    validation_results["no_fsdp_wrapper"] = not uses_old_fsdp
                    
                    logging.info(f"{LOG_PREFIX} [Worker-{rank}] ✓ Wrapping check: fully_shard={uses_fully_shard}, no_FSDP_wrapper={not uses_old_fsdp}")
                except Exception as e:
                    validation_results["uses_fully_shard"] = False
                    validation_results["no_fsdp_wrapper"] = False
                    logging.error(f"{LOG_PREFIX} [Worker-{rank}] ✗ Wrapping check failed: {e}")
                
                # Test 4: Check DTensor detection (not isinstance FSDP check)
                try:
                    method_source = inspect.getsource(FSDP2ModelPatcher._wrap_with_fsdp)
                    has_dtensor_check = 'DTensor' in method_source
                    has_old_isinstance = 'isinstance(module, FSDP)' in method_source
                    
                    validation_results["has_dtensor_check"] = has_dtensor_check
                    validation_results["no_isinstance_fsdp"] = not has_old_isinstance
                    
                    logging.info(f"{LOG_PREFIX} [Worker-{rank}] ✓ Verification check: DTensor={has_dtensor_check}, no_isinstance_FSDP={not has_old_isinstance}")
                except Exception as e:
                    validation_results["has_dtensor_check"] = False
                    validation_results["no_isinstance_fsdp"] = False
                    logging.error(f"{LOG_PREFIX} [Worker-{rank}] ✗ Verification check failed: {e}")
                
                # Test 5: Check reshard_after_forward (not ShardingStrategy enum)
                try:
                    method_source = inspect.getsource(FSDP2ModelPatcher._wrap_with_fsdp)
                    has_reshard_after_forward = 'reshard_after_forward' in method_source
                    uses_sharding_strategy = 'ShardingStrategy.FULL_SHARD' in method_source
                    
                    validation_results["has_reshard_after_forward"] = has_reshard_after_forward
                    validation_results["no_sharding_strategy_enum"] = not uses_sharding_strategy
                    
                    logging.info(f"{LOG_PREFIX} [Worker-{rank}] ✓ Config check: reshard_after_forward={has_reshard_after_forward}, no_ShardingStrategy={not uses_sharding_strategy}")
                except Exception as e:
                    validation_results["has_reshard_after_forward"] = False
                    validation_results["no_sharding_strategy_enum"] = False
                    logging.error(f"{LOG_PREFIX} [Worker-{rank}] ✗ Config check failed: {e}")
                
                # Summary
                all_checks = [
                    validation_results.get("fsdp2_import", False),
                    validation_results.get("no_fsdp1_import", False),
                    validation_results.get("has_helper_method", False),
                    validation_results.get("uses_fully_shard", False),
                    validation_results.get("no_fsdp_wrapper", False),
                    validation_results.get("has_dtensor_check", False),
                    validation_results.get("no_isinstance_fsdp", False),
                    validation_results.get("has_reshard_after_forward", False),
                    validation_results.get("no_sharding_strategy_enum", False),
                ]
                
                passed = sum(all_checks)
                total = len(all_checks)
                validation_results["passed_checks"] = f"{passed}/{total}"
                validation_results["all_passed"] = passed == total
                
                if validation_results["all_passed"]:
                    logging.info(f"{LOG_PREFIX} [Worker-{rank}] ✅ FSDP2 API MIGRATION: ALL CHECKS PASSED ({passed}/{total})")
                else:
                    logging.error(f"{LOG_PREFIX} [Worker-{rank}] ❌ FSDP2 API MIGRATION: FAILED ({passed}/{total})")
                
                logging.info(f"{LOG_PREFIX} [Worker-{rank}] ╚══════════════════════════════════════════════════════════╝")
                
                pipe.send({"status": "success", "result": validation_results})
            
            elif method == "test_fsdp_load":
                from comfy.parallel_attention.fsdp2_registry import detect_model_type, get_fsdp2_strategy
                from comfy.parallel_attention.fsdp2_model_patcher import FSDP2ModelPatcher
                import comfy.utils
                
                # Get state dict from kwargs (passed from rank 0)
                model_name = request["kwargs"].get("model_name", "unknown")
                state_dict_sample = request["kwargs"].get("state_dict_sample", {})
                detected_type_override = request["kwargs"].get("detected_type_override", None)
                
                logging.info(f"{LOG_PREFIX} [Worker-{rank}] Testing FSDP load detection: {model_name}")
                logging.info(f"{LOG_PREFIX} [Worker-{rank}] State dict keys sample: {list(state_dict_sample.keys())[:5]}")
                
                # Test 1: Model type detection (or use override)
                if detected_type_override:
                    detected_type = detected_type_override
                    logging.info(f"{LOG_PREFIX} [Worker-{rank}] Using override model type: {detected_type}")
                else:
                    detected_type = detect_model_type(state_dict_sample)
                    logging.info(f"{LOG_PREFIX} [Worker-{rank}] Detected model type: {detected_type}")
                
                # Test 2: Get FSDP2 strategy
                try:
                    if detected_type:
                        policy_fn = get_fsdp2_strategy(model_type=detected_type)
                        policy = policy_fn()
                        policy_retrieved = True
                        logging.info(f"{LOG_PREFIX} [Worker-{rank}] FSDP policy retrieved: {type(policy).__name__}")
                    else:
                        policy_retrieved = False
                        logging.warning(f"{LOG_PREFIX} [Worker-{rank}] Could not detect model type")
                except Exception as e:
                    policy_retrieved = False
                    logging.error(f"{LOG_PREFIX} [Worker-{rank}] Failed to get policy: {e}")
                
                # Test 3: Calculate model parameters
                if state_dict_sample:
                    try:
                        params = comfy.utils.calculate_parameters(state_dict_sample)
                        params_gb = params / 1e9
                        logging.info(f"{LOG_PREFIX} [Worker-{rank}] Model parameters: {params_gb:.2f}B")
                    except Exception as e:
                        params = 0
                        logging.error(f"{LOG_PREFIX} [Worker-{rank}] Failed to calculate params: {e}")
                else:
                    params = 0
                
                # Test 4: Verify FSDP imports work
                try:
                    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
                    from torch.distributed.fsdp import ShardingStrategy, MixedPrecision
                    fsdp_imports_ok = True
                    logging.info(f"{LOG_PREFIX} [Worker-{rank}] FSDP imports successful")
                except Exception as e:
                    fsdp_imports_ok = False
                    logging.error(f"{LOG_PREFIX} [Worker-{rank}] FSDP imports failed: {e}")
                
                result = {
                    "rank": rank,
                    "model_name": model_name,
                    "detected_type": detected_type,
                    "policy_retrieved": policy_retrieved,
                    "params": params,
                    "fsdp_imports_ok": fsdp_imports_ok,
                    "device": str(device),
                }
                
                logging.info(f"{LOG_PREFIX} [Worker-{rank}] FSDP load test complete: detected={detected_type}, policy={policy_retrieved}")
                
                pipe.send({"status": "success", "result": result})
            
            elif method == "load_fsdp_model":
                # Load FSDP model from file path
                # Args: unet_path, model_options, scaffold
                logging.info(f"{LOG_PREFIX} [Worker-{rank}] Loading FSDP model...")
                
                try:
                    unet_path = request["kwargs"].get("unet_path")
                    model_options = request["kwargs"].get("model_options", {})
                    scaffold = request["kwargs"].get("scaffold")
                    
                    if scaffold:
                        logging.info(
                            f"{LOG_PREFIX} [Worker-{rank}] Received scaffold: "
                            f"type={scaffold.get('model_type')}, "
                            f"latent_format={scaffold.get('latent_format', {}).get('class_name')}"
                        )
                    
                    if not unet_path:
                        raise ValueError("unet_path required for load_fsdp_model")
                    
                    # Import at worker level (after torch.distributed initialized)
                    import comfy.sd
                    
                    # Load model - this will use FSDP loading if fsdp.enabled=True
                    model = comfy.sd.load_diffusion_model(unet_path, model_options=model_options)
                    
                    # Store in worker context for forward_pass calls
                    WorkerContext.model_patcher = model
                    WorkerContext.rank = rank
                    WorkerContext.world_size = world_size
                    
                    # Get dtype from model
                    if hasattr(model, 'model') and hasattr(model.model, 'get_dtype'):
                        dtype = str(model.model.get_dtype()).split('.')[-1]  # 'torch.bfloat16' -> 'bfloat16'
                    else:
                        dtype = 'bfloat16'  # Default for Flux
                    
                    # Return model info (can't send actual model object via pipe)
                    result = {
                        "success": True,
                        "model_type": type(model).__name__,
                        "is_fsdp": hasattr(model, 'is_fsdp_wrapped') and model.is_fsdp_wrapped,
                        "dtype": dtype,
                    }
                    
                    logging.info(
                        f"{LOG_PREFIX} [Worker-{rank}] Model loaded and stored: "
                        f"type={result['model_type']}, fsdp={result['is_fsdp']}, dtype={dtype}"
                    )
                    
                    pipe.send({"status": "success", "result": result})
                    
                except Exception as e:
                    import traceback
                    error_msg = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
                    logging.error(f"{LOG_PREFIX} [Worker-{rank}] Model load failed: {error_msg}")
                    
                    pipe.send({"status": "success", "result": {
                        "success": False,
                        "error": error_msg
                    }})
            
            elif method == "forward_pass":
                # Execute forward pass with FSDP model
                logging.info(f"{LOG_PREFIX} [Worker-{rank}] Executing forward pass...")
                
                try:
                    # Get inputs
                    inputs = request["kwargs"]
                    x = inputs["x"]
                    t = inputs["t"]
                    c_concat = inputs.get("c_concat")
                    c_crossattn = inputs.get("c_crossattn")
                    control = inputs.get("control")
                    transformer_options = inputs.get("transformer_options", {})
                    
                    # Get model from worker context
                    if not hasattr(WorkerContext, 'model_patcher') or WorkerContext.model_patcher is None:
                        raise RuntimeError("Model not loaded in worker")
                    
                    model_patcher = WorkerContext.model_patcher
                    
                    # Move inputs to device
                    device = model_patcher.load_device
                    x = x.to(device)
                    t = t.to(device) if torch.is_tensor(t) else t
                    
                    if c_crossattn is not None:
                        c_crossattn = c_crossattn.to(device)
                    if c_concat is not None:
                        c_concat = c_concat.to(device)
                    
                    # Execute forward pass
                    # model_patcher.model is BaseModel which has apply_model()
                    with torch.no_grad():
                        output = model_patcher.model.apply_model(
                            x, t,
                            c_concat=c_concat,
                            c_crossattn=c_crossattn,
                            control=control,
                            transformer_options=transformer_options
                        )
                    
                    # Move output back to CPU for pipe serialization
                    output_cpu = output.cpu()
                    
                    if rank == 0:
                        logging.info(
                            f"{LOG_PREFIX} [Worker-{rank}] Forward pass complete: "
                            f"input={tuple(x.shape)}, output={tuple(output.shape)}"
                        )
                    
                    # Only rank 0 returns result (all ranks compute identical output)
                    if rank == 0:
                        pipe.send({"status": "success", "result": {"output": output_cpu}})
                    else:
                        # Other ranks signal completion but don't send output
                        pipe.send({"status": "success", "result": {}})
                    
                except Exception as e:
                    import traceback
                    error_msg = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
                    logging.error(f"{LOG_PREFIX} [Worker-{rank}] Forward pass failed: {error_msg}")
                    
                    pipe.send({"status": "success", "result": {
                        "error": error_msg
                    }})
            
            else:
                error_msg = f"Unknown method: {method}"
                logging.error(f"{LOG_PREFIX} [Worker-{rank}] {error_msg}")
                pipe.send({
                    "status": "error",
                    "error": {
                        "type": "NotImplementedError",
                        "message": error_msg,
                        "traceback": ""
                    }
                })
    
    except Exception as e:
        import traceback
        
        logging.error(f"{LOG_PREFIX} [Worker-{rank}] Worker failed: {type(e).__name__}: {e}")
        logging.error(traceback.format_exc())
        
        error_msg = {
            "status": "error",
            "error": {
                "message": str(e),
                "type": type(e).__name__,
                "traceback": traceback.format_exc()
            }
        }
        try:
            pipe.send(error_msg)
        except:
            logging.error(f"{LOG_PREFIX} [Worker-{rank}] Failed to send error message to main process")
        
        raise
    
    finally:
        if dist.is_initialized():
            import time
            start = time.time()
            logging.info(f"{LOG_PREFIX} [Worker-{rank}] Destroying process group")
            dist.destroy_process_group()
            elapsed = time.time() - start
            logging.info(f"{LOG_PREFIX} [Worker-{rank}] Process group destroyed in {elapsed:.3f}s")
        
        logging.info(f"{LOG_PREFIX} [Worker-{rank}] Worker exiting")
