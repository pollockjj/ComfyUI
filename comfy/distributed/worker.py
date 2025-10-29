"""Worker process event loop."""

import torch
import torch.distributed as dist
import logging
import datetime
import os

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
        from comfy.distributed.parallel_state import initialize_parallel_state
        
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
                from comfy.distributed.parallel_state import (
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
                from comfy.distributed.fsdp_policies import FSDPPolicyRegistry
                
                # Get policy name from kwargs
                model_name = request["kwargs"].get("model_name", "flux")
                
                logging.info(f"{LOG_PREFIX} [Worker-{rank}] Testing FSDP policy: {model_name}")
                
                # Test registry operations
                is_registered = FSDPPolicyRegistry.is_registered(model_name)
                available_policies = FSDPPolicyRegistry.list_registered()
                
                logging.info(f"{LOG_PREFIX} [Worker-{rank}] Policy registered: {is_registered}")
                logging.info(f"{LOG_PREFIX} [Worker-{rank}] Available policies: {available_policies}")
                
                if is_registered:
                    # Get and instantiate policy
                    policy_fn = FSDPPolicyRegistry.get_policy(model_name)
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
            
            elif method == "test_fsdp_load":
                from comfy.distributed.fsdp_registry import detect_model_type, get_fsdp_strategy
                from comfy.distributed.fsdp_model_patcher import FSDPModelPatcher
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
                
                # Test 2: Get FSDP strategy
                try:
                    if detected_type:
                        policy_fn = get_fsdp_strategy(model_type=detected_type)
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
                # Args: unet_path, model_options
                logging.info(f"{LOG_PREFIX} [Worker-{rank}] Loading FSDP model...")
                
                try:
                    unet_path = request["kwargs"].get("unet_path")
                    model_options = request["kwargs"].get("model_options", {})
                    
                    if not unet_path:
                        raise ValueError("unet_path required for load_fsdp_model")
                    
                    # Import at worker level (after torch.distributed initialized)
                    import comfy.sd
                    
                    # Load model - this will use FSDP loading if fsdp.enabled=True
                    model = comfy.sd.load_diffusion_model(unet_path, model_options=model_options)
                    
                    # Return model info (can't send actual model object via pipe)
                    result = {
                        "success": True,
                        "model_type": type(model).__name__,
                        "is_fsdp": hasattr(model, 'is_fsdp_wrapped') and model.is_fsdp_wrapped,
                    }
                    
                    logging.info(
                        f"{LOG_PREFIX} [Worker-{rank}] Model loaded: "
                        f"type={result['model_type']}, fsdp={result['is_fsdp']}"
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
