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
