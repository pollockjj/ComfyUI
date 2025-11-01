"""MultiprocExecutor - FastVideo multiprocess worker management.

Spawns worker processes with torch.distributed for distributed inference.
Communication via multiprocessing.Pipe for control, NCCL/GLOO for tensors.

Based on FastVideo worker/multiproc_executor.py, adapted for ComfyUI integration.
"""

import multiprocessing as mp
import torch
import torch.distributed as dist
import os
import logging
from typing import Dict, Any, List, Optional
import time

LOG_PREFIX = "⚡ [Parallel-Attention]"


def select_backend() -> str:
    """Auto-select distributed backend.
    
    Returns:
        "nccl" if CUDA available, else "gloo"
    """
    if torch.cuda.is_available():
        return "nccl"
    return "gloo"


def worker_process(rank: int, world_size: int, master_addr: str, master_port: str,
                   backend: str, pipe_conn):
    """Worker process entry point.
    
    Args:
        rank: Worker rank (0 to world_size-1)
        world_size: Total number of workers
        master_addr: Master node address (localhost for single-machine)
        master_port: Master node port
        backend: "nccl" or "gloo"
        pipe_conn: Pipe connection for RPC communication with main process
    """
    try:
        # Set environment for torch.distributed
        os.environ['MASTER_ADDR'] = master_addr
        os.environ['MASTER_PORT'] = master_port
        os.environ['RANK'] = str(rank)
        os.environ['WORLD_SIZE'] = str(world_size)
        
        # Set CUDA device if using NCCL
        if backend == "nccl" and torch.cuda.is_available():
            torch.cuda.set_device(rank)
            device = torch.device(f'cuda:{rank}')
        else:
            device = torch.device('cpu')
        
        logging.info(f"{LOG_PREFIX} [Worker-{rank}] Initializing torch.distributed (backend={backend})")
        
        # Initialize torch.distributed
        dist.init_process_group(
            backend=backend,
            rank=rank,
            world_size=world_size,
        )
        
        logging.info(f"{LOG_PREFIX} [Worker-{rank}] Distributed initialized on device={device}")
        
        # Send ready signal to main process
        pipe_conn.send({"status": "ready", "rank": rank})
        
        # Worker event loop
        while True:
            # Wait for RPC call from main process
            msg = pipe_conn.recv()
            
            if msg.get("command") == "shutdown":
                logging.info(f"{LOG_PREFIX} [Worker-{rank}] Received shutdown signal")
                break
            
            # Execute RPC method
            method = msg.get("method")
            args = msg.get("args", {})
            
            try:
                # Create worker instance once and reuse
                if not hasattr(worker_process, '_worker_instance'):
                    from comfy.parallel_attention.fsdp2_worker import FSDP2Worker
                    worker_process._worker_instance = FSDP2Worker(rank=rank, world_size=world_size)
                
                result = worker_process._worker_instance.execute(method, args)
                
                # Send result back to main process
                pipe_conn.send({"status": "success", "result": result})
                
            except Exception as e:
                import traceback
                error_msg = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
                logging.error(f"{LOG_PREFIX} [Worker-{rank}] RPC error: {error_msg}")
                pipe_conn.send({"status": "error", "error": error_msg})
        
        # Cleanup
        dist.destroy_process_group()
        logging.info(f"{LOG_PREFIX} [Worker-{rank}] Shutdown complete")
        
    except Exception as e:
        import traceback
        error_msg = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        logging.error(f"{LOG_PREFIX} [Worker-{rank}] Fatal error: {error_msg}")
        pipe_conn.send({"status": "error", "error": error_msg})


class MultiprocExecutor:
    """Multiprocess executor for distributed workers.
    
    Spawns worker processes with torch.distributed initialization.
    Provides RPC interface for collective operations.
    
    Example:
        executor = MultiprocExecutor(world_size=2, backend="auto")
        result = executor.execute_collective("echo", {"message": "hello"})
        executor.shutdown()
    """
    
    def __init__(self, world_size: int, backend: str = "auto",
                 master_addr: str = "127.0.0.1", master_port: str = "29500"):
        """Initialize executor and spawn workers.
        
        Args:
            world_size: Number of worker processes
            backend: "nccl", "gloo", or "auto" (auto-select)
            master_addr: Master node address
            master_port: Master node port
        """
        self.world_size = world_size
        self.master_addr = master_addr
        self.master_port = master_port
        
        # Auto-select backend
        if backend == "auto":
            self.backend = select_backend()
        else:
            self.backend = backend
        
        logging.info(f"{LOG_PREFIX} [Executor] Initializing with world_size={world_size}, backend={self.backend}")
        
        # Spawn workers
        self.workers: List[mp.Process] = []
        self.pipes: List[Any] = []
        
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            # Already set, ignore
            pass
        
        for rank in range(world_size):
            parent_conn, child_conn = mp.Pipe()
            
            worker = mp.Process(
                target=worker_process,
                args=(rank, world_size, master_addr, master_port, self.backend, child_conn)
            )
            worker.start()
            
            self.workers.append(worker)
            self.pipes.append(parent_conn)
        
        # Wait for all workers to be ready
        self._wait_for_ready()
        
        logging.info(f"{LOG_PREFIX} [Executor] All workers ready")
    
    def _wait_for_ready(self, timeout: float = 30.0):
        """Wait for all workers to signal ready.
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Raises:
            TimeoutError: If workers don't become ready in time
        """
        start_time = time.time()
        ready_workers = set()
        
        while len(ready_workers) < self.world_size:
            if time.time() - start_time > timeout:
                raise TimeoutError(
                    f"Workers failed to initialize within {timeout}s. "
                    f"Ready: {len(ready_workers)}/{self.world_size}"
                )
            
            for rank, pipe in enumerate(self.pipes):
                if rank in ready_workers:
                    continue
                
                if pipe.poll(timeout=0.1):
                    msg = pipe.recv()
                    if msg.get("status") == "ready":
                        ready_workers.add(rank)
                        logging.info(f"{LOG_PREFIX} [Executor] Worker {rank} ready")
    
    def execute_collective(self, method: str, args: Dict[str, Any]) -> Any:
        """Execute RPC method on all workers.
        
        Sends same method call to all workers and waits for results.
        Returns result from rank 0.
        
        Args:
            method: Method name in worker module
            args: Arguments dict to pass to method
            
        Returns:
            Result from rank 0 worker
            
        Raises:
            RuntimeError: If any worker returns error
        """
        logging.debug(f"{LOG_PREFIX} [Executor] Executing collective: {method}")
        
        # Send RPC to all workers
        for pipe in self.pipes:
            pipe.send({
                "command": "execute",
                "method": method,
                "args": args
            })
        
        # Collect results
        results = []
        for rank, pipe in enumerate(self.pipes):
            msg = pipe.recv()
            
            if msg.get("status") == "error":
                error = msg.get("error", "Unknown error")
                raise RuntimeError(f"Worker {rank} error: {error}")
            
            results.append(msg.get("result"))
        
        # Return result from rank 0
        return results[0]
    
    def shutdown(self):
        """Shutdown all workers gracefully."""
        logging.info(f"{LOG_PREFIX} [Executor] Shutting down workers...")
        
        # Send shutdown to all workers
        for pipe in self.pipes:
            pipe.send({"command": "shutdown"})
        
        # Wait for workers to terminate
        for rank, worker in enumerate(self.workers):
            worker.join(timeout=5.0)
            if worker.is_alive():
                logging.warning(f"{LOG_PREFIX} [Executor] Worker {rank} did not terminate, killing...")
                worker.terminate()
                worker.join()
        
        logging.info(f"{LOG_PREFIX} [Executor] Shutdown complete")
    
    def __del__(self):
        """Ensure workers are shutdown on deletion."""
        if hasattr(self, 'workers') and hasattr(self, 'pipes'):
            try:
                self.shutdown()
            except:
                pass
