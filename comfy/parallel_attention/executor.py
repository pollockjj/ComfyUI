"""Multiprocess executor for distributed inference."""

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import socket
import logging
import os
import time

logger = logging.getLogger(__name__)

# Logging prefix for visibility in ComfyUI logs
LOG_PREFIX = "⚡ [Parallel-Attention]"

# Timeout constants
DEFAULT_INIT_TIMEOUT = 60.0  # seconds to wait for torch.distributed initialization
DEFAULT_RPC_TIMEOUT = 60.0   # seconds to wait for RPC responses

# Worker shutdown timeouts
WORKER_SHUTDOWN_TIMEOUT = 40.0  # seconds to wait for worker to exit gracefully
WORKER_TERMINATE_TIMEOUT = 40.0  # seconds to wait after terminate before kill

# World size constants
DEFAULT_WORLD_SIZE = 2  # default number of workers when device count allows
MIN_CUDA_DEVICES = 2    # minimum CUDA devices required for distributed runtime

@dataclass
class WorkerContext:
    """Worker process context."""
    rank: int
    proc: mp.Process
    pipe: Any  # multiprocessing.Connection
    device: Optional[torch.device] = None

class MultiprocExecutor:
    """Spawns worker processes and manages torch.distributed initialization."""
    
    def __init__(self, 
                 world_size: Optional[int] = None, 
                 backend: str = "nccl",
                 timeout: float = DEFAULT_INIT_TIMEOUT):
        """Initialize multiprocess executor."""
        # Default world_size to 2 if we have at least 2 CUDA devices
        if world_size is None:
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                if device_count >= MIN_CUDA_DEVICES:
                    world_size = DEFAULT_WORLD_SIZE
                else:
                    raise RuntimeError(f"Distributed runtime requires at least {MIN_CUDA_DEVICES} CUDA devices, found {device_count}")
            else:
                raise RuntimeError("Distributed runtime requires CUDA devices, none available")
        
        if world_size < MIN_CUDA_DEVICES:
            raise ValueError(f"world_size must be >= {MIN_CUDA_DEVICES}, got {world_size}")
        
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            if world_size > device_count:
                raise ValueError(f"world_size ({world_size}) exceeds available CUDA devices ({device_count})")
        
        self.world_size = world_size
        self.backend = self._select_backend(backend)
        self.timeout = timeout
        self.workers: List[WorkerContext] = []
        
        self.init_method = self._get_init_method()
        
        logger.info(f"{LOG_PREFIX} [Executor] Initializing: world_size={world_size}, backend={self.backend}")
        
        self._spawn_workers()
        self._wait_for_ready()
        
        logger.info(f"{LOG_PREFIX} [Executor] Initialized: {world_size} workers ready")
    
    def _select_backend(self, requested: str) -> str:
        """Auto-detect best backend for platform."""
        if requested == "auto":
            requested = "nccl"
            
        if requested == "nccl":
            if not torch.cuda.is_available():
                logger.warning(f"{LOG_PREFIX} [Executor] NCCL requires CUDA, falling back to GLOO")
                return "gloo"
            if os.name == "nt":
                logger.warning(f"{LOG_PREFIX} [Executor] NCCL not supported on Windows, using GLOO")
                return "gloo"
            try:
                version = torch.cuda.nccl.version()
                logger.debug(f"{LOG_PREFIX} [Executor] NCCL version: {version}")
            except Exception as e:
                logger.warning(f"{LOG_PREFIX} [Executor] NCCL not available: {e}, falling back to GLOO")
                return "gloo"
        
        return requested
    
    def _get_init_method(self) -> str:
        """Find free port for distributed init."""
        s = socket.socket()
        s.bind(('', 0))
        port = s.getsockname()[1]
        s.close()
        init_method = f"tcp://127.0.0.1:{port}"
        logger.debug(f"{LOG_PREFIX} [Executor] Using init_method: {init_method}")
        return init_method
    
    def _spawn_workers(self):
        """Spawn worker processes with pipes."""
        from comfy.parallel_attention.worker import worker_main
        
        logger.info(f"{LOG_PREFIX} [Executor] Spawning {self.world_size} workers")
        
        for rank in range(self.world_size):
            pipe_main, pipe_worker = mp.Pipe()
            
            if torch.cuda.is_available():
                device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
            else:
                device = torch.device("cpu")
            
            proc = mp.Process(
                target=worker_main,
                args=(rank, self.world_size, self.init_method, self.backend, pipe_worker),
                daemon=False,
                name=f"DistWorker-{rank}"
            )
            proc.start()
            
            self.workers.append(WorkerContext(rank, proc, pipe_main, device))
            logger.debug(f"{LOG_PREFIX} [Executor] Spawned rank {rank}: PID {proc.pid}, device {device}")
    
    def _wait_for_ready(self, timeout: Optional[float] = None):
        """Wait for all workers to signal READY."""
        if timeout is None:
            timeout = self.timeout
            
        start = time.time()
        logger.info(f"{LOG_PREFIX} [Executor] Waiting for workers to initialize torch.distributed")
        
        for worker in self.workers:
            remaining = timeout - (time.time() - start)
            if remaining <= 0:
                raise TimeoutError(
                    f"Worker rank {worker.rank} did not signal READY in {timeout}s"
                )
            
            if not worker.pipe.poll(timeout=remaining):
                raise TimeoutError(
                    f"Worker rank {worker.rank} timed out during initialization"
                )
            
            msg = worker.pipe.recv()
            
            if msg.get("status") == "error":
                error_info = msg.get("error", {})
                raise RuntimeError(
                    f"Worker rank {worker.rank} failed initialization:\n"
                    f"  Type: {error_info.get('type', 'Unknown')}\n"
                    f"  Message: {error_info.get('message', 'Unknown')}\n"
                    f"  Traceback:\n{error_info.get('traceback', 'Not available')}"
                )
            
            if msg.get("status") != "ready":
                raise RuntimeError(
                    f"Worker rank {worker.rank} unexpected response: {msg}"
                )
            
            logger.debug(f"{LOG_PREFIX} [Executor] Rank {worker.rank} ready")
        
        elapsed = time.time() - start
        logger.info(f"{LOG_PREFIX} [Executor] All workers initialized in {elapsed:.2f}s")
    
    def execute_collective(self, 
                          method: str, 
                          kwargs: Dict[str, Any],
                          timeout: Optional[float] = None) -> Any:
        """Execute method on all workers, return rank 0 result."""
        if timeout is None:
            timeout = self.timeout
        
        logger.debug(f"{LOG_PREFIX} [Executor] Executing {method} with {len(kwargs)} args")
        
        request = {"method": method, "kwargs": kwargs}
        for worker in self.workers:
            worker.pipe.send(request)
        
        results = []
        start = time.time()
        
        for worker in self.workers:
            remaining = timeout - (time.time() - start)
            if remaining <= 0:
                raise TimeoutError(
                    f"Worker rank {worker.rank} timed out executing {method}"
                )
            
            if not worker.pipe.poll(timeout=remaining):
                raise TimeoutError(
                    f"Worker rank {worker.rank} timed out after {timeout}s"
                )
            
            response = worker.pipe.recv()
            
            if response.get("status") == "error":
                error_info = response.get("error", {})
                raise RuntimeError(
                    f"Worker rank {worker.rank} failed executing {method}:\n"
                    f"  Type: {error_info.get('type', 'Unknown')}\n"
                    f"  Message: {error_info.get('message', 'Unknown')}\n"
                    f"  Traceback:\n{error_info.get('traceback', 'Not available')}"
                )
            
            results.append(response.get("result"))
            logger.debug(f"{LOG_PREFIX} [Executor] Rank {worker.rank} completed {method}")
        
        elapsed = time.time() - start
        logger.debug(f"{LOG_PREFIX} [Executor] {method} completed in {elapsed:.3f}s")
        
        return results[0]
    
    def shutdown(self):
        """Clean shutdown of all workers."""
        logger.info(f"{LOG_PREFIX} [Executor] Shutting down workers")
        
        # Send shutdown signal to all workers simultaneously
        for worker in self.workers:
            try:
                logger.debug(f"{LOG_PREFIX} [Executor] Sending shutdown to rank {worker.rank}")
                worker.pipe.send({"method": "shutdown"})
            except Exception as e:
                logger.warning(f"{LOG_PREFIX} [Executor] Error sending shutdown to rank {worker.rank}: {e}")
        
        # Wait for all workers to exit
        for worker in self.workers:
            try:
                worker.proc.join(timeout=WORKER_SHUTDOWN_TIMEOUT)
                
                if worker.proc.is_alive():
                    logger.warning(f"{LOG_PREFIX} [Executor] Rank {worker.rank} did not exit cleanly, terminating")
                    worker.proc.terminate()
                    worker.proc.join(timeout=WORKER_TERMINATE_TIMEOUT)
                    
                    if worker.proc.is_alive():
                        logger.error(f"{LOG_PREFIX} [Executor] Rank {worker.rank} force kill required")
                        worker.proc.kill()
                else:
                    logger.debug(f"{LOG_PREFIX} [Executor] Rank {worker.rank} shutdown complete")
                    
            except Exception as e:
                logger.warning(f"{LOG_PREFIX} [Executor] Error shutting down rank {worker.rank}: {e}")
        
        logger.info(f"{LOG_PREFIX} [Executor] Shutdown complete")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with automatic shutdown."""
        self.shutdown()
        return False
