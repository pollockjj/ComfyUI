"""FSDP2Executor - Worker process management for distributed inference.

Manages multiple worker processes that hold FSDP2 sharded models.
Based on FastVideo MultiprocExecutor pattern using torch.multiprocessing.

Architecture:
- Parent process holds meta device model interface
- Worker processes hold FSDP2 sharded models
- Communication via multiprocessing.Pipe (RPC)
- torch.distributed for NCCL/GLOO collectives
- DeviceMesh for process group topology (PyTorch 2.2+)
"""

import torch
import torch.multiprocessing as mp
from multiprocessing.connection import Connection
import logging
import os

LOG_PREFIX = "⚡ [Parallel-Attention][FSDP2Executor]"


class FSDP2Executor:
    """Manages FSDP2 worker processes for distributed inference.
    
    Based on FastVideo MultiprocExecutor pattern.
    Uses torch.multiprocessing with 'spawn' start method.
    Uses DeviceMesh for process group topology (ARCHITECTURE.md requirement).
    """
    
    def __init__(self, world_size: int = 2, backend: str = "nccl"):
        self.world_size = world_size
        self.backend = backend if torch.cuda.is_available() else "gloo"
        self.workers = []
        self.pipes = []
        self.device_mesh = None  # Will be created after workers spawn
        
        self._spawn_workers()
        
        # Create DeviceMesh on parent process (ARCHITECTURE.md requirement)
        # Note: DeviceMesh creation requires distributed to be initialized in workers first
        # We'll pass mesh info to workers instead
        logging.info(f"{LOG_PREFIX} Spawned {world_size} workers (backend={self.backend})")
    
    def _find_free_port(self):
        """Find an available port to avoid collisions."""
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            s.listen(1)
            port = s.getsockname()[1]
        return port
    
    def _spawn_workers(self):
        """Spawn worker processes with torch.multiprocessing."""
        # Set spawn method
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass  # Already set
        
        # Find free port to avoid collisions
        port = self._find_free_port()
        logging.info(f"{LOG_PREFIX} Using port {port} for distributed init")
        
        for rank in range(self.world_size):
            parent_conn, child_conn = mp.Pipe()
            
            process = mp.Process(
                target=_worker_main,
                args=(rank, self.world_size, self.backend, child_conn, port),
                daemon=False
            )
            process.start()
            
            self.workers.append(process)
            self.pipes.append(parent_conn)
        
        # Wait for workers to be ready
        for i, pipe in enumerate(self.pipes):
            try:
                status = pipe.recv()
                if status != "READY":
                    raise RuntimeError(f"Worker {i} failed to initialize: {status}")
                logging.info(f"{LOG_PREFIX} Worker-{i} ready")
            except Exception as e:
                raise RuntimeError(f"Worker {i} initialization failed: {e}")
    
    def execute_collective(self, command: str, args: dict):
        """Execute command on all workers, wait for result from rank 0.
        
        Args:
            command: Worker function name (e.g., "forward", "initialize_fsdp2")
            args: Arguments to pass to worker function
        
        Returns:
            Result dictionary from rank 0 worker
        """
        # Add DeviceMesh info to args if initializing
        if command == "initialize_fsdp2":
            device_type = "cuda" if torch.cuda.is_available() else "cpu"
            args["device_mesh_info"] = {
                "device_type": device_type,
                "mesh_shape": [self.world_size],
                "mesh_dim_names": ["dp"],  # Data parallel dimension
            }
            logging.info(f"{LOG_PREFIX} Passing DeviceMesh info: {device_type} mesh shape ({self.world_size},)")
        
        # Send command to all workers
        for i, pipe in enumerate(self.pipes):
            try:
                pipe.send({"command": command, "args": args})
            except Exception as e:
                logging.error(f"{LOG_PREFIX} Failed to send to worker {i}: {e}")
                raise
        
        # Wait for result from rank 0
        try:
            result = self.pipes[0].recv()

            if isinstance(result, dict):
                if command == "initialize_fsdp2" and result.get("success"):
                    self.device_mesh = {
                        "device_type": device_type,
                        "mesh_shape": (self.world_size,),
                        "mesh_dim_names": ["dp"],
                        "world_size": self.world_size,
                    }
                    logging.info(f"{LOG_PREFIX} DeviceMesh marker set (actual mesh in workers)")

                if result.get("has_mesh"):
                    self.device_mesh = {
                        "device_type": result.get("device_type"),
                        "mesh_shape": tuple(result.get("mesh_shape", ())),
                        "mesh_dim_names": list(result.get("mesh_dim_names", [])),
                        "world_size": result.get("world_size", self.world_size),
                        "dp_size": result.get("dp_size"),
                    }
                    logging.info(
                        f"{LOG_PREFIX} DeviceMesh metadata cached: type={self.device_mesh['device_type']} "
                        f"shape={self.device_mesh['mesh_shape']}"
                    )

            return result
        except Exception as e:
            logging.error(f"{LOG_PREFIX} Failed to receive from rank 0: {e}")
            raise
    
    def shutdown(self):
        """Shutdown all workers gracefully."""
        logging.info(f"{LOG_PREFIX} Shutting down workers...")
        
        for i, pipe in enumerate(self.pipes):
            try:
                pipe.send({"command": "shutdown"})
            except:
                pass
        
        for i, worker in enumerate(self.workers):
            worker.join(timeout=5)
            if worker.is_alive():
                logging.warning(f"{LOG_PREFIX} Worker-{i} did not terminate, forcing...")
                worker.terminate()
                worker.join(timeout=2)


def _worker_main(rank: int, world_size: int, backend: str, pipe: Connection, port: int):
    """Worker process main loop.
    
    Initializes torch.distributed, creates FSDP2 model, handles RPC commands.
    """
    import torch.distributed as dist
    from comfy.parallel_attention.fsdp2_worker import FSDP2Worker
    
    try:
        # Initialize distributed
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = str(port)
        os.environ['RANK'] = str(rank)
        os.environ['WORLD_SIZE'] = str(world_size)
        
        dist.init_process_group(
            backend=backend,
            init_method=f"tcp://localhost:{port}",
            world_size=world_size,
            rank=rank
        )
        
        worker = FSDP2Worker(rank, world_size)
        pipe.send("READY")
        
        # Command loop
        while True:
            try:
                msg = pipe.recv()
                command = msg.get("command")
                args = msg.get("args", {})
                
                if command == "shutdown":
                    break
                
                # Execute command
                result = worker.execute(command, args)
                
                # Rank 0 sends result back
                if rank == 0:
                    pipe.send(result)
            except EOFError:
                break
            except Exception as e:
                if rank == 0:
                    pipe.send({"success": False, "error": str(e)})
                break
        
        dist.destroy_process_group()
        
    except Exception as e:
        pipe.send(f"ERROR: {e}")
        raise
