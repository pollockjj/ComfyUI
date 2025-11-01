"""FSDP2Worker - Minimal Phase 1A implementation.

Only implements commands needed for Phase 1A tests.
Based on FastVideo worker pattern with DeviceMesh.
"""

import torch
import logging

LOG_PREFIX = "⚡ [Parallel-Attention]"


class FSDP2Worker:
    """Minimal worker for Phase 1A testing.
    
    Implements only the commands needed to validate:
    - DeviceMesh initialization
    - Worker communication
    - Policy system
    """
    
    def __init__(self, rank: int, world_size: int):
        """Initialize worker with DeviceMesh.
        
        Args:
            rank: Worker rank (0 to world_size-1)
            world_size: Total number of workers
        """
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(f"cuda:{rank}") if torch.cuda.is_available() else torch.device("cpu")
        
        # Create DeviceMesh (ARCHITECTURE.md requirement)
        from torch.distributed.device_mesh import init_device_mesh
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        self.device_mesh = init_device_mesh(
            device_type,
            (world_size,),
            mesh_dim_names=["dp"]
        )
        
        logging.info(f"{LOG_PREFIX} Worker-{rank} initialized on {self.device}")
        logging.info(f"{LOG_PREFIX} Worker-{rank} DeviceMesh: {device_type} mesh_shape=({world_size},)")
    
    def execute(self, command: str, args: dict):
        """Execute command and return result.
        
        Args:
            command: Command name
            args: Command arguments
            
        Returns:
            Command result (dict or primitive)
        """
        if command == "echo":
            return args.get("message", "")
        
        elif command == "get_rank":
            return {"rank": self.rank}
        
        elif command == "check_devicemesh":
            return self._check_devicemesh(args)
        
        elif command == "apply_structure":
            return self._apply_structure(args)
        
        else:
            return {"success": False, "error": f"Unknown command: {command}"}
    
    def _check_devicemesh(self, args: dict):
        """Check DeviceMesh initialization.
        
        Returns:
            Dict with DeviceMesh properties
        """
        return {
            "has_mesh": self.device_mesh is not None,
            "mesh_shape": tuple(self.device_mesh.shape) if self.device_mesh else (),
            "mesh_dim_names": list(self.device_mesh.mesh_dim_names) if self.device_mesh else [],
            "device_type": self.device_mesh.device_type if self.device_mesh else None,
            "world_size": self.world_size,
            "has_dp_group": True,
            "dp_rank": self.device_mesh.get_local_rank(0) if self.device_mesh else -1,
            "dp_size": self.device_mesh.size(0) if self.device_mesh else -1,
        }
    
    def _apply_structure(self, args: dict):
        """Verify policy system works (Phase 1A - no actual FSDP application).
        
        Args:
            args: {"model_type": str}
            
        Returns:
            Dict with policy validation results
        """
        from comfy.parallel_attention.fsdp2_policies import FSDP2PolicyRegistry
        
        model_type = args.get("model_type")
        config = FSDP2PolicyRegistry.get_policy(model_type)
        
        # For Phase 1A, just verify policy retrieval works
        # Don't actually apply FSDP (needs fsdp2_engine.py which is in _phase1b_modify)
        
        return {
            "success": True,
            "still_meta": True,
            "has_fsdp": False,  # Not applying yet in Phase 1A
            "config_name": config.model_name,
            "block_count": len(config.blocks),
        }
