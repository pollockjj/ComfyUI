"""FSDP2Worker - Distributed inference worker with FSDP2 sharding.

Implements multi-GPU model sharding using PyTorch FSDP2 and DeviceMesh.
Based on FastVideo worker pattern.
"""

import torch
import logging

LOG_PREFIX = "⚡ [Parallel-Attention]"


def log_rank0(rank: int, level: str, message: str):
    """Log message only for rank 0 (info), others use debug.
    
    Args:
        rank: Worker rank
        level: 'info', 'warning', or 'error'
        message: Log message
    """
    if rank == 0:
        getattr(logging, level)(message)
    else:
        logging.debug(message)


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
        
        log_rank0(rank, 'info', f"{LOG_PREFIX} Worker-{rank} initialized on {self.device}")
        log_rank0(rank, 'info', f"{LOG_PREFIX} Worker-{rank} DeviceMesh: {device_type} mesh_shape=({world_size},)")
    
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
        
        elif command == "initialize_fsdp2_from_checkpoint":
            return self._initialize_fsdp2_from_checkpoint(args)
        
        elif command == "cleanup_fsdp2_model":
            return self._cleanup_fsdp2_model(args)
        
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
    
    def _initialize_fsdp2_from_checkpoint(self, args: dict):
        """Initialize FSDP2 by creating meta model from checkpoint.
        
        Phase 0.5.1: Create meta model and apply FSDP2 structure (0GB).
        
        Steps:
        1. Load checkpoint state_dict
        2. Create meta model using ComfyUI infrastructure
        3. Apply FSDP2 sharding to meta model
        
        Args:
            args: {
                "checkpoint_path": str,
                "model_type": str
            }
        
        Returns:
            {
                "status": "success",
                "vram_gb": float,
                "rank": int,
                "total_params": int,
                "meta_params": int
            }
        """
        checkpoint_path = args.get("checkpoint_path")
        model_type = args.get("model_type")
        
        if not checkpoint_path:
            return {"status": "error", "error": "No checkpoint_path provided"}
        
        LOG_PREFIX = f"⚡ [Parallel-Attention][Worker][Rank {self.rank}]"
        
        try:
            # Load checkpoint state_dict
            log_rank0(self.rank, 'debug', f"{LOG_PREFIX} Loading checkpoint...")
            import comfy.utils
            sd = comfy.utils.load_torch_file(checkpoint_path)
            log_rank0(self.rank, 'debug', f"{LOG_PREFIX} Loaded {len(sd)} keys")
            
            # Create meta model
            log_rank0(self.rank, 'info', f"{LOG_PREFIX} Meta model created: {model_type}")
            import comfy.model_detection
            
            with torch.device("meta"):
                model_config = comfy.model_detection.model_config_from_unet(sd, "")
                if model_config is None:
                    return {"status": "error", "error": "Could not detect model type from checkpoint"}
                
                model = model_config.get_model(sd, "")
            
            # Free state_dict memory
            del sd
            import gc
            gc.collect()
            
            # Apply FSDP2 sharding
            from comfy.parallel_attention.fsdp2_engine import apply_fsdp2_sharding_structure_only
            
            policy = args["policy"]
            log_rank0(self.rank, 'debug', f"{LOG_PREFIX} Applying FSDP2 sharding...")
            
            fsdp_model = apply_fsdp2_sharding_structure_only(
                model,
                policy,
                self.device_mesh
            )
            
            # Load weights using iterator
            log_rank0(self.rank, 'info', f"{LOG_PREFIX} Loading weights...")
            
            from safetensors.torch import safe_open
            from torch.distributed._tensor import distribute_tensor
            
            # Get meta state dict for key lookup (works with FSDP2 wrapping)
            meta_sd = fsdp_model.state_dict()
            sharded_sd = {}
            
            log_rank0(self.rank, 'debug', f"{LOG_PREFIX} Streaming from safetensors...")
            
            with safe_open(checkpoint_path, framework="pt", device="cpu") as f:
                tensor_count = 0
                for param_name in f.keys():
                    # Load ONE tensor at a time
                    full_tensor = f.get_tensor(param_name)
                    
                    # Try exact match
                    meta_param = meta_sd.get(param_name)
                    
                    # Fallback: try with diffusion_model prefix
                    if meta_param is None and not param_name.startswith("diffusion_model."):
                        prefixed_name = f"diffusion_model.{param_name}"
                        meta_param = meta_sd.get(prefixed_name)
                        if meta_param is not None:
                            param_name = prefixed_name
                    
                    if meta_param is None:
                        continue
                    
                    # Move to GPU
                    full_tensor = full_tensor.to(device=self.device)
                    
                    # Distribute if FSDP-wrapped (has device_mesh)
                    if hasattr(meta_param, "device_mesh"):
                        sharded_tensor = distribute_tensor(
                            full_tensor,
                            meta_param.device_mesh,
                            meta_param.placements,
                        )
                    else:
                        sharded_tensor = full_tensor
                    
                    sharded_sd[param_name] = torch.nn.Parameter(sharded_tensor)
                    tensor_count += 1
                    
                    # full_tensor goes out of scope, memory freed
            
            log_rank0(self.rank, 'debug', f"{LOG_PREFIX} Loaded {tensor_count} tensors")
            
            # Load sharded dict into model
            log_rank0(self.rank, 'debug', f"{LOG_PREFIX} Applying weights to model...")
            fsdp_model.load_state_dict(sharded_sd, assign=True, strict=False)
            
            # Measure VRAM
            torch.cuda.synchronize(self.device)
            vram_after_load = torch.cuda.memory_allocated(self.device) / (1024 ** 3)
            
            # Count sharded vs replicated params
            sharded_count = 0
            replicated_count = 0
            for name, param in fsdp_model.named_parameters():
                if hasattr(param, "device_mesh"):
                    sharded_count += 1
                else:
                    replicated_count += 1
            
            log_rank0(self.rank, 'info', f"{LOG_PREFIX} VRAM: {vram_after_load:.2f}GB, Params: {sharded_count} sharded, {replicated_count} replicated")
            
            if vram_after_load < 0.1:
                log_rank0(self.rank, 'warning', f"{LOG_PREFIX} No VRAM allocated after loading")
            
            # Store for cleanup
            self.model = fsdp_model
            
            return {
                "status": "success",
                "vram_gb": vram_after_load,
                "rank": self.rank,
                "sharded_count": sharded_count,
                "replicated_count": replicated_count,
            }
            
        except Exception as e:
            logging.error(f"{LOG_PREFIX} Error: {e}", exc_info=True)
            return {"status": "error", "error": str(e), "rank": self.rank}
    
    def _cleanup_fsdp2_model(self, args: dict):
        """Cleanup FSDP2 sharded model and free VRAM.
        
        Critical for Phase 0.5 to avoid OOM when returning to normal loading.
        
        Args:
            args: {} (no args needed)
        
        Returns:
            {
                "status": "success",
                "vram_freed_gb": float,
                "vram_after_gb": float,
                "rank": int
            }
        """
        LOG_PREFIX = f"⚡ [Parallel-Attention][Worker][Rank {self.rank}][Cleanup]"
        
        try:
            # Measure VRAM before cleanup
            torch.cuda.synchronize(self.device)
            vram_before = torch.cuda.memory_allocated(self.device) / (1024 ** 3)
            
            log_rank0(self.rank, 'debug', f"{LOG_PREFIX} Cleanup: {vram_before:.2f}GB before")
            
            # Delete sharded model
            if hasattr(self, 'model') and self.model is not None:
                del self.model
                self.model = None
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Clear CUDA cache
            torch.cuda.empty_cache()
            torch.cuda.synchronize(self.device)
            
            # Measure VRAM after cleanup
            vram_after = torch.cuda.memory_allocated(self.device) / (1024 ** 3)
            vram_freed = vram_before - vram_after
            
            log_rank0(self.rank, 'info', f"{LOG_PREFIX} Cleanup complete: freed {vram_freed:.2f}GB → {vram_after:.2f}GB")
            
            return {
                "status": "success",
                "vram_freed_gb": vram_freed,
                "vram_after_gb": vram_after,
                "rank": self.rank,
            }
            
        except Exception as e:
            logging.error(f"{LOG_PREFIX} Error during cleanup: {e}", exc_info=True)
            return {"status": "error", "error": str(e), "rank": self.rank}
