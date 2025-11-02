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
        """Initialize FSDP2 using FastVideo iterator pattern.
        
        Streams tensors one at a time from safetensors file.
        Peak memory = largest tensor (~500MB) not full model (22GB).
        
        Args:
            args: {
                "checkpoint_path": str,
                "model_type": str,
                "meta_model": nn.Module (cleaned, on meta device)
            }
        
        Returns:
            {
                "status": "success",
                "vram_gb": float,
                "rank": int,
                "sharded_count": int,
                "replicated_count": int
            }
        """
        from safetensors.torch import safe_open
        from torch.distributed.tensor import distribute_tensor
        from comfy.parallel_attention.fsdp2_policies import FSDP2PolicyRegistry
        from comfy.parallel_attention.fsdp2_engine import apply_fsdp2_sharding_structure_only
        
        LOG_PREFIX = f"⚡ [Parallel-Attention][Worker][Rank {self.rank}]"
        
        try:
            checkpoint_path = args.get("checkpoint_path")
            model_type = args.get("model_type")
            meta_model = args.get("meta_model")
            
            if not checkpoint_path:
                return {"status": "error", "error": "No checkpoint_path provided"}
            if not meta_model:
                return {"status": "error", "error": "No meta_model provided"}
            
            logging.info(f"{LOG_PREFIX} Initializing FSDP2 from checkpoint")
            logging.info(f"{LOG_PREFIX}   Model type: {model_type}")
            logging.info(f"{LOG_PREFIX}   Checkpoint: {checkpoint_path}")
            
            # Get policy
            config = FSDP2PolicyRegistry.get_policy(model_type)
            logging.info(f"{LOG_PREFIX} Policy: {len(config.blocks)} block groups")
            
            # Apply FSDP2 wrapping (structure only, no weights yet)
            logging.info(f"{LOG_PREFIX} Applying FSDP2 sharding structure...")
            fsdp_model = apply_fsdp2_sharding_structure_only(
                meta_model,
                config,
                self.device_mesh
            )
            logging.info(f"{LOG_PREFIX} FSDP2 structure applied")
            
            # Load weights using iterator (FastVideo pattern)
            meta_sd = fsdp_model.state_dict()
            sharded_sd = {}
            
            logging.info(f"{LOG_PREFIX} Streaming weights from file...")
            
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
            
            logging.info(f"{LOG_PREFIX} Loaded {tensor_count} tensors via iterator")
            
            # Load sharded dict into model
            logging.info(f"{LOG_PREFIX} Loading sharded state dict...")
            fsdp_model.load_state_dict(sharded_sd, assign=True, strict=False)
            logging.info(f"{LOG_PREFIX} State dict loaded")
            
            # Measure VRAM
            torch.cuda.synchronize(self.device)
            vram_bytes = torch.cuda.memory_allocated(self.device)
            vram_gb = vram_bytes / (1024 ** 3)
            
            logging.info(f"{LOG_PREFIX} VRAM usage: {vram_gb:.2f}GB")
            
            # Count sharded vs replicated params
            sharded_count = 0
            replicated_count = 0
            for name, param in fsdp_model.named_parameters():
                if hasattr(param, "device_mesh"):
                    sharded_count += 1
                else:
                    replicated_count += 1
            
            logging.info(f"{LOG_PREFIX} Params: {sharded_count} sharded, {replicated_count} replicated")
            
            # Store model in worker
            self.model = fsdp_model
            
            return {
                "status": "success",
                "vram_gb": vram_gb,
                "rank": self.rank,
                "sharded_count": sharded_count,
                "replicated_count": replicated_count,
            }
            
        except Exception as e:
            logging.error(f"{LOG_PREFIX} Error during FSDP2 initialization: {e}", exc_info=True)
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
            
            logging.info(f"{LOG_PREFIX} VRAM before cleanup: {vram_before:.2f}GB")
            
            # Delete sharded model
            if hasattr(self, 'model') and self.model is not None:
                logging.info(f"{LOG_PREFIX} Deleting sharded model...")
                del self.model
                self.model = None
            else:
                logging.info(f"{LOG_PREFIX} No model to cleanup")
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Clear CUDA cache
            torch.cuda.empty_cache()
            torch.cuda.synchronize(self.device)
            
            # Measure VRAM after cleanup
            vram_after = torch.cuda.memory_allocated(self.device) / (1024 ** 3)
            vram_freed = vram_before - vram_after
            
            logging.info(f"{LOG_PREFIX} VRAM after cleanup: {vram_after:.2f}GB (freed {vram_freed:.2f}GB)")
            
            return {
                "status": "success",
                "vram_freed_gb": vram_freed,
                "vram_after_gb": vram_after,
                "rank": self.rank,
            }
            
        except Exception as e:
            logging.error(f"{LOG_PREFIX} Error during cleanup: {e}", exc_info=True)
            return {"status": "error", "error": str(e), "rank": self.rank}
