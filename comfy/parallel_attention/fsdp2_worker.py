"""FSDP2Worker - Worker process holding sharded model.

Each worker process holds a portion of the model sharded via FSDP2.
Executes forward passes and other operations on the sharded model.
Uses DeviceMesh for process group topology (ARCHITECTURE.md requirement).

Based on Raylight worker pattern + FastVideo FSDP2 wrapping.
"""

import torch
import torch.distributed as dist
import logging

LOG_PREFIX = "⚡ [Parallel-Attention]"


class FSDP2Worker:
    """Worker that holds FSDP2 sharded model and executes operations.
    
    Based on Raylight worker pattern with FastVideo FSDP2 wrapping.
    Uses DeviceMesh for process group topology.
    """
    
    def __init__(self, rank: int, world_size: int):
        self.rank = rank
        self.world_size = world_size
        self.model = None
        self.device = torch.device(f"cuda:{rank}") if torch.cuda.is_available() else torch.device("cpu")
        self.device_mesh = None  # Will be set during initialize_fsdp2
        
        logging.info(f"{LOG_PREFIX} Worker-{rank} initialized on {self.device}")
    
    def execute(self, command: str, args: dict):
        """Execute command and return result."""
        if command == "echo":
            # Simple echo for Phase 2.5 test
            return args.get("message", "")
        elif command == "load_checkpoint":
            return self._load_checkpoint(args)
        elif command == "initialize_fsdp2_from_checkpoint":
            return self._initialize_fsdp2_from_checkpoint(args)
        elif command == "initialize_fsdp2_from_meta":
            return {"status": "error", "error": "Deprecated: Use initialize_fsdp2_from_checkpoint"}
        elif command == "initialize_fsdp2":
            return self._initialize_fsdp2(args)
        elif command == "forward":
            return self._forward(args)
        elif command == "get_model_size":
            return self._get_model_size(args)
        elif command == "check_model_state":
            return self._check_model_state(args)
        else:
            return {"success": False, "error": f"Unknown command: {command}"}
    
    def _initialize_fsdp2(self, args: dict):
        """Load checkpoint and apply FSDP2 sharding.
        
        Flow:
        1. Create DeviceMesh from provided info
        2. Rank 0 loads checkpoint to CPU
        3. Broadcast to all ranks
        4. Create model from state dict structure
        5. Apply FSDP2 wrapper with model-specific policy
        6. Load state dict (FSDP handles sharding)
        """
        import comfy.utils
        import comfy.sd
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import ShardingStrategy, CPUOffload
        
        model_structure = args.get("model_structure", {})
        checkpoint_path = args.get("checkpoint_path")
        device_mesh_info = args.get("device_mesh_info", None)
        
        logging.info(f"{LOG_PREFIX} Worker-{self.rank} initializing FSDP2...")
        logging.info(f"{LOG_PREFIX} Worker-{self.rank}   Model: {model_structure.get('model_class')}")
        logging.info(f"{LOG_PREFIX} Worker-{self.rank}   Checkpoint: {checkpoint_path}")
        
        # Step 1: Create DeviceMesh
        if device_mesh_info:
            try:
                from torch.distributed.device_mesh import init_device_mesh
                device_type = device_mesh_info.get("device_type", "cuda")
                mesh_shape = tuple(device_mesh_info.get("mesh_shape", [self.world_size]))
                mesh_dim_names = device_mesh_info.get("mesh_dim_names", ["dp"])
                self.device_mesh = init_device_mesh(device_type, mesh_shape, mesh_dim_names=mesh_dim_names)
                logging.info(f"{LOG_PREFIX} Worker-{self.rank} DeviceMesh: {device_type} {mesh_shape}")
            except Exception as e:
                logging.error(f"{LOG_PREFIX} Worker-{self.rank} DeviceMesh failed: {e}")
                return {"success": False, "error": f"DeviceMesh creation failed: {e}"}
        
        # Step 2: Load checkpoint (rank 0 loads, broadcasts)
        if checkpoint_path:
            try:
                if self.rank == 0:
                    logging.info(f"{LOG_PREFIX} Worker-0 loading checkpoint...")
                    state_dict = comfy.utils.load_torch_file(checkpoint_path)
                    logging.info(f"{LOG_PREFIX} Worker-0 loaded {len(state_dict)} keys")
                else:
                    state_dict = None
                
                # Broadcast state dict
                state_dict = self._broadcast_state_dict(state_dict)
                
                # Step 3: Create model from state dict structure
                logging.info(f"{LOG_PREFIX} Worker-{self.rank} creating model...")
                model_config = comfy.sd.model_config_from_unet(state_dict, "")
                
                with torch.device('meta'):
                    model = model_config.get_model(state_dict, "")
                
                logging.info(f"{LOG_PREFIX} Worker-{self.rank} model created on meta")
                
                # Step 4: Apply FSDP2 wrapper
                logging.info(f"{LOG_PREFIX} Worker-{self.rank} applying FSDP2...")
                
                # Get model-specific policy
                from comfy.parallel_attention.fsdp2_policies import get_fsdp2_strategy
                model_type = model_structure.get('model_class', '').lower()
                fsdp_strategy = get_fsdp2_strategy(model_type)
                
                if fsdp_strategy and self.device_mesh:
                    # Apply model-specific wrapping
                    fsdp_strategy(model, self.device_mesh)
                    logging.info(f"{LOG_PREFIX} Worker-{self.rank} applied {model_type} policy")
                else:
                    # Fallback: wrap entire model
                    model = FSDP(
                        model,
                        device_mesh=self.device_mesh,
                        sharding_strategy=ShardingStrategy.FULL_SHARD,
                        device_id=self.device,
                        cpu_offload=CPUOffload(offload_params=False),
                    )
                    logging.info(f"{LOG_PREFIX} Worker-{self.rank} applied default FSDP2")
                
                self.model = model
                
                # Step 5: Load state dict (FSDP handles sharding)
                logging.info(f"{LOG_PREFIX} Worker-{self.rank} loading sharded weights...")
                self.model.load_state_dict(state_dict, assign=True)
                
                # Step 6: Move to GPU
                self.model = self.model.to(self.device)
                
                vram_gb = torch.cuda.memory_allocated(self.device) / (1024**3) if torch.cuda.is_available() else 0
                logging.info(f"{LOG_PREFIX} Worker-{self.rank} complete: {vram_gb:.2f}GB VRAM")
                
                return {
                    "success": True,
                    "vram_gb": vram_gb,
                    "rank": self.rank,
                    "device_mesh": "initialized" if self.device_mesh else "none",
                }
                
            except Exception as e:
                logging.error(f"{LOG_PREFIX} Worker-{self.rank} loading failed: {e}")
                import traceback
                traceback.print_exc()
                return {"success": False, "error": str(e)}
        else:
            # No checkpoint path - Phase 2.3 behavior
            vram_gb = torch.cuda.memory_allocated(self.device) / (1024**3) if torch.cuda.is_available() else 0
            return {
                "success": True,
                "vram_gb": vram_gb,
                "rank": self.rank,
                "device_mesh": "initialized" if self.device_mesh else "none",
            }
    
    def _broadcast_state_dict(self, state_dict):
        """Broadcast state dict from rank 0 to all ranks."""
        object_list = [state_dict] if self.rank == 0 else [None]
        dist.broadcast_object_list(object_list, src=0)
        return object_list[0]
    
    def _forward(self, args: dict):
        """Execute forward pass on sharded model.
        
        Phase 2.3: Placeholder - returns None to test infrastructure works.
        Phase 2.4: Will implement actual forward pass relay.
        """
        forward_args = args.get("args", ())
        forward_kwargs = args.get("kwargs", {})
        
        logging.info(f"{LOG_PREFIX} Worker-{self.rank} forward pass (placeholder)")
        
        with torch.no_grad():
            # TODO Phase 2.4: Implement actual forward pass
            # Move inputs to device
            # output = self.model(*forward_args, **forward_kwargs)
            pass
        
        # Only rank 0 returns result
        if self.rank == 0:
            return {"success": True, "output": None}
        
        return {"success": True}
    
    def _get_model_size(self, args: dict):
        """Return VRAM usage of sharded model."""
        if self.model is None:
            size_bytes = 0
            vram_gb = 0.0
        else:
            if torch.cuda.is_available():
                size_bytes = torch.cuda.memory_allocated(self.device)
                vram_gb = size_bytes / (1024**3)
            else:
                size_bytes = 0
                vram_gb = 0.0
        
        return {
            "success": True,
            "size_bytes": size_bytes,
            "vram_gb": vram_gb
        }
    
    def _check_model_state(self, args: dict):
        """Check if model is loaded and count parameters."""
        has_model = self.model is not None
        param_count = 0
        
        if has_model:
            try:
                param_count = sum(p.numel() for p in self.model.parameters())
            except Exception as e:
                logging.error(f"{LOG_PREFIX} Worker-{self.rank} param count failed: {e}")
        
        return {
            "success": True,
            "has_model": has_model,
            "param_count": param_count
        }
    
    def _load_checkpoint(self, args: dict):
        """Load checkpoint state_dict in worker.
        
        Args:
            args: {"checkpoint_path": str}
        
        Returns:
            {"status": "success"|"error", "key_count": int, "total_size_gb": float}
        """
        import comfy.utils
        
        checkpoint_path = args.get("checkpoint_path")
        if not checkpoint_path:
            return {"status": "error", "error": "No checkpoint_path provided"}
        
        try:
            logging.info(f"{LOG_PREFIX} [Worker-{self.rank}] Loading checkpoint: {checkpoint_path}")
            
            # Load state dict (same as parent does)
            state_dict = comfy.utils.load_torch_file(checkpoint_path)
            
            # Calculate total size
            total_params = 0
            for v in state_dict.values():
                if hasattr(v, 'numel') and hasattr(v, 'element_size'):
                    total_params += v.numel() * v.element_size()
            total_size_gb = total_params / (1024**3)
            
            key_count = len(state_dict.keys())
            
            logging.info(f"{LOG_PREFIX} [Worker-{self.rank}] Checkpoint loaded: {key_count} keys, {total_size_gb:.2f}GB")
            
            # Store in instance for next phase (FSDP2 wrapping)
            self._worker_state_dict = state_dict
            
            return {
                "status": "success",
                "key_count": key_count,
                "total_size_gb": total_size_gb
            }
            
        except Exception as e:
            logging.error(f"{LOG_PREFIX} [Worker-{self.rank}] Checkpoint load failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                "status": "error",
                "error": f"Failed to load checkpoint: {e}"
            }
    
    def _initialize_fsdp2_from_checkpoint(self, args: dict):
        """Initialize FSDP2 from state_dict (Raylight pattern).
        
        Receives state_dict from parent, reconstructs model on meta, wraps with FSDP2.
        
        Args:
            args: {
                "state_dict": dict (extracted from parent model),
                "checkpoint_path": str (for reference),
                "model_type": str ("flux", "wan", "qwen_image")
            }
        
        Returns:
            {"status": "success", "vram_gb": float, "rank": int}
        """
        from comfy.parallel_attention.fsdp2_policies import FSDP2PolicyRegistry
        from comfy.parallel_attention.fsdp2_engine import apply_fsdp2_sharding
        import comfy.model_detection
        
        state_dict = args.get("state_dict")
        checkpoint_path = args.get("checkpoint_path")
        model_type = args.get("model_type")
        
        logging.info(f"{LOG_PREFIX} [Worker-{self.rank}] Initializing FSDP2 from state_dict...")
        logging.info(f"{LOG_PREFIX} [Worker-{self.rank}]   Model type: {model_type}")
        
        try:
            # DEBUG: Log what we received
            logging.info(f"{LOG_PREFIX} [Worker-{self.rank}]   Received state_dict with {len(state_dict)} keys")
            sample_keys = list(state_dict.keys())[:5]
            logging.info(f"{LOG_PREFIX} [Worker-{self.rank}]   Sample keys: {sample_keys}")
            
            # Ensure ComfyUI model registry is loaded
            import comfy.supported_models
            
            # Detect prefix and model config
            unet_prefix = comfy.model_detection.unet_prefix_from_state_dict(state_dict)
            logging.info(f"{LOG_PREFIX} [Worker-{self.rank}]   Detected prefix: '{unet_prefix}'")
            
            # Try detected prefix first
            model_config = comfy.model_detection.model_config_from_unet(state_dict, unet_prefix)
            
            # Fallback: Try common prefixes if detection failed
            if model_config is None:
                for fallback_prefix in ['diffusion_model.', '', 'model.diffusion_model.']:
                    logging.info(f"{LOG_PREFIX} [Worker-{self.rank}]   Trying fallback prefix: '{fallback_prefix}'")
                    model_config = comfy.model_detection.model_config_from_unet(state_dict, fallback_prefix)
                    if model_config is not None:
                        unet_prefix = fallback_prefix
                        break
            
            if model_config is None:
                raise RuntimeError(f"Failed to detect model config. Tried prefixes: {unet_prefix}, 'diffusion_model.', '', 'model.diffusion_model.'")
            
            logging.info(f"{LOG_PREFIX} [Worker-{self.rank}]   model_config: {model_config.__class__.__name__} (prefix: '{unet_prefix}')")
            
            # Create model on meta device
            with torch.device('meta'):
                meta_model = model_config.get_model(state_dict, unet_prefix)
            
            logging.info(f"{LOG_PREFIX} [Worker-{self.rank}]   Created meta model")
            
            # Get FSDP2 policy
            config = FSDP2PolicyRegistry.get_policy(model_type)
            
            # Apply FSDP2 sharding + load state_dict
            self.model = apply_fsdp2_sharding(meta_model, config, state_dict)
            
            # Step 6: Measure VRAM
            vram_gb = torch.cuda.memory_allocated(self.device) / (1024**3) if torch.cuda.is_available() else 0
            
            logging.info(f"{LOG_PREFIX} [Worker-{self.rank}] FSDP2 initialization complete: {vram_gb:.2f}GB VRAM")
            
            return {
                "status": "success",
                "vram_gb": vram_gb,
                "rank": self.rank
            }
            
        except Exception as e:
            logging.error(f"{LOG_PREFIX} [Worker-{self.rank}] FSDP2 init failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                "status": "error",
                "error": str(e)
            }
