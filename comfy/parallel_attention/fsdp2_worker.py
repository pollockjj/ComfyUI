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
        
        # Create DeviceMesh for FSDP2 sharding (ARCHITECTURE.md requirement)
        from torch.distributed.device_mesh import init_device_mesh
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        self.device_mesh = init_device_mesh(
            device_type,
            (world_size,),
            mesh_dim_names=["dp"]
        )
        
        logging.info(f"{LOG_PREFIX} Worker-{rank} initialized on {self.device}")
        logging.info(f"{LOG_PREFIX} Worker-{rank} DeviceMesh: {device_type} mesh_shape=({world_size},)")
        
        # GOLDEN DATA COLLECTION - DeviceMesh (EXHAUSTIVE)
        import json
        import os
        golden_mesh = {
            "source": "version2_worker_mesh",
            "rank": rank,
            "device_type": device_type,
            "mesh_shape": [world_size],
            "mesh_dim_names": ["dp"],
            "world_size": world_size,
            "device": str(self.device),
        }
        
        output_file = f"/home/johnj/parallel-attention/docs/reference/flux_golden_version2_worker{rank}_mesh.json"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(golden_mesh, f, indent=2)
        logging.info(f"⚡ [GOLDEN] Worker mesh data written to {output_file}")
    
    def execute(self, command: str, args: dict):
        """Execute command and return result."""
        if command == "echo":
            # Simple echo for Phase 2.5 test
            return args.get("message", "")
        elif command == "load_checkpoint":
            return self._load_checkpoint(args)
        elif command == "initialize_fsdp2_from_state_dict":
            return self._initialize_fsdp2_from_state_dict(args)
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
        elif command == "check_param_sharding":
            return self._check_param_sharding(args)
        elif command == "get_vram_breakdown":
            return self._get_vram_breakdown(args)
        elif command == "validate_sharding_strategy":
            return self._validate_sharding_strategy(args)
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
        """Execute forward pass on FSDP2-sharded model.
        
        Generic implementation - works for FSDP-only and future FSDP+USP.
        FSDP2 handles all-gather/reshard automatically via reshard_after_forward=True.
        
        Args:
            args: {
                "args": tuple - Positional arguments for model.forward()
                "kwargs": dict - Keyword arguments for model.forward()
            }
        
        Returns:
            {
                "status": "success"|"error",
                "output": tensor|dict (only from rank 0),
                "rank": int
            }
        
        Pattern: FastVideo gpu_worker.py line 75 + Raylight automatic communication
        """
        if self.model is None:
            return {"status": "error", "error": "Model not loaded"}
        
        forward_args = args.get("args", ())
        forward_kwargs = args.get("kwargs", {})
        
        logging.info(f"{LOG_PREFIX} [Worker-{self.rank}] Executing forward pass...")
        
        try:
            with torch.no_grad():
                # Move inputs to device
                forward_args = tuple(
                    arg.to(self.device) if isinstance(arg, torch.Tensor) else arg
                    for arg in forward_args
                )
                forward_kwargs = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in forward_kwargs.items()
                }
                
                # Execute forward pass on diffusion_model (ComfyUI structure)
                # FSDP2 automatically:
                # 1. All-gathers parameters (each rank has full params temporarily)
                # 2. Executes forward computation
                # 3. Reshards parameters (back to 11GB per GPU)
                if hasattr(self.model, 'diffusion_model'):
                    # ComfyUI models: wrapper.diffusion_model has forward()
                    output = self.model.diffusion_model(*forward_args, **forward_kwargs)
                else:
                    # Direct model (FastVideo pattern)
                    output = self.model(*forward_args, **forward_kwargs)
                
                logging.info(f"{LOG_PREFIX} [Worker-{self.rank}] Forward pass complete")
                
                # Move output to CPU for return (avoid GPU memory buildup)
                if isinstance(output, torch.Tensor):
                    output = output.cpu()
                elif isinstance(output, dict):
                    output = {
                        k: v.cpu() if isinstance(v, torch.Tensor) else v
                        for k, v in output.items()
                    }
                elif isinstance(output, (list, tuple)):
                    output = type(output)(
                        v.cpu() if isinstance(v, torch.Tensor) else v
                        for v in output
                    )
            
            # Only rank 0 returns output (data parallel - all ranks compute same thing)
            if self.rank == 0:
                return {
                    "status": "success",
                    "output": output,
                    "rank": self.rank
                }
            else:
                return {
                    "status": "success",
                    "rank": self.rank
                }
        
        except Exception as e:
            logging.error(f"{LOG_PREFIX} [Worker-{self.rank}] Forward pass failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                "status": "error",
                "error": str(e),
                "rank": self.rank
            }
    
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
    
    def _check_param_sharding(self, args: dict):
        """Check if a specific parameter is sharded or replicated.
        
        Args:
            args: {"param_name": str}
            
        Returns:
            {
                "status": "success",
                "rank": int,
                "param_name": str,
                "is_sharded": bool,
                "shape": tuple,
                "local_shape": tuple (if sharded),
                "size_mb": float
            }
        """
        param_name = args.get("param_name")
        
        if self.model is None:
            return {"status": "error", "error": "No model loaded"}
        
        # Find the parameter
        param = None
        for name, p in self.model.named_parameters():
            if name == param_name:
                param = p
                break
        
        if param is None:
            return {"status": "error", "error": f"Parameter {param_name} not found"}
        
        # Check if it's a DTensor (sharded)
        from torch.distributed.tensor import DTensor
        is_sharded = isinstance(param, DTensor)
        
        if is_sharded:
            # Get local shard info
            local_tensor = param.to_local()
            local_shape = tuple(local_tensor.shape)
            global_shape = tuple(param.shape)
            size_mb = local_tensor.numel() * local_tensor.element_size() / (1024**2)
            placements = str(param.placements)
        else:
            # Regular tensor (replicated)
            local_shape = tuple(param.shape)
            global_shape = local_shape
            size_mb = param.numel() * param.element_size() / (1024**2)
            placements = "None (replicated)"
        
        return {
            "status": "success",
            "rank": self.rank,
            "param_name": param_name,
            "is_sharded": is_sharded,
            "local_shape": local_shape,
            "global_shape": global_shape,
            "size_mb": size_mb,
            "placements": placements
        }
    
    def _get_vram_breakdown(self, args: dict):
        """Get detailed VRAM breakdown by parameter type.
        
        Returns:
            {
                "status": "success",
                "rank": int,
                "sharded_vram_gb": float,
                "replicated_vram_gb": float,
                "total_vram_gb": float,
                "sharded_count": int,
                "replicated_count": int
            }
        """
        if self.model is None:
            return {"status": "error", "error": "No model loaded"}
        
        from torch.distributed.tensor import DTensor
        
        sharded_vram = 0
        replicated_vram = 0
        sharded_count = 0
        replicated_count = 0
        
        for name, param in self.model.named_parameters():
            if isinstance(param, DTensor):
                # Local shard size
                local_tensor = param.to_local()
                sharded_vram += local_tensor.numel() * local_tensor.element_size()
                sharded_count += 1
            else:
                # Full tensor size (replicated)
                replicated_vram += param.numel() * param.element_size()
                replicated_count += 1
        
        total_vram = torch.cuda.memory_allocated(self.device) if torch.cuda.is_available() else 0
        
        return {
            "status": "success",
            "rank": self.rank,
            "sharded_vram_gb": sharded_vram / (1024**3),
            "replicated_vram_gb": replicated_vram / (1024**3),
            "total_vram_gb": total_vram / (1024**3),
            "sharded_count": sharded_count,
            "replicated_count": replicated_count
        }
    
    def _validate_sharding_strategy(self, args: dict):
        """Validate that sharding strategy was applied correctly.
        
        Returns:
            {
                "status": "success",
                "rank": int,
                "total_fsdp_modules": int,
                "double_blocks_wrapped": int,
                "single_blocks_wrapped": int
            }
        """
        if self.model is None:
            return {"status": "error", "error": "No model loaded"}
        
        from torch.distributed.fsdp import FSDPModule
        
        # Count FSDP-wrapped modules
        total_fsdp = 0
        for name, module in self.model.named_modules():
            if isinstance(module, FSDPModule):
                total_fsdp += 1
        
        # Count wrapped blocks
        double_wrapped = 0
        single_wrapped = 0
        
        if hasattr(self.model, 'diffusion_model'):
            dm = self.model.diffusion_model
            
            if hasattr(dm, 'double_blocks'):
                for block in dm.double_blocks:
                    if isinstance(block, FSDPModule):
                        double_wrapped += 1
            
            if hasattr(dm, 'single_blocks'):
                for block in dm.single_blocks:
                    if isinstance(block, FSDPModule):
                        single_wrapped += 1
        
        return {
            "status": "success",
            "rank": self.rank,
            "total_fsdp_modules": total_fsdp,
            "double_blocks_wrapped": double_wrapped,
            "single_blocks_wrapped": single_wrapped
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
        """Initialize FSDP2 using FastVideo iterator pattern.
        
        Streams tensors one at a time from safetensors file.
        Never loads full 22GB state_dict into memory.
        
        Args:
            args: {
                "checkpoint_path": str (path to .safetensors file),
                "model_type": str ("flux", "wan", "qwen_image")
            }
        
        Returns:
            {"status": "success", "vram_gb": float, "rank": int}
        """
        from safetensors.torch import safe_open
        from torch.distributed.tensor import distribute_tensor
        from comfy.parallel_attention.fsdp2_policies import FSDP2PolicyRegistry
        from comfy.parallel_attention.fsdp2_engine import apply_fsdp2_sharding_structure_only
        import comfy.model_detection
        import comfy.supported_models
        
        checkpoint_path = args.get("checkpoint_path")
        model_type = args.get("model_type")
        
        if not checkpoint_path:
            return {"status": "error", "error": "No checkpoint_path provided"}
        
        logging.info(f"{LOG_PREFIX} [Worker-{self.rank}] Initializing FSDP2 (FastVideo iterator pattern)...")
        logging.info(f"{LOG_PREFIX} [Worker-{self.rank}]   Model type: {model_type}")
        logging.info(f"{LOG_PREFIX} [Worker-{self.rank}]   Checkpoint: {checkpoint_path}")
        
        try:
            # Step 1: Receive meta_model from parent (EXTEND ComfyUI)
            # Parent already created meta model with correct config
            meta_model = args.get("meta_model")
            if meta_model is None:
                raise ValueError("No meta_model provided from parent")
            
            logging.info(f"{LOG_PREFIX} [Worker-{self.rank}] Received meta_model from parent: {type(meta_model).__name__}")
            logging.info(f"{LOG_PREFIX} [Worker-{self.rank}] Meta model on device: {next(meta_model.parameters()).device}")
            
            # GOLDEN DATA COLLECTION - Worker received meta (EXHAUSTIVE)
            import json
            import os
            
            all_param_names = [name for name, _ in meta_model.named_parameters()]
            meta_sd = meta_model.state_dict()
            
            golden_meta = {
                "source": "version2_worker_meta",
                "rank": self.rank,
                "meta_model_class": type(meta_model).__name__,
                "all_param_names": all_param_names,
                "param_shapes": {k: list(v.shape) for k, v in meta_sd.items()},
                "meta_device": str(next(meta_model.parameters()).device),
            }
            
            output_file = f"/home/johnj/parallel-attention/docs/reference/flux_golden_version2_worker{self.rank}_meta.json"
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(golden_meta, f, indent=2)
            logging.info(f"⚡ [GOLDEN] Worker meta data written to {output_file}")
            
            # Step 3: Apply FSDP2 wrapping (structure only, no weights yet)
            config = FSDP2PolicyRegistry.get_policy(model_type)
            self.model = apply_fsdp2_sharding_structure_only(
                meta_model, 
                config,
                self.device_mesh
            )
            
            logging.info(f"{LOG_PREFIX} [Worker-{self.rank}] FSDP2 wrapping applied with DeviceMesh")
            
            # GOLDEN DATA COLLECTION - After FSDP wrapping (EXHAUSTIVE)
            wrapped_params = []
            for n, p in self.model.named_parameters():
                wrapped_params.append({
                    "name": n,
                    "type": type(p).__name__,
                    "has_device_mesh": hasattr(p, 'device_mesh'),
                    "shape": list(p.shape) if hasattr(p, 'shape') else None,
                    "device": str(p.device) if hasattr(p, 'device') else None,
                })
            
            golden_wrapped = {
                "source": "version2_worker_wrapped",
                "rank": self.rank,
                "wrapped_params": wrapped_params,
                "dtensor_count": sum(1 for p in wrapped_params if p['has_device_mesh']),
                "regular_count": sum(1 for p in wrapped_params if not p['has_device_mesh']),
            }
            
            output_file = f"/home/johnj/parallel-attention/docs/reference/flux_golden_version2_worker{self.rank}_wrapped.json"
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(golden_wrapped, f, indent=2)
            logging.info(f"⚡ [GOLDEN] Worker wrapped data written to {output_file}")
            
            # Step 4: Load weights using iterator (FastVideo pattern)
            meta_sd = self.model.state_dict()
            sharded_sd = {}
            
            logging.info(f"{LOG_PREFIX} [Worker-{self.rank}] Streaming weights from file...")
            
            with safe_open(checkpoint_path, framework="pt", device="cpu") as f:
                tensor_count = 0
                for param_name in f.keys():
                    # Load ONE tensor at a time
                    full_tensor = f.get_tensor(param_name)
                    
                    # Get corresponding meta param
                    meta_param = meta_sd.get(param_name)
                    if meta_param is None:
                        # Try with diffusion_model. prefix
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
            
            logging.info(f"{LOG_PREFIX} [Worker-{self.rank}] Loaded {tensor_count} tensors via iterator")
            
            # Step 5: Load sharded dict into model
            self.model.load_state_dict(sharded_sd, assign=True, strict=False)
            
            # Measure VRAM
            vram_gb = torch.cuda.memory_allocated(self.device) / (1024**3) if torch.cuda.is_available() else 0
            
            logging.info(f"{LOG_PREFIX} [Worker-{self.rank}] FSDP2 initialization complete: {vram_gb:.2f}GB VRAM")
            
            # GOLDEN DATA COLLECTION - After weight loading (EXHAUSTIVE)
            from torch.distributed.tensor import DTensor
            
            all_params = []
            for name, param in self.model.named_parameters():
                param_data = {
                    "name": name,
                    "shape": list(param.shape),
                    "dtype": str(param.dtype),
                    "device": str(param.device),
                }
                if isinstance(param, DTensor):
                    param_data["is_dtensor"] = True
                    param_data["local_shape"] = list(param.to_local().shape)
                    param_data["placements"] = str(param.placements)
                else:
                    param_data["is_dtensor"] = False
                all_params.append(param_data)
            
            golden_loaded = {
                "source": "version2_worker_loaded",
                "rank": self.rank,
                "vram_gb": vram_gb,
                "tensor_count": tensor_count,
                "all_params": all_params,
            }
            
            output_file = f"/home/johnj/parallel-attention/docs/reference/flux_golden_version2_worker{self.rank}_loaded.json"
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(golden_loaded, f, indent=2)
            logging.info(f"⚡ [GOLDEN] Worker loaded data written to {output_file}")
            
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
    
    def _initialize_fsdp2_from_state_dict(self, args: dict):
        """DEPRECATED: Causes 'too many open files' error.
        
        Use _initialize_fsdp2_from_checkpoint with iterator pattern instead.
        """
        logging.warning(f"{LOG_PREFIX} [Worker-{self.rank}] initialize_fsdp2_from_state_dict is deprecated")
        return {"status": "error", "error": "Deprecated: Use initialize_fsdp2_from_checkpoint"}
        
        try:
            # Ensure ComfyUI model registry is loaded
            import comfy.supported_models
            
            # Detect model config from state_dict
            unet_prefix = comfy.model_detection.unet_prefix_from_state_dict(state_dict)
            logging.info(f"{LOG_PREFIX} [Worker-{self.rank}]   Detected prefix: '{unet_prefix}'")
            
            model_config = comfy.model_detection.model_config_from_unet(state_dict, unet_prefix)
            
            if model_config is None:
                for fallback_prefix in ['diffusion_model.', '', 'model.diffusion_model.']:
                    model_config = comfy.model_detection.model_config_from_unet(state_dict, fallback_prefix)
                    if model_config is not None:
                        unet_prefix = fallback_prefix
                        break
            
            if model_config is None:
                raise RuntimeError(f"Failed to detect model config")
            
            logging.info(f"{LOG_PREFIX} [Worker-{self.rank}]   model_config: {model_config.__class__.__name__}")
            
            # Create meta model (structure only, no weights)
            with torch.device('meta'):
                meta_model = model_config.get_model(state_dict, unet_prefix)
            
            logging.info(f"{LOG_PREFIX} [Worker-{self.rank}]   Created meta model")
            
            # Get FSDP2 policy
            config = FSDP2PolicyRegistry.get_policy(model_type)
            
            # Apply FSDP2 sharding + load state_dict
            self.model = apply_fsdp2_sharding(meta_model, config, state_dict)
            
            # Measure VRAM
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
