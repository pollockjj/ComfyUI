"""FSDP2Worker - Distributed inference worker with FSDP2 sharding.

Implements multi-GPU model sharding using PyTorch FSDP2 and DeviceMesh.
Based on FastVideo worker pattern.
"""

import os
import torch
import logging
import types
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
    
    def __init__(self, rank: int, world_size: int, backend: str = "nccl"):
        """Initialize worker process.
        
        Args:
            rank: Worker rank (0 to world_size-1)
            world_size: Total number of workers
            backend: Communication backend (nccl or gloo)
        """
        self.rank = rank
        self.world_size = world_size
        self.backend = backend
        self.device = torch.device(f"cuda:{rank}") if torch.cuda.is_available() else torch.device("cpu")
        
        # CRITICAL: Set CUDA device for this worker process
        if torch.cuda.is_available():
            torch.cuda.set_device(self.device)
        
        self.usp_config = None
        self.usp_enabled = False
        self._usp_parallel_initialized = False

        self.use_backend_plugin_system = (
            os.getenv("USE_BACKEND_PLUGIN_SYSTEM", "false").lower() == "true"
        )
        if self.use_backend_plugin_system:
            log_rank0(
                rank,
                "info",
                f"{LOG_PREFIX} Worker-{rank} Backend Plugin System feature flag enabled",
            )
        else:
            log_rank0(
                rank,
                "debug",
                f"{LOG_PREFIX} Worker-{rank} Backend Plugin System feature flag disabled",
            )

        # Create DeviceMesh (ARCHITECTURE.md requirement)
        from torch.distributed.device_mesh import init_device_mesh
        # Use "cpu" device type for gloo backend, "cuda" for nccl
        device_type = "cpu" if backend == "gloo" else "cuda"
        self.device_mesh = init_device_mesh(
            device_type,
            (world_size,),
            mesh_dim_names=["dp"]
        )
        
        log_rank0(rank, 'info', f"{LOG_PREFIX} Worker-{rank} initialized on {self.device}")
        log_rank0(rank, 'info', f"{LOG_PREFIX} Worker-{rank} DeviceMesh: {device_type} mesh_shape=({world_size},)")
        
        # LOG ACTUAL BACKEND BEING USED
        import torch.distributed as dist
        if dist.is_initialized():
            actual_backend = dist.get_backend()
            log_rank0(rank, 'info', f"{LOG_PREFIX} Worker-{rank} ACTUAL BACKEND: {actual_backend}")
        
        # Worker initialization complete; USP-specific hooks removed in purge
    
    # Legacy USP helpers removed during purge
    
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
        
        elif command == "apply_model_step":
            return self._apply_model_step(args)
        
        elif command == "common_ksampler":
            return self._common_ksampler(args)
        
        elif command == "apply_model":
            return self._apply_model(args)
        
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
    
    def _sample(self, args: dict):
        """Execute full sampling session for distributed inference.
        
        Workers call comfy.sample.sample() with FSDP2-sharded ModelPatcher.
        FSDP2 handles all-gather/reshard transparently during forward passes.
        
        Args:
            args: Full sampling config (tensors on CPU for serialization):
                - noise: Noise tensor
                - steps: Number of steps
                - cfg: CFG scale
                - sampler_name: Sampler type
                - scheduler: Scheduler type
                - positive: Positive conditioning
                - negative: Negative conditioning
                - latent_image: Initial latent
                - denoise: Denoise strength
                - seed: Random seed
        
        Returns:
            dict: {"samples": tensor} on rank 0, {"status": "ok"} on others
        """
        import comfy.sample
        import comfy.utils
        
        LOG_PREFIX = f"⚡ [Worker][Rank {self.rank}][Sample]"
        
        # Log model state BEFORE sampling (M3 instrumentation)
        logging.info(f"{LOG_PREFIX} Model state check:")
        logging.info(f"{LOG_PREFIX}   self.model type: {type(self.model).__name__}")
        logging.info(f"{LOG_PREFIX}   self.model.model type: {type(self.model.model).__name__}")
        logging.info(f"{LOG_PREFIX}   self.model.load_device: {self.model.load_device}")
        
        # Check if FSDP2 wrapped
        from torch.distributed._composable.fsdp import FSDPModule
        is_fsdp = isinstance(self.model.model.diffusion_model, FSDPModule)
        logging.info(f"{LOG_PREFIX}   FSDP2 wrapped: {is_fsdp}")
        
        # Log VRAM BEFORE sampling
        vram_before = 0.0
        if torch.cuda.is_available():
            vram_before = torch.cuda.memory_allocated(self.device) / 1024**3
            logging.info(f"{LOG_PREFIX}   VRAM before sampling: {vram_before:.3f}GB")
        
        # Move inputs to worker device
        noise = args["noise"].to(self.device)
        latent_image = args["latent_image"].to(self.device)
        positive = self._move_conditioning_to_device(args["positive"])
        negative = self._move_conditioning_to_device(args["negative"])
        
        # Model is ALREADY loaded and sharded - DON'T call model.load()!
        # That would load the full model to parent process
        
        # Disable progress bar on non-rank-0 workers
        disable_pbar = True
        if self.rank == 0:
            disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
        
        logging.info(f"{LOG_PREFIX} Starting sampling: {args['steps']} steps, seed={args['seed']}")
        logging.info(f"{LOG_PREFIX} Model device: {self.model.model.device if hasattr(self.model.model, 'device') else 'unknown'}")
        
        # Execute standard ComfyUI sampling with FSDP2-sharded model
        # FSDP2 handles all-gather/reshard automatically during forward
        with torch.no_grad():
            samples = comfy.sample.sample(
                model=self.model,
                noise=noise,
                steps=args["steps"],
                cfg=args["cfg"],
                sampler_name=args["sampler_name"],
                scheduler=args["scheduler"],
                positive=positive,
                negative=negative,
                latent_image=latent_image,
                denoise=args["denoise"],
                disable_pbar=disable_pbar,
                seed=args["seed"]
            )
        
        # Log VRAM AFTER sampling (M3 instrumentation)
        vram_after = 0.0
        if torch.cuda.is_available():
            vram_after = torch.cuda.memory_allocated(self.device) / 1024**3
            vram_delta = vram_after - vram_before
            logging.info(f"{LOG_PREFIX}   VRAM after sampling: {vram_after:.3f}GB")
            logging.info(f"{LOG_PREFIX}   VRAM delta: {vram_delta:+.3f}GB")
        
        logging.info(f"{LOG_PREFIX} Sampling complete, output shape={samples.shape}")
        
        # Rank 0 returns samples, others return status
        if self.rank == 0:
            return {"samples": samples.cpu()}
        else:
            return {"status": "ok"}
    
    def _move_conditioning_to_device(self, cond):
        """Move conditioning to worker device."""
        if isinstance(cond, list):
            return [[self._move_tensor_to_device(c[0]), c[1]] for c in cond]
        return cond
    
    def _move_tensor_to_device(self, tensor):
        """Recursively move tensors to worker device."""
        if isinstance(tensor, torch.Tensor):
            return tensor.to(self.device)
        elif isinstance(tensor, dict):
            return {k: self._move_tensor_to_device(v) for k, v in tensor.items()}
        elif isinstance(tensor, (list, tuple)):
            return type(tensor)(self._move_tensor_to_device(v) for v in tensor)
        return tensor
    
    def _move_tensors_to_device(self, obj, device):
        """Recursively move all tensors in nested structure to device.
        
        Legacy method for compatibility.
        """
        if isinstance(obj, torch.Tensor):
            return obj.to(device)
        elif isinstance(obj, dict):
            return {k: self._move_tensors_to_device(v, device) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return type(obj)(self._move_tensors_to_device(item, device) for item in obj)
        else:
            return obj
    
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
        checkpoint_path = args["checkpoint_path"]
        model_type = args["model_type"]
        
        LOG_PREFIX = f"⚡ [Parallel-Attention][Worker][Rank {self.rank}]"
        
        # Override worker BPS flag if explicitly requested via args
        requested_bps = bool(args.get("use_backend_plugin_system", False))
        if requested_bps:
            self.use_backend_plugin_system = True
            log_rank0(
                self.rank,
                "info",
                f"{LOG_PREFIX} Backend Plugin System enabled via workflow request",
            )
        elif self.use_backend_plugin_system and not requested_bps:
            log_rank0(
                self.rank,
                "debug",
                f"{LOG_PREFIX} Worker feature flag enabled; upstream request did not opt-in",
            )
        
        # Get policy from registry (don't pass through pipe - dataclass serialization issues)
        from comfy.parallel_attention.fsdp2_policies import FSDP2PolicyRegistry
        policy = FSDP2PolicyRegistry.get_policy(model_type)
        
        log_rank0(self.rank, 'info', f"⚡ [Parallel-Attention][Worker] USE checkpoint_path: {checkpoint_path}")
        log_rank0(self.rank, 'debug', f"{LOG_PREFIX} Policy: {policy.model_name}")
        
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
            
            # Materialize model_sampling to device (NOT part of diffusion_model)
            # model_sampling is created fresh (not from checkpoint) so we recreate it on device
            import comfy.model_sampling
            model_type = model.model_type
            model.model_sampling = comfy.model_base.model_sampling(model.model_config, model_type)
            log_rank0(self.rank, 'debug', f"{LOG_PREFIX} Recreated model_sampling on {self.device}")
            
            # Apply FSDP2 sharding
            from comfy.parallel_attention.fsdp2_engine import apply_fsdp2_sharding_structure_only
            
            log_rank0(self.rank, 'debug', f"{LOG_PREFIX} Applying FSDP2 sharding...")
            
            fsdp_model = apply_fsdp2_sharding_structure_only(
                model,
                policy,
                self.device_mesh
            )
            
            # Load weights using iterator (WORKING pattern)
            log_rank0(self.rank, 'info', f"{LOG_PREFIX} Loading weights...")
            
            # Track VRAM before loading
            if torch.cuda.is_available():
                vram_before_load = torch.cuda.memory_allocated(self.device) / 1024**3
                log_rank0(self.rank, 'info', f"{LOG_PREFIX} VRAM before weight loading: {vram_before_load:.2f}GB")
            
            from safetensors.torch import safe_open
            from torch.distributed._tensor import distribute_tensor
            
            # Get meta state dict for key lookup
            meta_sd = fsdp_model.state_dict()
            sharded_sd = {}
            
            # Get reference dtype from first sharded param for casting ALL tensors
            ref_dtype = torch.bfloat16  # Default for Flux
            for name, param in fsdp_model.named_parameters():
                if hasattr(param, "device_mesh") and param.dtype != torch.float32:
                    ref_dtype = param.dtype
                    break
            
            log_rank0(self.rank, 'debug', f"{LOG_PREFIX} Reference dtype: {ref_dtype}")
            log_rank0(self.rank, 'debug', f"{LOG_PREFIX} Streaming from safetensors...")
            
            with safe_open(checkpoint_path, framework="pt", device="cpu") as f:
                tensor_count = 0
                for param_name in f.keys():
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
                    
                    # Move to device first (cheap copy), then cast to avoid temp allocation
                    full_tensor = full_tensor.to(device=self.device)
                    if full_tensor.dtype != ref_dtype:
                        full_tensor = full_tensor.to(dtype=ref_dtype)
                    
                    # Distribute if FSDP-wrapped (has device_mesh)
                    if hasattr(meta_param, "device_mesh"):
                        sharded_tensor = distribute_tensor(
                            full_tensor,
                            meta_param.device_mesh,
                            meta_param.placements,
                        )
                    else:
                        # Replicated param - keep on device with correct dtype
                        sharded_tensor = full_tensor
                    
                    sharded_sd[param_name] = torch.nn.Parameter(sharded_tensor)
                    tensor_count += 1
            
            log_rank0(self.rank, 'debug', f"{LOG_PREFIX} Loaded {tensor_count} tensors")
            
            # Load sharded dict into model
            log_rank0(self.rank, 'debug', f"{LOG_PREFIX} Applying weights to model...")
            fsdp_model.load_state_dict(sharded_sd, assign=True, strict=False)
            
            # Install attention forwards RIGHT AFTER weight loading
            usp_config = args.get("usp_config")
            attention_impl = args.get("attention_impl", "xfuser")  # Default to xFuser for backward compat
            
            log_rank0(self.rank, 'info', f"{LOG_PREFIX} 🔥🔥🔥 PATH SELECTION: usp_config={usp_config is not None}, use_backend_plugin_system={self.use_backend_plugin_system}, attention_impl={attention_impl}")
            
            if usp_config:
                if self.use_backend_plugin_system:
                    # Use Backend Plugin System for attention
                    log_rank0(self.rank, 'info', f"{LOG_PREFIX} 🔥🔥🔥 TAKING BPS PATH")
                    ulysses_degree = int(usp_config.get("ulysses_degree", 1))
                    ring_degree = int(usp_config.get("ring_degree", 1))
                    attention_backend = usp_config.get("attention_backend", "FLASH_ATTN")
                    
                    # Map impl to backend name for BPS
                    if attention_impl == "torch_sp_ulysses":
                        bps_backend_name = "TORCH_SP_ULYSSES"
                        log_rank0(self.rank, 'info', f"{LOG_PREFIX} 🔥🔥🔥 BPS BACKEND: TORCH_SP_ULYSSES")
                    else:
                        bps_backend_name = "XFUSER_USP"
                        log_rank0(self.rank, 'info', f"{LOG_PREFIX} 🔥🔥🔥 BPS BACKEND: XFUSER_USP")
                    
                    self._patch_distributed_attention_bps(
                        fsdp_model, ulysses_degree, ring_degree, attention_backend, bps_backend_name
                    )
                elif attention_impl == "fastvideo":
                    # Use FastVideo-style pure Ulysses implementation
                    log_rank0(self.rank, 'info', f"{LOG_PREFIX} 🔥🔥🔥 TAKING FASTVIDEO PATH")
                    ulysses_degree = int(usp_config.get("ulysses_degree", 2))
                    attention_backend = usp_config.get("attention_backend", "sdpa")
                    self._apply_fastvideo_ulysses_patches(fsdp_model, ulysses_degree, attention_backend)
                else:
                    # Use xFuser USP implementation (default)
                    log_rank0(self.rank, 'info', f"{LOG_PREFIX} 🔥🔥🔥 TAKING XFUSER USP PATH")
                    self._patch_usp_forwards_after_load(fsdp_model, usp_config)
            
            # CRITICAL: Enable comfy_cast_weights on ALL modules (multi-GPU pattern)
            # This ensures weights auto-cast to input device during forward (handles FSDP2 cross-device)
            for module in fsdp_model.diffusion_model.modules():
                if hasattr(module, "comfy_cast_weights"):
                    module.comfy_cast_weights = True
            
            log_rank0(self.rank, 'debug', f"{LOG_PREFIX} Enabled comfy_cast_weights on all modules")
            
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
            
            # Set manual_cast_dtype to match loaded weights (critical for input casting)
            # Without this, inputs stay Float32 while weights are BFloat16
            fsdp_model.manual_cast_dtype = ref_dtype
            log_rank0(self.rank, 'debug', f"{LOG_PREFIX} Set manual_cast_dtype={ref_dtype}")
            
            # Wrap in ModelPatcher so workers retain standard ComfyUI APIs
            from comfy.model_patcher import ModelPatcher
            import comfy.model_management
            
            model_patcher = ModelPatcher(
                fsdp_model,
                load_device=self.device,
                offload_device=comfy.model_management.unet_offload_device()
            )
            
            # Store ModelPatcher, not raw model
            self.model = model_patcher
            
            # Legacy ring-attention configuration removed during purge            # Apply object patches from parent (if any)
            # This is the clean extension point for distributed model modifications
            object_patches = args.get("object_patches", {})
            if object_patches:
                self._apply_object_patches_to_worker_model(object_patches)
            
            return {
                "status": "success",
                "vram_gb": vram_after_load,
                "rank": self.rank,
                "sharded_count": sharded_count,
                "replicated_count": replicated_count,
            }
            
        except Exception as e:
            logging.error(f"{LOG_PREFIX} Error: {e}", exc_info=True)
            return {"status": "error", "error": str(e)}

    def _patch_distributed_attention_bps(self, fsdp_model, ulysses_degree, ring_degree, attention_backend, bps_backend_name="XFUSER_USP"):
        """Patch model forwards to use Backend Plugin System DistributedAttention.

        Called when USE_BACKEND_PLUGIN_SYSTEM=true. Initializes xFuser parallel state
        (required for process groups), then monkey-patches forwards to delegate to
        the unified DistributedAttention layer.
        """
        LOG_PREFIX = f"⚡ [Parallel-Attention][Worker][Rank {self.rank}][BPS]"

        if ulysses_degree <= 1 and ring_degree <= 1:
            log_rank0(self.rank, "debug", f"{LOG_PREFIX} Skipping BPS patching (no parallelism)")
            return

        sequence_degree = ulysses_degree * ring_degree
        if sequence_degree != self.world_size:
            raise RuntimeError(
                f"{LOG_PREFIX} USP configuration mismatch: ulysses_degree={ulysses_degree}, "
                f"ring_degree={ring_degree}, world_size={self.world_size}"
            )

        # Initialize parallel state based on backend
        if bps_backend_name == "TORCH_SP_ULYSSES":
            # Use TorchSP communicator (expects dist already initialized by FSDP2)
            from .torch_sp_ulysses import initialize_sp_group, get_sp_rank, get_sp_world_size, get_sp_group
            if not self._usp_parallel_initialized:
                initialize_sp_group(sp_degree=self.world_size)
                self._usp_parallel_initialized = True
                log_rank0(self.rank, "info", f"{LOG_PREFIX} Initialized TorchSP communicator")
            
            # Store backend-agnostic accessors
            self._sp_get_rank = get_sp_rank
            self._sp_get_world_size = get_sp_world_size
            self._sp_get_group = get_sp_group
        else:
            # Initialize xFuser parallel state (for xFuser backend)
            try:
                from xfuser.core.distributed import (
                    init_distributed_environment,
                    initialize_model_parallel,
                    get_sequence_parallel_rank,
                    get_sequence_parallel_world_size,
                    get_sp_group,
                )
            except ImportError as exc:
                raise RuntimeError("xfuser package is required for process group management") from exc

            if not self._usp_parallel_initialized:
                init_distributed_environment(rank=self.rank, world_size=self.world_size)
                initialize_model_parallel(
                    sequence_parallel_degree=self.world_size,
                    ring_degree=ring_degree,
                    ulysses_degree=ulysses_degree,
                )
                self._usp_parallel_initialized = True
                log_rank0(self.rank, "info", f"{LOG_PREFIX} Initialized xFuser parallel state")
            
            # Store backend-agnostic accessors (xFuser naming)
            self._sp_get_rank = get_sequence_parallel_rank
            self._sp_get_world_size = get_sequence_parallel_world_size
            self._sp_get_group = get_sp_group

        # Import Backend Plugin System components
        from .distributed_attention import DistributedAttention
        from .usp_single_forward import apply_mod

        # Determine backend kwargs based on attention type
        backend_kwargs = {
            "ulysses_degree": ulysses_degree,
            "ring_degree": ring_degree,
            "attn_type": attention_backend,
        }

        # Create DistributedAttention instances for this model
        # We need to inspect the model to get head count and head_dim
        diffusion_model = fsdp_model.diffusion_model
        first_double = getattr(diffusion_model, "double_blocks", [None])[0]
        if first_double is None:
            log_rank0(self.rank, "warning", f"{LOG_PREFIX} No double_blocks found, skipping")
            return

        num_heads = first_double.num_heads
        head_dim = first_double.img_attn.qkv.out_features // (3 * num_heads)

        log_rank0(
            self.rank,
            "info",
            f"{LOG_PREFIX} Creating DistributedAttention (heads={num_heads}, dim={head_dim}, backend={bps_backend_name})",
        )

        # Patch double blocks
        import types
        double_blocks = getattr(diffusion_model, "double_blocks", [])
        for block in double_blocks:
            # Store the DistributedAttention instance on the block
            block._distributed_attn = DistributedAttention(
                num_heads=num_heads,
                head_dim=head_dim,
                device=self.device,
                dtype=torch.bfloat16,
                backend_name=bps_backend_name,
                **backend_kwargs,
            )

            def bps_double_forward(
                self,
                *,
                img,
                txt,
                vec,
                pe,
                attn_mask=None,
                modulation_dims_img=None,
                modulation_dims_txt=None,
                **kwargs,
            ):
                """Backend Plugin System double-stream forward."""
                print(f"🔥🔥🔥 BPS DOUBLE FORWARD CALLED: img.shape={img.shape}, txt.shape={txt.shape}, vec={'None' if vec is None else vec.shape}")
                print(f"🔥🔥🔥 self type: {type(self).__name__}, has img_mod: {hasattr(self, 'img_mod')}, has _distributed_attn: {hasattr(self, '_distributed_attn')}")
                
                img_mod1, img_mod2 = self.img_mod(vec)
                txt_mod1, txt_mod2 = self.txt_mod(vec)

                img_modulated = apply_mod(
                    self.img_norm1(img), (1 + img_mod1.scale), img_mod1.shift, modulation_dims_img
                )
                txt_modulated = apply_mod(
                    self.txt_norm1(txt), (1 + txt_mod1.scale), txt_mod1.shift, modulation_dims_txt
                )

                img_qkv = self.img_attn.qkv(img_modulated)
                txt_qkv = self.txt_attn.qkv(txt_modulated)

                img_q, img_k, img_v = img_qkv.view(
                    img_qkv.shape[0], img_qkv.shape[1], 3, self.num_heads, -1
                ).permute(2, 0, 3, 1, 4)
                txt_q, txt_k, txt_v = txt_qkv.view(
                    txt_qkv.shape[0], txt_qkv.shape[1], 3, self.num_heads, -1
                ).permute(2, 0, 3, 1, 4)

                img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)
                txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

                # FastVideo pattern: img is sharded via Ulysses, txt is replicated
                # Transpose to FastVideo convention: [batch, heads, seq, dim] → [batch, seq, heads, dim]
                img_q = img_q.transpose(1, 2)
                img_k = img_k.transpose(1, 2)
                img_v = img_v.transpose(1, 2)
                txt_q = txt_q.transpose(1, 2)
                txt_k = txt_k.transpose(1, 2)
                txt_v = txt_v.transpose(1, 2)
                
                # Pass img as main q/k/v, txt as replicated_q/k/v
                img_attn, txt_attn = self._distributed_attn.forward(
                    img_q, img_k, img_v,
                    replicated_q=txt_q,
                    replicated_k=txt_k,
                    replicated_v=txt_v,
                    freqs_cis=pe,
                    attn_mask=attn_mask,
                )

                # Transpose back: [batch, seq, heads, dim] → [batch, heads, seq, dim]
                img_attn = img_attn.transpose(1, 2).contiguous()
                txt_attn = txt_attn.transpose(1, 2).contiguous() if txt_attn is not None else txt_attn

                img_updated = img + apply_mod(
                    self.img_attn.proj(img_attn), img_mod1.gate, None, modulation_dims_img
                )
                img_updated = img_updated + apply_mod(
                    self.img_mlp(
                        apply_mod(
                            self.img_norm2(img_updated),
                            (1 + img_mod2.scale),
                            img_mod2.shift,
                            modulation_dims_img,
                        )
                    ),
                    img_mod2.gate,
                    None,
                    modulation_dims_img,
                )

                txt_updated = txt + apply_mod(
                    self.txt_attn.proj(txt_attn), txt_mod1.gate, None, modulation_dims_txt
                )
                txt_updated = txt_updated + apply_mod(
                    self.txt_mlp(
                        apply_mod(
                            self.txt_norm2(txt_updated),
                            (1 + txt_mod2.scale),
                            txt_mod2.shift,
                            modulation_dims_txt,
                        )
                    ),
                    txt_mod2.gate,
                    None,
                    modulation_dims_txt,
                )

                if txt_updated.dtype == torch.float16:
                    txt_updated = torch.nan_to_num(txt_updated, nan=0.0, posinf=65504, neginf=-65504)

                return img_updated, txt_updated

            block.forward = types.MethodType(bps_double_forward, block)

        log_rank0(self.rank, "info", f"{LOG_PREFIX} Patched {len(double_blocks)} double blocks")

        # Patch single blocks (similar pattern)
        single_blocks = getattr(diffusion_model, "single_blocks", [])
        for block in single_blocks:
            block._distributed_attn = DistributedAttention(
                num_heads=num_heads,
                head_dim=head_dim,
                device=self.device,
                dtype=torch.bfloat16,
                backend_name=bps_backend_name,
                **backend_kwargs,
            )

            def bps_single_forward(self, tensor, *, vec, pe, attn_mask=None, **kwargs):
                """Backend Plugin System single-stream forward."""
                mod_mlp, mod_attn = self.modulation(vec)

                norm_tensor = self.pre_norm(tensor)
                modulated_tensor = apply_mod(norm_tensor, (1 + mod_attn.scale), mod_attn.shift, None)

                qkv = self.attn.qkv(modulated_tensor)
                q, k, v = qkv.view(qkv.shape[0], qkv.shape[1], 3, self.num_heads, -1).permute(
                    2, 0, 3, 1, 4
                )
                q, k = self.attn.norm(q, k, v)

                attn_out = self._distributed_attn.forward(q, k, v, freqs_cis=pe, attn_mask=attn_mask)
                attn_out = apply_mod(self.attn.proj(attn_out), mod_attn.gate, None, None)

                mlp_out = self.mlp(apply_mod(norm_tensor, (1 + mod_mlp.scale), mod_mlp.shift, None))
                mlp_out = apply_mod(mlp_out, mod_mlp.gate, None, None)

                return tensor + attn_out + mlp_out

            block.forward = types.MethodType(bps_single_forward, block)

        log_rank0(self.rank, "info", f"{LOG_PREFIX} Patched {len(single_blocks)} single blocks")

        # Patch DiT orchestrator
        # Create closure to capture worker instance for backend-agnostic group access
        worker = self
        
        def bps_dit_forward(
            self,  # This is the DiT model instance
            img,
            img_ids,
            txt,
            txt_ids,
            timesteps,
            y,
            guidance=None,
            control=None,
            transformer_options=None,
            attn_mask=None,
        ):
            """Backend Plugin System DiT orchestrator forward (backend-agnostic)."""
            from .usp_dit_forward import pad_if_odd
            from comfy.ldm.flux.layers import timestep_embedding
            
            # Use worker's stored backend-agnostic accessors instead of importing xFuser
            get_sp_rank = worker._sp_get_rank
            get_sp_world_size = worker._sp_get_world_size

            transformer_options = transformer_options or {}
            patches_replace = transformer_options.get("patches_replace", {})

            img = pad_if_odd(img, dim=1)
            if img_ids is not None:
                img_ids = pad_if_odd(img_ids, dim=1)
            txt = pad_if_odd(txt, dim=1)
            if txt_ids is not None:
                txt_ids = pad_if_odd(txt_ids, dim=1)

            if y is None:
                y = torch.zeros(
                    (img.shape[0], self.params.vec_in_dim), device=img.device, dtype=img.dtype
                )

            img = self.img_in(img)
            vec = self.time_in(timestep_embedding(timesteps, 256).to(img.dtype))
            if self.params.guidance_embed and guidance is not None:
                vec = vec + self.guidance_in(timestep_embedding(guidance, 256).to(img.dtype))

            vec = vec + self.vector_in(y[:, : self.params.vec_in_dim])
            txt = self.txt_in(txt)

            if img_ids is not None:
                ids = torch.cat((txt_ids, img_ids), dim=1)
                pe_combine = self.pe_embedder(ids)
                pe_image = self.pe_embedder(img_ids)
                pe_combine = torch.chunk(pe_combine, get_sp_world_size(), dim=2)[get_sp_rank()]
                pe_image = torch.chunk(pe_image, get_sp_world_size(), dim=2)[get_sp_rank()]
            else:
                pe_combine = None
                pe_image = None

            img = torch.chunk(img, get_sp_world_size(), dim=1)[get_sp_rank()]
            txt = torch.chunk(txt, get_sp_world_size(), dim=1)[get_sp_rank()]

            blocks_replace = patches_replace.get("dit", {})
            for i, block in enumerate(self.double_blocks):
                block_key = ("double_block", i)
                if block_key in blocks_replace:

                    def block_wrap(args):
                        out_img, out_txt = block(
                            img=args["img"],
                            txt=args["txt"],
                            vec=args["vec"],
                            pe=args["pe"],
                            attn_mask=args.get("attn_mask"),
                        )
                        return {"img": out_img, "txt": out_txt}

                    patched = blocks_replace[block_key](
                        {
                            "img": img,
                            "txt": txt,
                            "vec": vec,
                            "pe": pe_combine,  # Use combined pe for txt+img
                            "attn_mask": attn_mask,
                            "transformer_options": transformer_options,
                        },
                        {"original_block": block_wrap},
                    )
                    img = patched["img"]
                    txt = patched["txt"]
                else:
                    img, txt = block(img=img, txt=txt, vec=vec, pe=pe_combine, attn_mask=attn_mask)

                if control is not None:
                    control_i = control.get("input")
                    if control_i and i < len(control_i):
                        add = control_i[i]
                        if add is not None:
                            img += add

            if img.dtype == torch.float16:
                img = torch.nan_to_num(img, nan=0.0, posinf=65504, neginf=-65504)

            img = get_sp_group().all_gather(img.contiguous(), dim=1)
            txt = get_sp_group().all_gather(txt.contiguous(), dim=1)

            txt_len = txt.shape[1]
            combined = torch.cat((txt, img), dim=1)
            combined = torch.chunk(combined, get_sequence_parallel_world_size(), dim=1)[
                get_sequence_parallel_rank()
            ]

            for i, block in enumerate(self.single_blocks):
                block_key = ("single_block", i)
                if block_key in blocks_replace:

                    def block_wrap(args):
                        out = block(
                            args["img"], vec=args["vec"], pe=args["pe"], attn_mask=args.get("attn_mask")
                        )
                        return {"img": out}

                    patched = blocks_replace[block_key](
                        {
                            "img": combined,
                            "vec": vec,
                            "pe": pe_combine,
                            "attn_mask": attn_mask,
                            "transformer_options": transformer_options,
                        },
                        {"original_block": block_wrap},
                    )
                    combined = patched["img"]
                else:
                    combined = block(combined, vec=vec, pe=pe_combine, attn_mask=attn_mask)

                if control is not None:
                    control_o = control.get("output")
                    if control_o and i < len(control_o):
                        add = control_o[i]
                        if add is not None:
                            combined[:, txt_len:] += add

            combined = get_sp_group().all_gather(combined, dim=1)
            result = combined[:, txt_len:]

            return self.final_layer(result, vec)

        diffusion_model.forward_orig = types.MethodType(bps_dit_forward, diffusion_model)
        log_rank0(self.rank, "info", f"{LOG_PREFIX} Patched DiT orchestrator")

    def _patch_usp_forwards_after_load(self, fsdp_model, usp_config):
        """Patch USP forwards after weights loaded into FSDP-wrapped blocks.
        
        Called immediately after load_state_dict but before ModelPatcher wrapping.
        At this point blocks are FSDP modules with loaded weights, so we can safely
        patch their forward methods without DTensor mixing issues.
        """
        if self.usp_enabled:
            return

        ulysses_degree = int(usp_config.get("ulysses_degree", 1))
        ring_degree = int(usp_config.get("ring_degree", 1))
        attention_backend = usp_config.get("attention_backend", "FLASH_ATTN")

        if ulysses_degree <= 1 and ring_degree <= 1:
            return

        sequence_degree = ulysses_degree * ring_degree
        if sequence_degree != self.world_size:
            raise RuntimeError(
                f"{LOG_PREFIX} USP configuration mismatch: ulysses_degree={ulysses_degree}, "
                f"ring_degree={ring_degree}, world_size={self.world_size}"
            )

        try:
            from xfuser.core.distributed import (
                init_distributed_environment,
                initialize_model_parallel,
            )
        except ImportError as exc:
            raise RuntimeError("xfuser package is required for USP attention") from exc

        if not self._usp_parallel_initialized:
            init_distributed_environment(rank=self.rank, world_size=self.world_size)
            initialize_model_parallel(
                sequence_parallel_degree=self.world_size,
                ring_degree=ring_degree,
                ulysses_degree=ulysses_degree,
            )
            self._usp_parallel_initialized = True

        from .usp_attention import initialize_usp_attention
        from .usp_single_forward import usp_single_forward
        from .usp_double_forward import usp_double_forward
        from .usp_dit_forward import usp_dit_forward

        initialize_usp_attention(
            ulysses_degree,
            ring_degree,
            attn_type=attention_backend,
        )

        # Patch blocks directly on FSDP model (blocks have weights loaded, no DTensor issues)
        diffusion_model = fsdp_model.diffusion_model
        double_blocks = getattr(diffusion_model, "double_blocks", [])
        single_blocks = getattr(diffusion_model, "single_blocks", [])

        patched_double = 0
        for block in double_blocks:
            block.forward = types.MethodType(usp_double_forward, block)
            patched_double += 1

        patched_single = 0
        for block in single_blocks:
            block.forward = types.MethodType(usp_single_forward, block)
            patched_single += 1

        diffusion_model.forward_orig = types.MethodType(usp_dit_forward, diffusion_model)

        self.usp_config = {
            "ulysses_degree": ulysses_degree,
            "ring_degree": ring_degree,
            "attention_backend": attention_backend,
        }
        self.usp_enabled = True

        log_rank0(
            self.rank,
            'info',
            (
                f"{LOG_PREFIX} USP forwards patched (ulysses={ulysses_degree}, ring={ring_degree}, "
                f"double_blocks={patched_double}, single_blocks={patched_single}, backend={attention_backend})"
            ),
        )
    
    def _apply_fastvideo_ulysses_patches(self, fsdp_model, ulysses_degree: int, attention_backend: str = "sdpa"):
        """Apply FastVideo-style pure Ulysses patches (no xFuser dependency).
        
        This is a minimal, self-contained implementation using raw torch.distributed.
        
        Args:
            fsdp_model: FSDP2-wrapped model
            ulysses_degree: Degree of Ulysses parallelism (typically world_size)
            attention_backend: Backend for local attention - "flash", "sdpa", or "math"
        """
        import types
        
        LOG_PREFIX = f"⚡ [Worker][Rank {self.rank}][FastVideo-Ulysses]"
        
        # Initialize SP group using our minimal communicator
        from comfy.parallel_attention.fastvideo_ulysses.communicator import initialize_sp_group
        
        if not hasattr(self, '_fastvideo_sp_initialized'):
            initialize_sp_group(sp_degree=ulysses_degree)
            self._fastvideo_sp_initialized = True
            log_rank0(self.rank, 'info', f"{LOG_PREFIX} SP group initialized (degree={ulysses_degree})")
        
        # Set attention backend globally for this worker
        from comfy.parallel_attention.fastvideo_ulysses.hooks import set_attention_backend
        set_attention_backend(attention_backend)
        log_rank0(self.rank, 'info', f"{LOG_PREFIX} Attention backend: {attention_backend}")
        
        # Import forward hooks
        from comfy.parallel_attention.fastvideo_ulysses.hooks import (
            ulysses_dit_forward,
            ulysses_double_block_forward,
            ulysses_single_block_forward,
        )
        
        # Patch blocks
        diffusion_model = fsdp_model.diffusion_model
        double_blocks = getattr(diffusion_model, "double_blocks", [])
        single_blocks = getattr(diffusion_model, "single_blocks", [])
        
        patched_double = 0
        for block in double_blocks:
            block.forward = types.MethodType(ulysses_double_block_forward, block)
            patched_double += 1
        
        patched_single = 0
        for block in single_blocks:
            block.forward = types.MethodType(ulysses_single_block_forward, block)
            patched_single += 1
        
        # Patch DiT forward
        diffusion_model.forward_orig = types.MethodType(ulysses_dit_forward, diffusion_model)
        
        log_rank0(
            self.rank,
            'info',
            f"{LOG_PREFIX} Pure Ulysses forwards patched (ulysses={ulysses_degree}, "
            f"double_blocks={patched_double}, single_blocks={patched_single})"
        )
    
    def _apply_object_patches_to_worker_model(self, object_patches: dict):
        """Apply object patches to worker's FSDP2-sharded model.
        
        This is the OFFICIAL extension point for distributed model modifications.
        Solves the core problem: parent process installs patches via add_object_patch,
        but FSDP2 sharding in workers creates new model graph. This method reapplies
        patches AFTER sharding.
        
        Pattern: Clean alternative to ad-hoc monkey-patching.
        
        Args:
            object_patches: Dict mapping module paths to replacement modules.
                           Serialized from parent's model.object_patches.
        
        Example:
            {
                "diffusion_model.double_blocks.0.img_attn": <CustomModule>,
                "diffusion_model.double_blocks.0.txt_attn": <CustomModule>,
                ...
            }
        """
        LOG_PREFIX = f"⚡ [Worker][Rank {self.rank}]"
        
        if not object_patches:
            return
        
        log_rank0(self.rank, 'info', 
            f"{LOG_PREFIX} Applying {len(object_patches)} object patches to worker model"
        )
        
        applied_count = 0
        failed_count = 0
        
        for path, patch_module in object_patches.items():
            try:
                # Navigate to parent module
                parts = path.split(".")
                parent = self.model.model  # Access inner BaseModel
                
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                
                # Replace target module
                target_name = parts[-1]
                setattr(parent, target_name, patch_module)
                applied_count += 1
                
                if applied_count <= 5:  # Log first few for debugging
                    log_rank0(self.rank, 'debug', f"{LOG_PREFIX}   ✅ {path}")
                    
            except AttributeError as e:
                failed_count += 1
                log_rank0(self.rank, 'warning', 
                    f"{LOG_PREFIX}   ❌ Failed to patch {path}: {e}"
                )
        
        log_rank0(self.rank, 'info', 
            f"{LOG_PREFIX} Object patches applied: {applied_count} success, {failed_count} failed"
        )
    
    def _apply_model_step(self, args: dict):
        """Execute single apply_model forward pass (Phase 2 per-step pattern).
        
        This is the "Dumb Worker" handler for Phase 2. Workers receive
        per-step kwargs, execute one forward pass, return output.
        
        This is the stateless RPC pattern - workers don't track step count,
        session state, or sampling parameters. They just execute the forward pass.
        
        Args:
            args: {
                "x": input tensor (latent),
                "timestep": timestep tensor,
                "c": conditioning dict,
                "ring_enabled": bool (optional, for Ring-Attention)
            }
        
        Returns:
            {"output": tensor} - model output for this step (rank 0 only)
        """
        LOG_PREFIX = f"⚡ [Worker][Rank {self.rank}][ApplyStep]"
        
        # Move inputs to worker device
        x = args["x"].to(self.device)
        timestep = args["timestep"].to(self.device)
        c_dict = args.get("c", {})
        
        # Extract conditioning (c_crossattn and y) FIRST
        c_crossattn = c_dict.get("c_crossattn")
        if c_crossattn is not None:
            c_crossattn = c_crossattn.to(self.device)
        
        y = c_dict.get("y")
        if y is not None:
            y = y.to(self.device)
        
        transformer_options = c_dict.get("transformer_options", {})
        
        # Execute single forward pass with FSDP2 model
        with torch.no_grad():
            kwargs = {}
            if c_crossattn is not None:
                kwargs["c_crossattn"] = c_crossattn
            if y is not None:
                kwargs["y"] = y
            if transformer_options:
                kwargs["transformer_options"] = transformer_options
            
            output = self.model.model.apply_model(x, timestep, **kwargs)
        
        # Only rank 0 returns output (needs CPU for multiprocessing Queue pickling)
        if self.rank == 0:
            return {"output": output.cpu()}
        else:
            # Rank 1+ returns empty dict (no dummy tensor needed)
            return {}
    
    def _move_conditioning_to_device(self, c: dict):
        """Recursively move conditioning tensors to worker device."""
        result = {}
        for key, value in c.items():
            if isinstance(value, torch.Tensor):
                result[key] = value.to(self.device)
            elif isinstance(value, dict):
                result[key] = self._move_conditioning_to_device(value)
            elif isinstance(value, list):
                result[key] = [v.to(self.device) if isinstance(v, torch.Tensor) else v for v in value]
            else:
                result[key] = value
        return result
    
    def _common_ksampler(self, args: dict):
        """Execute sampling using standard ComfyUI APIs.
        
        Adapted from distributed reference implementation.
        Workers execute full comfy.sample.sample() with FSDP2 model.
        
        Args:
            args: {
                "seed": int,
                "steps": int,
                "cfg": float,
                "sampler_name": str,
                "scheduler": str,
                "positive": conditioning,
                "negative": conditioning,
                "latent": {"samples": tensor},
                "denoise": float,
                "disable_noise": bool,
                "start_step": int,
                "last_step": int,
                "force_full_denoise": bool
            }
        
        Returns:
            {"status": "success", "result": {"samples": tensor}} (rank 0 only)
        """
        import comfy.sample
        import comfy.utils
        
        LOG_PREFIX = f"⚡ [Parallel-Attention][Worker][Rank {self.rank}][Forward]"
        
        # Extract args
        seed = args["seed"]
        steps = args["steps"]
        cfg = args["cfg"]
        sampler_name = args["sampler_name"]
        scheduler = args["scheduler"]
        positive = args["positive"]
        negative = args["negative"]
        latent = args["latent"]
        denoise = args.get("denoise", 1.0)
        disable_noise = args.get("disable_noise", False)
        start_step = args.get("start_step", None)
        last_step = args.get("last_step", None)
        force_full_denoise = args.get("force_full_denoise", False)
        
        try:
            # Step 1: Extract and fix latent channels
            latent_image = latent["samples"]
            latent_image = comfy.sample.fix_empty_latent_channels(self.model, latent_image)
            
            log_rank0(self.rank, 'debug', f"{LOG_PREFIX} Latent shape: {latent_image.shape}")
            
            # Step 2: Prepare noise
            if disable_noise:
                noise = torch.zeros(
                    latent_image.size(),
                    dtype=latent_image.dtype,
                    layout=latent_image.layout,
                    device="cpu",
                )
            else:
                batch_inds = latent.get("batch_index", None)
                noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)
            
            # Step 3: Get noise mask if present
            noise_mask = latent.get("noise_mask", None)
            
            # Step 4: Disable progress bar on non-rank-0
            disable_pbar = True
            if self.rank == 0:
                disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
            
            # Log VRAM BEFORE sampling
            vram_before = 0.0
            if torch.cuda.is_available():
                vram_before = torch.cuda.memory_allocated(self.device) / 1024**3
                log_rank0(self.rank, 'info', f"{LOG_PREFIX} VRAM before sampling: {vram_before:.3f}GB")
            
            log_rank0(self.rank, 'info', f"{LOG_PREFIX} Starting sampling: steps={steps}, cfg={cfg}")
            
            # Add barrier to sync all ranks before sampling
            import torch.distributed as dist
            if dist.is_initialized():
                log_rank0(self.rank, 'debug', f"{LOG_PREFIX} Syncing ranks before sampling...")
                dist.barrier()
                log_rank0(self.rank, 'debug', f"{LOG_PREFIX} All ranks synced")
            
            # Step 5: Execute standard ComfyUI sampling
            try:
                with torch.no_grad():
                    samples = comfy.sample.sample(
                        self.model,
                        noise,
                        steps,
                        cfg,
                        sampler_name,
                        scheduler,
                        positive,
                        negative,
                        latent_image,
                        denoise=denoise,
                        disable_noise=disable_noise,
                        start_step=start_step,
                        last_step=last_step,
                        force_full_denoise=force_full_denoise,
                        noise_mask=noise_mask,
                        disable_pbar=disable_pbar,
                        seed=seed,
                    )
            except RuntimeError as e:
                if "same number of dimensions" in str(e):
                    log_rank0(self.rank, 'error', f"{LOG_PREFIX} DIMENSION MISMATCH ERROR: {e}")
                    log_rank0(self.rank, 'error', f"{LOG_PREFIX} This error happens during forward pass")
                    log_rank0(self.rank, 'error', f"{LOG_PREFIX} Likely in xfuser attention call")
                raise
            
            # Log VRAM AFTER sampling
            vram_after = 0.0
            if torch.cuda.is_available():
                vram_after = torch.cuda.memory_allocated(self.device) / 1024**3
                vram_delta = vram_after - vram_before
                log_rank0(self.rank, 'info', f"{LOG_PREFIX} VRAM after sampling: {vram_after:.3f}GB")
                log_rank0(self.rank, 'info', f"{LOG_PREFIX} VRAM delta: {vram_delta:+.3f}GB")
            
            log_rank0(self.rank, 'info', f"{LOG_PREFIX} Sampling complete")
            
            # Step 6: Return result (rank 0 only)
            if self.rank == 0:
                out = latent.copy()
                out["samples"] = samples
                return {"status": "success", "result": out}
            else:
                return {"status": "success", "result": None}
                
        except Exception as e:
            logging.error(f"{LOG_PREFIX} Error during sampling: {e}", exc_info=True)
            return {"status": "error", "error": str(e), "rank": self.rank}
