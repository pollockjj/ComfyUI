"""FSDP2Worker - Distributed inference worker with FSDP2 sharding.

Implements multi-GPU model sharding using PyTorch FSDP2 and DeviceMesh.
Based on FastVideo worker pattern.
"""

import torch
import logging
import os

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
    """Worker process for FSDP2 distributed inference.
    
    Each worker:
    1. Initializes on assigned device (cuda:rank)
    2. Creates DeviceMesh for FSDP2 coordination
    3. Loads model shard with FSDP2 wrapping
    4. Executes inference commands from parent
    
    Pattern: Distributed Patcher (workers hold ModelPatcher instances).
    """
    
    def __init__(self, rank: int, world_size: int, device_id: int):
        """Initialize worker.
        
        Args:
            rank: Worker rank (0-based)
            world_size: Total number of workers
            device_id: CUDA device ID for this worker
        """
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(f"cuda:{device_id}")
        self.model = None
        self.device_mesh = None
        
        # Set device
        torch.cuda.set_device(self.device)
        
        self.use_backend_plugin_system = (
            os.getenv("USE_BACKEND_PLUGIN_SYSTEM", "false").lower() == "true"
        )
        if self.use_backend_plugin_system:
            log_rank0(
                rank,
                'info',
                f"{LOG_PREFIX}[Worker][Rank {rank}] Backend Plugin System feature flag enabled",
            )
        else:
            log_rank0(
                rank,
                'debug',
                f"{LOG_PREFIX}[Worker][Rank {rank}] Backend Plugin System feature flag disabled",
            )

        log_rank0(rank, 'info', f"{LOG_PREFIX}[Worker][Rank {rank}] Initialized on {self.device}")
    
    def execute(self, command: str, args: dict):
        """Execute command from parent process.
        
        Args:
            command: Command name
            args: Command arguments
        
        Returns:
            dict: {"status": "success"|"error", ...}
        """
        if command == "initialize_fsdp2_from_checkpoint":
            return self._initialize_fsdp2_from_checkpoint(args)
        elif command == "apply_model":
            return self._apply_model(args)
        elif command == "get_worker_status":
            return self._get_worker_status()
        elif command == "shutdown":
            return {"status": "success"}
        else:
            return {"status": "error", "error": f"Unknown command: {command}"}
    
    def _get_worker_status(self):
        """Get worker status for debugging."""
        return {
            "status": "success",
            "rank": self.rank,
            "device": str(self.device),
            "model_loaded": self.model is not None,
            "device_mesh": str(self.device_mesh) if self.device_mesh else None,
        }
    
    def _get_model_info(self):
        """Get model information for golden data collection."""
        if not self.model:
            return {"status": "error", "error": "No model loaded"}
        
        import comfy.model_detection as model_detection
        
        # Detect model type
        config = model_detection.detect_unet_config(self.model.model)
        
        return {
            "status": "success",
            "model_type": type(self.model.model).__name__,
            "config_name": config.model_name,
            "block_count": len(config.blocks),
        }
    
    def _apply_model(self, args: dict):
        """Execute apply_model for FSDP2 distributed inference.
        
        INSTRUMENTED VERSION - Tracks CUDA memory allocations.
        
        Args:
            args: kwargs from sampler (tensors are on CPU, need to move to device)
        
        Returns:
            dict: {"status": "success", "result": tensor} (rank 0 only)
        """
        LOG_PREFIX = f"⚡ [Worker][Rank {self.rank}][ApplyModel]"
        
        try:
            # Log VRAM before starting
            vram_before = torch.cuda.memory_allocated(self.device) / 1024**3
            current_device = torch.cuda.current_device()
            logging.info(f"{LOG_PREFIX} Using device: {self.device}, current_device={current_device}, VRAM before: {vram_before:.2f}GB")
            
            # Barrier sync
            import torch.distributed as dist
            if dist.is_initialized():
                logging.info(f"{LOG_PREFIX} Waiting at barrier...")
                dist.barrier()
                logging.info(f"{LOG_PREFIX} Barrier passed")
            
            # Move tensors to device
            logging.info(f"{LOG_PREFIX} Moving tensors to device {self.device}...")
            args_device = self._move_tensors_to_device(args, self.device)
            
            vram_after_move = torch.cuda.memory_allocated(self.device) / 1024**3
            logging.info(f"{LOG_PREFIX} VRAM after tensor move: {vram_after_move:.2f}GB (delta: {vram_after_move - vram_before:.2f}GB)")
            
            # Prepare apply_model args
            apply_model_args = {
                "x": args_device["input"],
                "t": args_device["timestep"],
                "c_concat": args_device["c"].get("c_concat", None) if "c" in args_device else None,
                "c_crossattn": args_device["c"].get("c_crossattn", None) if "c" in args_device else None,
                "control": args_device["c"].get("control", None) if "c" in args_device else None,
                "transformer_options": args_device["c"].get("transformer_options", {}) if "c" in args_device else {},
                "cond_or_uncond": args_device.get("cond_or_uncond", None),
            }
            
            logging.info(f"{LOG_PREFIX} Prepared apply_model args, x.shape={apply_model_args['x'].shape}")
            
            # ====================
            # MEMORY INSTRUMENTATION
            # ====================
            logging.info(f"{LOG_PREFIX} 🔍 Starting CUDA memory profiling...")
            
            torch.cuda.memory._record_memory_history(enabled=True, max_entries=100000)
            
            vram_before_forward = torch.cuda.memory_allocated(self.device) / 1024**3
            vram_reserved_before = torch.cuda.memory_reserved(self.device) / 1024**3
            logging.info(f"{LOG_PREFIX} 🔍 BEFORE apply_model: allocated={vram_before_forward:.3f}GB, reserved={vram_reserved_before:.3f}GB")
            
            # Execute apply_model
            logging.info(f"{LOG_PREFIX} Executing apply_model...")
            with torch.no_grad():
                result = self.model.model.apply_model(**apply_model_args)
            
            vram_after_forward = torch.cuda.memory_allocated(self.device) / 1024**3
            vram_reserved_after = torch.cuda.memory_reserved(self.device) / 1024**3
            logging.info(f"{LOG_PREFIX} 🔍 AFTER apply_model: allocated={vram_after_forward:.3f}GB, reserved={vram_reserved_after:.3f}GB")
            logging.info(f"{LOG_PREFIX} 🔍 DELTA: allocated={vram_after_forward - vram_before_forward:.3f}GB, reserved={vram_reserved_after - vram_reserved_before:.3f}GB")
            
            # Save memory snapshot
            snapshot_dir = "/home/johnj/parallel-attention/unified_sequence/test_logs"
            os.makedirs(snapshot_dir, exist_ok=True)
            snapshot_file = os.path.join(snapshot_dir, f"cuda_memory_rank{self.rank}.pickle")
            
            try:
                torch.cuda.memory._dump_snapshot(snapshot_file)
                logging.info(f"{LOG_PREFIX} 🔍 Memory snapshot saved to: {snapshot_file}")
                logging.info(f"{LOG_PREFIX} 🔍 Analyze with: python -m torch.cuda._memory_viz trace_plot {snapshot_file} -o memory_trace_rank{self.rank}.html")
            except Exception as snap_err:
                logging.warning(f"{LOG_PREFIX} Could not save snapshot: {snap_err}")
            
            torch.cuda.memory._record_memory_history(enabled=False)
            logging.info(f"{LOG_PREFIX} 🔍 Memory profiling complete")
            # ====================
            # END INSTRUMENTATION
            # ====================
            
            logging.info(f"{LOG_PREFIX} Forward complete, result shape={result.shape}")
            
            # Return result
            if self.rank == 0:
                result_cpu = result.cpu()
                return {"status": "success", "result": result_cpu}
            else:
                return {"status": "success"}
                
        except Exception as e:
            logging.error(f"{LOG_PREFIX} Error: {e}", exc_info=True)
            return {"status": "error", "error": str(e)}
    
    def _move_tensors_to_device(self, obj, device):
        """Recursively move all tensors to device."""
        if isinstance(obj, torch.Tensor):
            return obj.to(device)
        elif isinstance(obj, dict):
            return {k: self._move_tensors_to_device(v, device) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return type(obj)(self._move_tensors_to_device(item, device) for item in obj)
        else:
            return obj
    
    def _initialize_fsdp2_from_checkpoint(self, args: dict):
        """Initialize FSDP2-sharded model from checkpoint."""
        LOG_PREFIX = f"⚡ [Worker][Rank {self.rank}][Init]"
        
        try:
            checkpoint_path = args["checkpoint_path"]
            model_type = args["model_type"]
            policy_name = args.get("policy_name", "flux_dev_fsdp2")
            requested_bps = bool(args.get("use_backend_plugin_system", False))
            if requested_bps and not self.use_backend_plugin_system:
                log_rank0(
                    self.rank,
                    'warning',
                    f"{LOG_PREFIX} Requested Backend Plugin System but feature flag disabled for worker",
                )
            elif self.use_backend_plugin_system and not requested_bps:
                log_rank0(
                    self.rank,
                    'debug',
                    f"{LOG_PREFIX} Worker feature flag enabled; upstream request did not opt-in",
                )
            
            logging.info(f"{LOG_PREFIX} Initializing FSDP2 from {checkpoint_path}")
            
            # Initialize DeviceMesh
            from torch.distributed.device_mesh import init_device_mesh
            import torch.distributed as dist
            
            if not dist.is_initialized():
                logging.info(f"{LOG_PREFIX} Initializing process group...")
                dist.init_process_group(
                    backend="nccl",
                    rank=self.rank,
                    world_size=self.world_size,
                    device_id=self.device
                )
            
            self.device_mesh = init_device_mesh(
                "cuda",
                (self.world_size,),
                mesh_dim_names=("fsdp",)
            )
            
            log_rank0(self.rank, 'info', f"{LOG_PREFIX} DeviceMesh created: {self.device_mesh}")
            
            # Load model with FSDP2
            from comfy.parallel_attention.fsdp2_engine import load_model_with_fsdp2
            from comfy import model_management
            
            vram_before = torch.cuda.memory_allocated(self.device) / 1024**3
            logging.info(f"{LOG_PREFIX} VRAM before load: {vram_before:.2f}GB")
            
            fsdp_model = load_model_with_fsdp2(
                checkpoint_path=checkpoint_path,
                model_type=model_type,
                policy_name=policy_name,
                device_mesh=self.device_mesh,
                rank=self.rank
            )
            
            vram_after = torch.cuda.memory_allocated(self.device) / 1024**3
            logging.info(f"{LOG_PREFIX} VRAM after load: {vram_after:.2f}GB (delta: {vram_after - vram_before:.2f}GB)")
            
            # Wrap in ModelPatcher
            import comfy.model_patcher
            
            self.model = comfy.model_patcher.ModelPatcher(
                fsdp_model,
                load_device=self.device,
                offload_device=model_management.unet_offload_device()
            )
            
            log_rank0(self.rank, 'info', f"{LOG_PREFIX} ✅ FSDP2 initialized, VRAM: {vram_after:.2f}GB")
            
            return {"status": "success", "vram_gb": vram_after}
            
        except Exception as e:
            logging.error(f"{LOG_PREFIX} Initialization failed: {e}", exc_info=True)
            return {"status": "error", "error": str(e)}
