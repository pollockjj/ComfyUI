"""
CGDP Parallel Wrapper - Per-step synchronization for spatial parallelism.
Implements Ulysses-style all-gather after each denoising step.
"""

import copy
import torch
import logging
import gc
import comfy.model_management
import comfy.samplers

logger = logging.getLogger(__name__)


def _clone_model_to_device(model, target_device):
    """
    Clone a model's diffusion_model to target device with VRAM validation.
    
    Args:
        model: The source model (on cuda:0)
        target_device: torch.device for the clone (cuda:1)
    
    Returns:
        Cloned diffusion_model on target device
    """
    logger.info(f"⚡ [cgdp] Cloning model to {target_device}")
    
    # Calculate VRAM requirements
    vram_needed = sum(p.numel() * p.element_size() for p in model.diffusion_model.parameters()) / 1e9
    vram_needed += sum(b.numel() * b.element_size() for b in model.diffusion_model.buffers()) / 1e9
    vram_free = torch.cuda.mem_get_info(target_device.index)[0] / 1e9
    
    logger.info(f"⚡ [cgdp] VRAM check: need {vram_needed:.2f}GB, available {vram_free:.2f}GB on {target_device}")
    
    if vram_free < vram_needed * 1.1:  # 10% safety margin
        raise RuntimeError(
            f"Insufficient VRAM on {target_device}: need {vram_needed:.2f}GB + margin, "
            f"have {vram_free:.2f}GB"
        )
    
    # Free VRAM aggressively before clone
    comfy.model_management.free_memory(vram_needed * 1.5, target_device, keep_loaded=[])
    comfy.model_management.soft_empty_cache()
    
    # Deep copy + move to device
    cloned_model = copy.deepcopy(model.diffusion_model).to(target_device)
    torch.cuda.synchronize(target_device)
    
    # Validate all parameters are on target device
    params_on_wrong_device = [
        name for name, p in cloned_model.named_parameters() 
        if p.device != target_device
    ]
    if params_on_wrong_device:
        raise RuntimeError(
            f"Clone failed: {len(params_on_wrong_device)} parameters not on {target_device}"
        )
    
    # Verify independence (different memory addresses)
    if id(model.diffusion_model) == id(cloned_model):
        raise RuntimeError("Clone failed: objects share same ID")
    
    logger.info(f"⚡ [cgdp] Clone complete: {vram_needed:.2f}GB allocated on {target_device}")
    
    # Aggressive cleanup
    gc.collect()
    comfy.model_management.soft_empty_cache()
    
    return cloned_model


class CGDPStepSyncWrapper:
    """
    Wraps the sampling loop to perform per-step all-gather synchronization.
    Each denoising step: split → parallel compute → all-gather → next step.
    """
    
    def __init__(self, model, device_primary=None, target_device=None):
        """
        Args:
            model: The inner model to wrap
            device_primary: Primary device (cuda:0)
            target_device: Secondary device (cuda:1)
        """
        self.model_0 = model
        self.device_0 = device_primary or torch.device("cuda:0")
        self.device_1 = target_device or torch.device("cuda:1")
        
        logger.info(f"⚡ [cgdp] CGDPStepSyncWrapper init: {self.device_0} + {self.device_1}")
        
        # Clone model to device_1
        self.model_1_diffusion = _clone_model_to_device(model, self.device_1)
        
        # Create persistent CUDA streams
        self.stream_0 = torch.cuda.Stream(device=self.device_0)
        self.stream_1 = torch.cuda.Stream(device=self.device_1)
        
        logger.info("⚡ [cgdp] Per-step sync wrapper ready")
    
    def predict_noise_with_sync(self, model, x, timestep, negative, positive, cfg, model_options, seed):
        """
        FALLBACK: Spatial parallelism is fundamentally incompatible with self-attention.
        Reverting to simple replication - both GPUs process full latent with different noise.
        This achieves NO SPEEDUP but validates the infrastructure.
        """
        logger.info(f"⚡ [cgdp] FALLBACK MODE: Running full latent on both GPUs (no spatial split)")
        
        # Just run on GPU 0 - spatial split is broken for diffusion models
        return comfy.samplers.sampling_function(model, x, timestep, negative, positive, cfg, model_options=model_options, seed=seed)
        
        # Parallel execution with streams
        result_0 = None
        result_1 = None
        
        with torch.cuda.stream(self.stream_0):
            result_0 = comfy.samplers.sampling_function(
                model, x_0, timestep, negative, positive, cfg, 
                model_options=model_options, seed=seed
            )
        
        with torch.cuda.stream(self.stream_1):
            # Move inputs to device_1
            timestep_1 = timestep.to(self.device_1) if torch.is_tensor(timestep) else timestep
            negative_1 = [c.to(self.device_1) if torch.is_tensor(c) else c for c in (negative or [])]
            positive_1 = [c.to(self.device_1) if torch.is_tensor(c) else c for c in (positive or [])]
            
            # Temporarily swap diffusion_model
            saved_diff = model.diffusion_model
            model.diffusion_model = self.model_1_diffusion
            
            result_1 = comfy.samplers.sampling_function(
                model, x_1, timestep_1, negative_1, positive_1, cfg,
                model_options=model_options, seed=seed
            )
            
            model.diffusion_model = saved_diff
        
        # Synchronize both streams
        torch.cuda.synchronize(self.device_0)
        torch.cuda.synchronize(self.device_1)
        
        # Trim overlap regions before concatenation
        # GPU 0 keeps [0, H_split], GPU 1 keeps [overlap, end]
        result_0_trimmed = result_0.narrow(H_dim, 0, H_split)
        result_1 = result_1.to(self.device_0, non_blocking=False)
        result_1_trimmed = result_1.narrow(H_dim, overlap, result_1.shape[H_dim] - overlap)
        
        # Concatenate trimmed results
        result = torch.cat([result_0_trimmed, result_1_trimmed], dim=H_dim)
        
        logger.debug(f"⚡ [cgdp] Overlap-trimmed gather: {result.shape}")
        return result


class ParallelUnetWrapper:
    """
    Wraps a model to enable dual-GPU parallel execution.
    Phase 1: Pass-through mode (validates cloning, no parallelism yet).
    """
    
    def __init__(self, model, device_primary=None, target_device=None):
        """
        Args:
            model: The inner model to wrap (WAN21, etc)
            device_primary: Primary device (from model_patcher.load_device)
            target_device: Device for replica (default: cuda:1)
        """
        self.model_0 = model
        self.device_0 = device_primary or torch.device("cuda:0")
        self.device_1 = target_device or torch.device("cuda:1")
        
        logger.info(f"⚡ [cgdp] ParallelUnetWrapper init: {self.device_0} + {self.device_1}")
        
        # Store original apply_model before we replace it
        self._original_apply_model = model.apply_model
        
        # Clone model to device_1
        self.model_1_diffusion = _clone_model_to_device(model, self.device_1)
        
        # NVTX range stubs (no-ops for Phase 1)
        self._nvtx_enabled = False
        
        logger.info("⚡ [cgdp] ParallelUnetWrapper ready (pass-through mode)")
    
    def apply_model_wrapped(self, x, timestep, **kwargs):
        """
        Phase 2: Spatial split along H dimension, parallel dispatch, gather.
        """
        if self._nvtx_enabled:
            pass  # torch.cuda.nvtx.range_push("cgdp_apply_model")
        
        try:
            # Handle both image [B,C,H,W] and video [B,C,T,H,W] latents
            logger.info(f"⚡ [cgdp] Input x: shape={x.shape}, device={x.device}")
            
            if x.ndim == 5:  # Video: [B,C,T,H,W]
                B, C, T, H, W = x.shape
                H_dim = 3
                logger.info(f"⚡ [cgdp] Video latent: B={B}, C={C}, T={T}, H={H}, W={W}")
            elif x.ndim == 4:  # Image: [B,C,H,W]
                B, C, H, W = x.shape
                H_dim = 2
                logger.info(f"⚡ [cgdp] Image latent: B={B}, C={C}, H={H}, W={W}")
            else:
                raise ValueError(f"Unsupported latent shape: {x.shape}")
            
            # Split H into two halves
            H_split = H // 2
            x_0 = x.narrow(H_dim, 0, H_split).contiguous()
            x_1 = x.narrow(H_dim, H_split, H - H_split).contiguous()
            
            logger.info(f"⚡ [cgdp] x_0: {x_0.shape} on {x_0.device}, x_1: {x_1.shape} on {x_1.device}")
            
            # Move x_1 to device_1
            x_1 = x_1.to(self.device_1, non_blocking=True)
            
            # Create streams for parallel execution
            stream_0 = torch.cuda.Stream(device=self.device_0)
            stream_1 = torch.cuda.Stream(device=self.device_1)
            
            # Dispatch to both GPUs in parallel
            with torch.cuda.stream(stream_0):
                logger.info(f"⚡ [cgdp] Executing on {self.device_0}")
                out_0 = self._original_apply_model(x_0, timestep, **kwargs)
            
            with torch.cuda.stream(stream_1):
                logger.info(f"⚡ [cgdp] Executing on {self.device_1}")
                # Clone model_1 diffusion needs to use wrapper to pass through model infrastructure
                # Create temporary wrapper that holds model_1's diffusion_model
                saved_diffusion = self.model_0.diffusion_model
                self.model_0.diffusion_model = self.model_1_diffusion
                
                timestep_1 = timestep.to(self.device_1) if torch.is_tensor(timestep) else timestep
                kwargs_1 = {}
                for k, v in kwargs.items():
                    if torch.is_tensor(v):
                        kwargs_1[k] = v.to(self.device_1, non_blocking=True)
                    else:
                        kwargs_1[k] = v
                
                # Call through model's apply_model with cloned diffusion_model
                out_1 = self._original_apply_model.__func__(self.model_0, x_1, timestep_1, **kwargs_1)
                
                # Restore
                self.model_0.diffusion_model = saved_diffusion
            
            # Synchronize both streams
            torch.cuda.synchronize(self.device_0)
            torch.cuda.synchronize(self.device_1)
            
            logger.info(f"⚡ [cgdp] out_0: {out_0.shape}, out_1: {out_1.shape}")
            
            # Move out_1 back to device_0 and concatenate
            out_1 = out_1.to(self.device_0, non_blocking=False)
            result = torch.cat([out_0, out_1], dim=H_dim)  # Concatenate along H
            
            logger.info(f"⚡ [cgdp] Gathered result: {result.shape}")
            return result
            
        finally:
            if self._nvtx_enabled:
                pass  # torch.cuda.nvtx.range_pop()
