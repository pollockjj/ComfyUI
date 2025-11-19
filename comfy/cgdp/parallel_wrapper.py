"""
CGDP Parallel Wrapper - Model cloning and pass-through execution.
Phase 1: Clone model to cuda:1, delegate to model_0 (pass-through mode).
Phase 2: Implement spatial split + parallel dispatch.
"""

import copy
import torch
import logging
import gc
import comfy.model_management

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
        Phase 1: Pass-through to model_0 only (validate wiring).
        Phase 2: Split x, dispatch to both GPUs, gather results.
        """
        # NVTX range stub (no-op)
        if self._nvtx_enabled:
            pass  # torch.cuda.nvtx.range_push("cgdp_apply_model")
        
        try:
            # Phase 1: Delegate to original model method
            logger.debug(f"⚡ [cgdp] apply_model called: x.shape={x.shape}, device={x.device}")
            result = self._original_apply_model(x, timestep, **kwargs)
            logger.debug(f"⚡ [cgdp] apply_model result: shape={result.shape}")
            return result
        finally:
            if self._nvtx_enabled:
                pass  # torch.cuda.nvtx.range_pop()
