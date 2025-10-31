"""FSDP2ModelPatcher - Distributed inference via meta device interface.

Extends ModelPatcher to support FSDP2 distributed inference while maintaining
100% compatibility with ComfyUI's existing workflow system.

Architecture:
- Meta device model stays as self.model permanently (never replaced)
- FSDP2 workers hold sharded model for actual computation
- Forward pass intercepted via PyTorch hook, relayed to workers
- ComfyUI sees standard ModelPatcher API, unaware of distribution

Based on Raylight meta interface pattern + FastVideo multiprocess.
"""

from comfy.model_patcher import ModelPatcher
import torch
import logging

LOG_PREFIX = "⚡ [FSDP2]"


class FSDP2ModelPatcher(ModelPatcher):
    """ModelPatcher with FSDP2 distributed inference support.
    
    Meta device model stays as self.model permanently.
    Workers hold FSDP2 sharded model for actual computation.
    Forward pass intercepted via PyTorch hook and relayed to workers.
    
    Based on Raylight meta interface pattern.
    """
    
    def __init__(self, model, load_device, offload_device, size=0, 
                 weight_inplace_update=False):
        # CRITICAL: Initialize FSDP2 attributes BEFORE calling super().__init__()
        # because ModelPatcher.__init__() calls self.model_size() at line 234
        # which is overridden by this class and accesses these attributes
        self._fsdp_initialized = False
        self._executor = None
        self._forward_hook = None
        self._original_model_size = size  # Full model size
        self._checkpoint_path = None  # Will be set by sd.py
        
        # Now safe to call parent init (which will call our overridden model_size())
        super().__init__(model, load_device, offload_device, size, 
                        weight_inplace_update)
        
        logging.info(f"{LOG_PREFIX} Created FSDP2ModelPatcher (meta device interface)")
    
    def load(self, device, lowvram_model_memory=0, force_patch_weights=False, 
             full_load=False):
        """Override load() to initialize FSDP2 workers on first call.
        
        Flow:
        1. Check if model is on meta device
        2. If yes and not initialized, spawn FSDP2 workers
        3. Register forward hook for interception
        4. Continue normal ModelPatcher load flow
        """
        if not self._fsdp_initialized and self._is_meta_device():
            logging.info(f"{LOG_PREFIX} Initializing FSDP2 workers...")
            self._initialize_fsdp2(device)
            self._register_forward_hook()
            self._fsdp_initialized = True
        
        # Call parent load (handles device management)
        return super().load(device, lowvram_model_memory, force_patch_weights, full_load)
    
    def model_size(self, include_patches=False):
        """Report sharded model size after FSDP2 initialization.
        
        Before init: 0 bytes (meta device)
        After init: Per-GPU sharded size (e.g., 11GB/GPU)
        """
        if not self._fsdp_initialized:
            return 0  # Meta device has no memory
        
        # Get sharded size from workers
        if self._executor:
            try:
                result = self._executor.execute_collective("get_model_size", {})
                return result.get("size_bytes", self._original_model_size)
            except Exception as e:
                logging.error(f"{LOG_PREFIX} Failed to get model size from workers: {e}")
                return self._original_model_size
        
        return self._original_model_size
    
    def _is_meta_device(self):
        """Check if model is on meta device."""
        try:
            first_param = next(self.model.parameters())
            return first_param.device.type == 'meta'
        except StopIteration:
            return False
    
    def _initialize_fsdp2(self, device):
        """Spawn workers and initialize FSDP2 sharded model."""
        from comfy.parallel_attention.fsdp2_executor import FSDP2Executor
        
        world_size = 2  # TODO: Make configurable
        self._executor = FSDP2Executor(world_size=world_size)
        
        # Workers load checkpoint with FSDP2 sharding
        result = self._executor.execute_collective("initialize_fsdp2", {
            "model_structure": self._serialize_model_structure(),
            "checkpoint_path": self._checkpoint_path,
            "device": str(device),
        })
        
        if not result.get("success"):
            raise RuntimeError(f"FSDP2 initialization failed: {result.get('error')}")
        
        logging.info(f"{LOG_PREFIX} FSDP2 workers ready: {result.get('vram_gb', 0)}GB/GPU")
    
    def _register_forward_hook(self):
        """Register PyTorch hook to intercept forward pass."""
        def fsdp_forward_hook(module, args, kwargs):
            """Intercept forward pass and relay to FSDP2 workers."""
            if not self._fsdp_initialized:
                return None  # Let normal forward proceed
            
            # Send inputs to workers
            result = self._executor.execute_collective("forward", {
                "args": args,
                "kwargs": kwargs,
            })
            
            # Return output (hook will pass this through)
            return result.get("output")
        
        self._forward_hook = self.model.register_forward_pre_hook(
            fsdp_forward_hook, 
            with_kwargs=True
        )
        logging.info(f"{LOG_PREFIX} Forward hook registered")
    
    def _serialize_model_structure(self):
        """Serialize meta model structure for worker initialization."""
        # Extract module names and structure
        module_names = [name for name, _ in self.model.named_modules() if name]
        
        return {
            "module_names": module_names[:10],  # First 10 for debugging
            "model_class": self.model.__class__.__name__,
            "num_modules": len(module_names),
        }
    
    def unpatch_model(self, device_to=None, unpatch_weights=True):
        """Override to handle FSDP2 cleanup."""
        if self._forward_hook:
            self._forward_hook.remove()
            self._forward_hook = None
        
        if self._executor:
            self._executor.shutdown()
            self._executor = None
            self._fsdp_initialized = False
        
        return super().unpatch_model(device_to, unpatch_weights)
