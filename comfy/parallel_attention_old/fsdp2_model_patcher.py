"""FSDP2ModelPatcher - Extends ComfyUI ModelPatcher for distributed inference.

Integrates FSDP2 parameter sharding with ComfyUI's model lifecycle.

Based on Raylight comfy_dist/model_patcher.py, adapted for ComfyUI core.
"""

import comfy.model_patcher
import torch
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import set_model_state_dict, StateDictOptions
import logging

LOG_PREFIX = "⚡ [Parallel-Attention]"


class FSDP2ModelPatcher(comfy.model_patcher.ModelPatcher):
    """ModelPatcher extension with FSDP2 parameter sharding.
    
    Extends ComfyUI's ModelPatcher to support distributed inference
    via FSDP2 parameter sharding across multiple GPUs.
    
    Key features:
    - Lazy FSDP2 wrapping on first load()
    - Reports sharded memory size to ComfyUI scheduler
    - Sets comfy_cast_weights = True (tells ComfyUI "don't manage my device")
    - Compatible with LoRA, memory scheduler, existing nodes
    
    Based on Raylight's FSDPModelPatcher pattern, adapted for ComfyUI core.
    """
    
    def __init__(self, model, load_device, offload_device, 
                 size: int = 0,
                 weight_inplace_update: bool = False,
                 model_type: str = None):
        """Initialize FSDP2ModelPatcher.
        
        Args:
            model: Model instance (typically on meta device)
            load_device: Device to load model to (e.g., cuda:0)
            offload_device: Device to offload to when not in use
            size: Original model size in bytes (before sharding)
            weight_inplace_update: Whether to update weights in-place
            model_type: Model type for policy lookup (e.g., "flux", "wan")
        """
        # Set attributes BEFORE calling super().__init__ since parent calls model_size()
        self.model_type = model_type
        self.is_fsdp2_wrapped = False
        
        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        
        self.rank = rank
        self.world_size = world_size
        
        # Now call parent init
        super().__init__(model, load_device, offload_device, size, weight_inplace_update)
        
        # Store original size after parent init (parent may have calculated it)
        self.original_model_size = size if size > 0 else super().model_size()
        
        logging.info(
            f"{LOG_PREFIX} [FSDP2Patcher-{rank}] Created for model type: {model_type}, "
            f"original_size: {self.original_model_size / (1024**3):.2f}GB"
        )
    
    def load(self, *args, **kwargs):
        """Load model with lazy FSDP2 wrapping.
        
        Overrides ModelPatcher.load() to apply FSDP2 wrapping
        on first load. Subsequent loads skip wrapping.
        """
        if not self.is_fsdp2_wrapped:
            logging.info(f"{LOG_PREFIX} [FSDP2Patcher-{self.rank}] Applying FSDP2 wrapping...")
            self._apply_fsdp2_wrapping()
            self.is_fsdp2_wrapped = True
        
        return super().load(*args, **kwargs)
    
    def _apply_fsdp2_wrapping(self):
        """Apply FSDP2 wrapping using registry policy.
        
        1. Get policy from registry for model_type
        2. Apply fully_shard() per-block via policy
        3. Set comfy_cast_weights = True on all FSDP modules
        """
        from comfy.parallel_attention.fsdp2_policies import FSDP2PolicyRegistry
        
        if not FSDP2PolicyRegistry.is_registered(self.model_type):
            raise RuntimeError(
                f"No FSDP2 policy for model type '{self.model_type}'. "
                f"Available: {FSDP2PolicyRegistry.list_registered()}"
            )
        
        # Get policy and apply sharding
        policy_fn = FSDP2PolicyRegistry.get_policy(self.model_type)
        sharding_fn = policy_fn()
        
        # Policy modifies model in-place
        logging.info(f"{LOG_PREFIX} [FSDP2Patcher-{self.rank}] Applying {self.model_type} FSDP2 policy...")
        sharding_fn(self.model, state_dict=None)
        
        # Set comfy_cast_weights on all FSDP-wrapped modules
        self._set_comfy_cast_weights_flag()
        
        logging.info(f"{LOG_PREFIX} [FSDP2Patcher-{self.rank}] FSDP2 wrapping applied")
    
    def _set_comfy_cast_weights_flag(self):
        """Set comfy_cast_weights=True on FSDP-wrapped modules.
        
        This flag tells ComfyUI to not manage device placement for these modules.
        FSDP2 handles its own device management.
        """
        count = 0
        for module in self.model.modules():
            # Check if module is FSDP-wrapped by looking for _fsdp_wrapped_module attribute
            if hasattr(module, '_fsdp_wrapped_module'):
                module.comfy_cast_weights = True
                count += 1
        
        logging.info(
            f"{LOG_PREFIX} [FSDP2Patcher-{self.rank}] "
            f"Set comfy_cast_weights=True on {count} FSDP modules"
        )
    
    def model_memory_required(self, device):
        """Return sharded memory size for scheduler.
        
        Overrides ModelPatcher.model_memory_required() to report
        the sharded size (original / world_size) so ComfyUI's
        memory scheduler accounts for distributed sharding.
        
        Args:
            device: Device to check memory for
            
        Returns:
            Sharded model size in bytes
        """
        if self.is_fsdp2_wrapped and self.world_size > 1:
            sharded_size = self.original_model_size // self.world_size
            logging.debug(
                f"{LOG_PREFIX} [FSDP2Patcher-{self.rank}] "
                f"Reporting sharded size: {sharded_size / (1024**3):.2f}GB "
                f"(original: {self.original_model_size / (1024**3):.2f}GB)"
            )
            return sharded_size
        
        return super().model_memory_required(device)
    
    def model_size(self):
        """Return model size accounting for FSDP2 sharding.
        
        Override to report sharded size after FSDP2 wrapping.
        
        Args:
            include_lowvram_weight: Whether to include lowvram weights
            
        Returns:
            Model size in bytes (sharded if FSDP2 wrapped)
        """
        if self.is_fsdp2_wrapped and self.world_size > 1:
            return self.original_model_size // self.world_size
        
        return super().model_size()
