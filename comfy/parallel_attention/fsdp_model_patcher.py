"""FSDP-aware ModelPatcher extension for distributed model loading.

Extends ComfyUI's ModelPatcher to handle FSDP-wrapped models with
correct memory reporting, lazy wrapping, and distributed state dict loading.

Design Philosophy: EXTEND, Don't Bypass
- Inherits all ComfyUI ModelPatcher functionality
- Overrides only what's needed for FSDP compatibility
- Maintains LoRA, hooks, callbacks, injection system
- Transparent to existing ComfyUI code

Key Differences from Base ModelPatcher:
1. Lazy FSDP wrapping on first load()
2. Sharded memory reporting in model_memory_required()
3. Disabled clone() (FSDP incompatible with deepcopy)
4. Sets comfy_cast_weights=True to prevent interference

Based on Raylight's FSDPModelPatcher pattern, adapted to ComfyUI.
"""

from __future__ import annotations
import logging
from typing import Optional
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, MixedPrecision
from torch.distributed.fsdp.api import FullStateDictConfig, StateDictType

import comfy.model_patcher
import comfy.model_management

LOG_PREFIX = "⚡ [Parallel-Attention]"


class FSDPModelPatcher(comfy.model_patcher.ModelPatcher):
    """FSDP-aware extension of ComfyUI's ModelPatcher.
    
    Handles FSDP wrapping, sharded memory calculations, and distributed
    state dict loading while maintaining full compatibility with ComfyUI's
    model management system.
    
    Attributes:
        fsdp_config (dict): FSDP configuration (policy, cpu_offload, etc.)
        is_fsdp_wrapped (bool): Whether model has been wrapped with FSDP
        shard_factor (float): Memory reduction factor from sharding (e.g., 0.5 for 2 GPUs)
    
    Example:
        # Create FSDP patcher instead of regular patcher
        patcher = FSDPModelPatcher(
            model=unet,
            load_device=device,
            offload_device=offload_device,
            fsdp_config={
                'auto_wrap_policy': policy,
                'cpu_offload': True,
                'device_mesh': mesh
            }
        )
        
        # Use exactly like ModelPatcher
        patcher.patch_model(device_to=device)
        output = model.forward(...)
        patcher.unpatch_model()
    """
    
    def __init__(
        self,
        model,
        load_device,
        offload_device,
        size=0,
        weight_inplace_update=False,
        fsdp_config: Optional[dict] = None
    ):
        """Initialize FSDP-aware ModelPatcher.
        
        Args:
            model: PyTorch model to wrap
            load_device: Device to load model to (CUDA device)
            offload_device: Device to offload model to (CPU or CUDA)
            size: Model size in bytes (auto-calculated if 0)
            weight_inplace_update: Whether to update weights in-place
            fsdp_config: FSDP configuration dict with keys:
                - auto_wrap_policy: FSDP wrapping policy function
                - cpu_offload: Whether to offload parameters to CPU
                - device_mesh: Optional DeviceMesh for topology
                - sharding_strategy: FSDP ShardingStrategy (default: FULL_SHARD)
                - mixed_precision: FSDP MixedPrecision config (optional)
        """
        super().__init__(
            model=model,
            load_device=load_device,
            offload_device=offload_device,
            size=size,
            weight_inplace_update=weight_inplace_update
        )
        
        self.fsdp_config = fsdp_config or {}
        self.is_fsdp_wrapped = False
        
        # Calculate shard factor based on world size
        if dist.is_initialized():
            world_size = dist.get_world_size()
            self.shard_factor = 1.0 / world_size
        else:
            self.shard_factor = 1.0
            logging.warning(
                f"{LOG_PREFIX} [FSDPPatcher] torch.distributed not initialized, "
                f"shard_factor=1.0 (no sharding)"
            )
        
        logging.info(
            f"{LOG_PREFIX} [FSDPPatcher] Created FSDP patcher: "
            f"shard_factor={self.shard_factor:.2f}, "
            f"cpu_offload={self.fsdp_config.get('cpu_offload', False)}"
        )
        
        # WRAP IMMEDIATELY if state_dict provided
        # Don't wait for load() - that may never be called in workers
        if self.fsdp_config.get('state_dict') is not None:
            logging.info(f"{LOG_PREFIX} [FSDPPatcher] Wrapping with FSDP immediately (state dict provided)")
            self._wrap_with_fsdp()
    
    def _wrap_with_fsdp(self):
        """Wrap model with FSDP on first load() and load sharded weights.
        
        Applies FSDP wrapping using the configured auto_wrap_policy, then
        loads state dict using FSDP's distributed loading if provided.
        This is done lazily on first load() to ensure model is on correct
        device and properly initialized.
        
        Side Effects:
            - Wraps self.model with FSDP
            - Loads sharded weights if state_dict in fsdp_config
            - Sets self.is_fsdp_wrapped = True
            - Sets comfy_cast_weights flag on all modules
        """
        if self.is_fsdp_wrapped:
            return
        
        logging.info(f"{LOG_PREFIX} [FSDPPatcher] Wrapping model with FSDP...")
        
        # Get FSDP configuration
        auto_wrap_policy = self.fsdp_config.get('auto_wrap_policy', None)
        cpu_offload = self.fsdp_config.get('cpu_offload', False)
        sharding_strategy = self.fsdp_config.get(
            'sharding_strategy',
            ShardingStrategy.FULL_SHARD
        )
        mixed_precision = self.fsdp_config.get('mixed_precision', None)
        device_mesh = self.fsdp_config.get('device_mesh', None)
        state_dict = self.fsdp_config.get('state_dict', None)
        
        if auto_wrap_policy is None:
            logging.error(f"{LOG_PREFIX} [FSDPPatcher] No auto_wrap_policy provided!")
            raise ValueError("FSDP requires auto_wrap_policy in fsdp_config")
        
        # Configure CPU offload
        from torch.distributed.fsdp import CPUOffload
        cpu_offload_config = CPUOffload(offload_params=True) if cpu_offload else None
        
        # Build FSDP kwargs
        fsdp_kwargs = {
            'auto_wrap_policy': auto_wrap_policy,
            'sharding_strategy': sharding_strategy,
            'cpu_offload': cpu_offload_config,
            'device_id': self.load_device if self.load_device.type == 'cuda' else None,
            'sync_module_states': True,  # Sync params across ranks
            'use_orig_params': True,  # Maintain original parameter names
        }
        
        if mixed_precision is not None:
            fsdp_kwargs['mixed_precision'] = mixed_precision
        
        if device_mesh is not None:
            fsdp_kwargs['device_mesh'] = device_mesh
        
        # Wrap with FSDP FIRST (with empty model on CPU)
        # FSDP will handle moving shards to GPU during load_state_dict
        logging.info(f"{LOG_PREFIX} [FSDPPatcher] Wrapping empty model with FSDP...")
        self.model = FSDP(self.model, **fsdp_kwargs)
        self.is_fsdp_wrapped = True
        
        rank = dist.get_rank() if dist.is_initialized() else 0
        logging.info(f"{LOG_PREFIX} [FSDPPatcher] Rank {rank} FSDP wrapper created (model still empty)")
        
        # NOW load state dict - FSDP will scatter shards across ranks
        # Each rank only receives its shard (~11GB for 2 GPUs)
        if state_dict is not None:
            logging.info(f"{LOG_PREFIX} [FSDPPatcher] Loading sharded weights via FSDP scatter...")
            
            # Process state dict to remove 'diffusion_model.' prefix if present
            # KEEP TENSORS ON CPU - FSDP will move and shard them
            processed_sd = {}
            for k, v in state_dict.items():
                key = k
                if k.startswith('diffusion_model.'):
                    key = k[len('diffusion_model.'):]
                # Keep on CPU! FSDP will scatter to GPUs
                processed_sd[key] = v
            
            # Use FSDP's scatter-based loading
            # FSDP will:
            # 1. Take each parameter from CPU state dict
            # 2. Shard it across ranks
            # 3. Each rank only gets its shard on GPU
            with FSDP.state_dict_type(
                self.model,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(offload_to_cpu=False, rank0_only=False)
            ):
                missing, unexpected = self.model.load_state_dict(processed_sd, strict=False)
                
                if len(missing) > 0:
                    logging.warning(f"{LOG_PREFIX} [FSDPPatcher] Missing keys: {len(missing)} keys")
                if len(unexpected) > 0:
                    logging.warning(f"{LOG_PREFIX} [FSDPPatcher] Unexpected keys: {len(unexpected)} keys")
            
            allocated = torch.cuda.memory_allocated(self.load_device) / 1024**3
            logging.info(
                f"{LOG_PREFIX} [FSDPPatcher] Rank {rank} loaded shard: {allocated:.2f}GB VRAM"
            )
        
        # Set comfy_cast_weights=True on all modules to prevent ComfyUI interference
        # This is critical - prevents ComfyUI's lowvram system from interfering with FSDP
        for module in self.model.modules():
            if hasattr(module, 'comfy_cast_weights'):
                module.prev_comfy_cast_weights = module.comfy_cast_weights
            module.comfy_cast_weights = True
        
        logging.info(
            f"{LOG_PREFIX} [FSDPPatcher] FSDP initialization complete: "
            f"strategy={sharding_strategy}, cpu_offload={cpu_offload}, "
            f"shard_factor={self.shard_factor:.2f}"
        )
    
    def load(self, device_to=None, lowvram_model_memory=0, force_patch_weights=False, full_load=False):
        """Override load() to wrap with FSDP on first load.
        
        Args:
            device_to: Device to load model to
            lowvram_model_memory: Memory threshold for lowvram mode (ignored for FSDP)
            force_patch_weights: Whether to force weight patching
            full_load: Whether to fully load model (ignored for FSDP)
        
        Notes:
            - FSDP wrapping happens on first load
            - lowvram_model_memory ignored (FSDP handles memory)
            - Calls parent load() to handle patches, hooks, callbacks
        """
        # Wrap with FSDP on first load
        if not self.is_fsdp_wrapped:
            self._wrap_with_fsdp()
        
        # Call parent load() to handle all ComfyUI patching logic
        # This handles: patches, hooks, callbacks, weight loading, etc.
        super().load(
            device_to=device_to,
            lowvram_model_memory=lowvram_model_memory,
            force_patch_weights=force_patch_weights,
            full_load=full_load
        )
    
    def model_size(self):
        """Return full unsharded model size.
        
        Returns size of model before sharding. Used by ComfyUI for
        memory management decisions.
        
        Returns:
            int: Model size in bytes (unsharded)
        """
        if self.size > 0:
            return self.size
        
        # For FSDP, calculate full model size (not sharded size)
        # This is used by ComfyUI's scheduler to decide loading order
        self.size = comfy.model_management.module_size(self.model)
        return self.size
    
    def loaded_size(self):
        """Return actual loaded (sharded) model size.
        
        Returns memory footprint of model on this GPU, accounting for
        FSDP sharding. This is what actually impacts VRAM usage.
        
        Returns:
            int: Sharded model size in bytes
        """
        # Return sharded size for accurate VRAM tracking
        full_size = super().loaded_size()
        sharded_size = int(full_size * self.shard_factor)
        
        return sharded_size
    
    def memory_required(self, input_shape):
        """Return memory required for forward pass with sharded model.
        
        Accounts for:
        1. Sharded parameter size (reduced by shard_factor)
        2. Activation memory (NOT reduced - depends on input shape)
        3. FSDP all-gather overhead (temporary full parameter materialization)
        
        Args:
            input_shape: Input tensor shape for activation calculation
        
        Returns:
            int: Estimated memory required in bytes
        """
        # Get base memory calculation from parent
        base_memory = self.model.memory_required(input_shape=input_shape)
        
        # Adjust for sharding:
        # - Parameters: Reduced by shard_factor
        # - Activations: NOT reduced (full activations on each GPU)
        # - Overhead: FSDP all-gather temporarily doubles parameter memory
        
        # Simplified estimate: base_memory * shard_factor + overhead
        # Overhead is approximately the full parameter size (for all-gather)
        param_memory = self.model_size()
        activation_memory = base_memory - param_memory
        
        # Sharded parameter memory + activation memory + all-gather overhead
        sharded_param_memory = int(param_memory * self.shard_factor)
        overhead = sharded_param_memory  # All-gather temporarily doubles params
        
        total_memory = sharded_param_memory + activation_memory + overhead
        
        return total_memory
    
    def clone(self):
        """Clone is disabled for FSDP models.
        
        FSDP models cannot be deep-copied due to distributed communication
        state. Attempting to clone would break FSDP internal state.
        
        Raises:
            RuntimeError: Always (FSDP models cannot be cloned)
        """
        raise RuntimeError(
            f"{LOG_PREFIX} [FSDPPatcher] clone() is not supported for FSDP models. "
            "FSDP models cannot be deep-copied due to distributed state. "
            "Create a new FSDPModelPatcher instance instead."
        )
    
    def patch_model(self, device_to=None, lowvram_model_memory=0, load_weights=True, force_patch_weights=False):
        """Patch model with FSDP wrapping.
        
        Extends parent patch_model() to handle FSDP wrapping.
        All ComfyUI patching logic (LoRA, hooks, etc.) works as normal.
        
        Args:
            device_to: Device to patch model to
            lowvram_model_memory: Memory threshold (ignored for FSDP)
            load_weights: Whether to load weights
            force_patch_weights: Whether to force weight patching
        
        Returns:
            torch.nn.Module: FSDP-wrapped model
        """
        # Call parent patch_model - this handles all ComfyUI logic
        model = super().patch_model(
            device_to=device_to,
            lowvram_model_memory=lowvram_model_memory,
            load_weights=load_weights,
            force_patch_weights=force_patch_weights
        )
        
        return model
    
    def unpatch_model(self, device_to=None, unpatch_weights=True):
        """Unpatch model and optionally unwrap FSDP.
        
        Args:
            device_to: Device to unpatch model to
            unpatch_weights: Whether to unpatch weights
        
        Notes:
            - FSDP wrapper is NOT removed (state is preserved)
            - Parent unpatch logic handles weight restoration
        """
        # Call parent unpatch_model - handles all ComfyUI unpatching
        super().unpatch_model(device_to=device_to, unpatch_weights=unpatch_weights)
        
        # Note: We do NOT unwrap FSDP here
        # FSDP state should persist across patch/unpatch cycles
        # Unwrapping would require redistributing weights, which is expensive
