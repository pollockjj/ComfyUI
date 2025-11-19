"""Core FSDP2 sharding engine - model-agnostic execution.

Applies FSDP2 sharding based on ShardingConfig.
Reusable across future distributed strategies (FSDP2-only today).

Based on Phase 2.7 refactor for separation of concerns and extensibility.
"""

import torch
import torch.nn as nn
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.checkpoint.state_dict import set_model_state_dict, StateDictOptions
import logging

from comfy.parallel_attention.fsdp2_config import ShardingConfig, BlockConfig
from comfy.parallel_attention.fsdp2_utils import detect_unshardable_params, get_reference_dtype

LOG_PREFIX = "⚡ [Parallel-Attention]"


def get_module_by_path(model: nn.Module, path: str) -> nn.Module:
    """Get nested module by dot-separated path.
    
    Args:
        model: Root model
        path: Dot-separated path (e.g., "diffusion_model.single_blocks")
        
    Returns:
        Module at path
        
    Raises:
        AttributeError: If path not found
        
    Example:
        blocks = get_module_by_path(model, "diffusion_model.single_blocks")
    """
    parts = path.split(".")
    current = model
    for part in parts:
        current = getattr(current, part)
    return current


def apply_fsdp2_sharding(
    meta_model: nn.Module,
    config: ShardingConfig,
    state_dict: dict = None,
    device_mesh = None
) -> nn.Module:
    """Apply FSDP2 sharding to meta model and optionally load state_dict.
    
    Core FSDP2 engine - model-agnostic, reusable across parallelism strategies.
    
    Flow:
    1. Collect ignored params from config patterns
    2. Shard blocks according to config (each block or whole list)
    3. Root wrap diffusion_model with ignored params
    4. Load state_dict with DCP (only if provided)
    
    Args:
        meta_model: Model on meta device (0 bytes, structure only)
        config: Sharding configuration from policy
        state_dict: Optional state dict to load (extracted from parent model)
        device_mesh: DeviceMesh for FSDP2 sharding (REQUIRED for multi-GPU)
        
    Returns:
        Model with FSDP2 sharding applied and weights loaded (if checkpoint provided)
        
    Example:
        from comfy.parallel_attention.fsdp2_policies import FSDP2PolicyRegistry
        config = FSDP2PolicyRegistry.get_policy("flux")
        model = apply_fsdp2_sharding(meta_model, config, device_mesh=device_mesh)
    """
    # Log DeviceMesh configuration
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
    mesh_info = f"mesh_shape={tuple(device_mesh.shape)}, mesh_dim_names={list(device_mesh.mesh_dim_names)}" if device_mesh else "None"
    
    if rank == 0:
        logging.info(f"{LOG_PREFIX} [FSDP2Engine] Applying sharding for: {config.model_name}")
        logging.info(f"{LOG_PREFIX} [FSDP2Engine]   Rank: {rank}, Device: {device}")
        logging.info(f"{LOG_PREFIX} [FSDP2Engine]   DeviceMesh: {mesh_info}")
    else:
        logging.debug(f"{LOG_PREFIX} [FSDP2Engine] Rank {rank} applying sharding")
    
    # Step 1: Detect reference dtype for universal dtype mismatch detection
    ref_dtype = None
    for block_config in config.blocks:
        try:
            block_list = get_module_by_path(meta_model, block_config.module_path)
            if len(block_list) > 0:
                ref_dtype = get_reference_dtype(block_list[0])
                if ref_dtype:
                    break
        except:
            continue
    
    if ref_dtype:
        if rank == 0:
            logging.info(f"{LOG_PREFIX} [FSDP2Engine] Reference dtype: {ref_dtype}")
    

    diffusion_model = meta_model.diffusion_model
    
    # Collect ignored params using policy's EXCLUSIVE logic
    # This gets everything NOT in shardable_param_patterns within diffusion_model
    ignored_params = config.get_ignored_params(diffusion_model)
    
    # Add universal unshardable params (scalars, dtype mismatches)
    universal_ignored = detect_unshardable_params(diffusion_model, ref_dtype)
    ignored_params.update(universal_ignored)
    
    if rank == 0:
        logging.info(
            f"{LOG_PREFIX} [FSDP2Engine] Excluding {len(ignored_params)} params from sharding "
            f"(shardable patterns: {', '.join(config.shardable_param_patterns)})"
        )
    
    # Shard blocks according to config
    total_blocks_sharded = 0
    for block_config in config.blocks:
        block_list = get_module_by_path(diffusion_model, block_config.module_path)
        
        if not isinstance(block_list, (nn.ModuleList, list)):
            raise TypeError(
                f"Expected ModuleList at {block_config.module_path}, "
                f"got {type(block_list)}"
            )
        
        if rank == 0:
            logging.info(
                f"{LOG_PREFIX} [FSDP2Engine] Sharding {len(block_list)} blocks at {block_config.module_path} "
                f"(mesh: {mesh_info}, device: {device})"
            )
        
        if block_config.shard_each:
            # Shard each block independently
            for i in range(len(block_list)):
                # Detect unshardable params (scalars + dtype mismatches)
                ignored_block_params = detect_unshardable_params(block_list[i], ref_dtype)
                
                block_list[i] = fully_shard(
                    module=block_list[i],
                    mp_policy=MixedPrecisionPolicy(),
                    reshard_after_forward=True,
                    mesh=device_mesh,
                    ignored_params=ignored_block_params
                )
            total_blocks_sharded += len(block_list)
        else:
            # Shard entire list as one unit
            parent_module = get_module_by_path(
                meta_model, 
                ".".join(block_config.module_path.split(".")[:-1])
            )
            attr_name = block_config.module_path.split(".")[-1]
            setattr(
                parent_module,
                attr_name,
                fully_shard(
                    module=block_list,
                    mp_policy=MixedPrecisionPolicy(),
                    reshard_after_forward=True,
                    mesh=device_mesh
                )
            )
            total_blocks_sharded += 1
    
    if rank == 0:
        logging.info(f"{LOG_PREFIX} [FSDP2Engine] Sharded {total_blocks_sharded} blocks")
    

    if config.root_wrap:
        # Add universal scalar detection to root ignored params
        from comfy.parallel_attention.fsdp2_utils import detect_scalar_params
        root_scalars = detect_scalar_params(diffusion_model)
        ignored_params.update(root_scalars)
        
        if rank == 0:
            logging.debug(
                f"{LOG_PREFIX} [FSDP2Engine] Applying root wrap with "
                f"{len(ignored_params)} ignored params"
            )
        fully_shard(
            diffusion_model,
            ignored_params=ignored_params,
            mp_policy=MixedPrecisionPolicy(),
            reshard_after_forward=True,
            mesh=device_mesh
        )
    
    meta_model.diffusion_model = diffusion_model
    
    # Load state_dict if provided
    if state_dict:
        if rank == 0:
            logging.debug(f"{LOG_PREFIX} [FSDP2Engine] Loading state_dict...")
        
        set_model_state_dict(
            model=meta_model,
            model_state_dict=state_dict,
            options=StateDictOptions(
                full_state_dict=True,
                broadcast_from_rank0=True,  # Only rank 0 loads, broadcasts shards
                cpu_offload=False  # Keep on GPU (hardcoded per requirements)
            )
        )
        
        # Measure VRAM after loading
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        if torch.cuda.is_available():
            vram_gb = torch.cuda.memory_allocated(rank) / (1024**3)
            logging.info(f"{LOG_PREFIX} [FSDP2Engine] Rank {rank} VRAM: {vram_gb:.2f}GB")
    
    logging.info(f"{LOG_PREFIX} [FSDP2Engine] Sharding complete for {config.model_name}")
    return meta_model


def apply_fsdp2_sharding_structure_only(
    meta_model: nn.Module,
    config: ShardingConfig,
    device_mesh
) -> nn.Module:
    """Apply FSDP2 sharding to meta model structure (no weight loading).
    
    Used with FastVideo iterator pattern where weights are loaded
    one tensor at a time after FSDP wrapping is applied.
    
    Args:
        meta_model: Model on meta device (0 bytes, structure only)
        config: Sharding configuration from policy
        device_mesh: DeviceMesh for FSDP2 sharding (REQUIRED)
        
    Returns:
        Model with FSDP2 sharding applied (no weights loaded yet)
    """
    return apply_fsdp2_sharding(meta_model, config, state_dict=None, device_mesh=device_mesh)
