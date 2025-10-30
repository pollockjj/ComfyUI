"""Worker RPC handlers for distributed operations.

Each function is callable via MultiprocExecutor.execute_collective().
Functions run on all workers and return results to main process.
"""

import torch
import torch.distributed as dist
import logging

LOG_PREFIX = "⚡ [Parallel-Attention]"


def echo(message: str) -> str:
    """Echo message back (Phase 1.1 test).
    
    Args:
        message: Message to echo
        
    Returns:
        Same message
    """
    rank = dist.get_rank()
    logging.debug(f"{LOG_PREFIX} [Worker-{rank}] Echo: {message}")
    return message


def allreduce_test() -> int:
    """Test NCCL all_reduce collective (Phase 1.2 test).
    
    Each worker creates a tensor with its rank value.
    all_reduce sums across all workers.
    
    Returns:
        Sum of all ranks (0 + 1 + ... + world_size-1)
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    # Create tensor with rank value on GPU
    tensor = torch.tensor([rank], device=f'cuda:{rank}')
    
    logging.info(f"{LOG_PREFIX} [Worker-{rank}] Before all_reduce: {tensor.item()}")
    
    # All-reduce sums across all workers
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    
    result = tensor.item()
    logging.info(f"{LOG_PREFIX} [Worker-{rank}] After all_reduce: {result}")
    
    return result


def devicemesh_test() -> dict:
    """Test DeviceMesh topology and SP all_gather (Phase 1.3 test).
    
    Initializes DeviceMesh with (dp_size=1, sp_size=world_size).
    Tests all_gather collective on SP group.
    
    Returns:
        Dict with mesh info and gathered results
    """
    from comfy.parallel_attention.parallel_state import (
        initialize_parallel_state,
        get_device_mesh,
        get_sp_group,
        get_sp_rank,
        get_sp_size
    )
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    # Initialize parallel state with full SP (no DP)
    logging.info(f"{LOG_PREFIX} [Worker-{rank}] Initializing DeviceMesh...")
    initialize_parallel_state(sp_size=world_size, dp_size=1)
    
    mesh = get_device_mesh()
    sp_group = get_sp_group()
    sp_rank = get_sp_rank()
    sp_size = get_sp_size()
    
    logging.info(f"{LOG_PREFIX} [Worker-{rank}] Mesh initialized: sp_rank={sp_rank}/{sp_size}")
    
    # Test all_gather on SP group
    tensor = torch.tensor([float(rank)], device=f'cuda:{rank}')
    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    
    dist.all_gather(gathered, tensor, group=sp_group)
    
    gathered_values = [t.item() for t in gathered]
    logging.info(f"{LOG_PREFIX} [Worker-{rank}] Gathered: {gathered_values}")
    
    return {
        "mesh_shape": [mesh.size(0), mesh.size(1)],  # [dp_size, sp_size]
        "sp_rank": sp_rank,
        "sp_size": sp_size,
        "gathered": gathered_values
    }


def test_fsdp_policy(model_name: str) -> dict:
    """Test FSDP2 policy registry (Phase 2.1 test).
    
    Validates that:
    - Policy is registered for model_name
    - Policy function is callable
    - Returns correct structure
    
    Args:
        model_name: Model to test (e.g., "flux")
        
    Returns:
        Dict with validation results
    """
    from comfy.parallel_attention.fsdp2_policies import FSDP2PolicyRegistry
    
    rank = dist.get_rank()
    
    # Check if registered
    is_registered = FSDP2PolicyRegistry.is_registered(model_name)
    available_policies = FSDP2PolicyRegistry.list_registered()
    
    logging.info(f"{LOG_PREFIX} [Worker-{rank}] Testing policy: {model_name}")
    logging.info(f"{LOG_PREFIX} [Worker-{rank}] Available policies: {available_policies}")
    
    policy_callable = False
    policy_type = None
    
    if is_registered:
        try:
            policy_fn = FSDP2PolicyRegistry.get_policy(model_name)
            policy_callable = callable(policy_fn)
            policy_type = str(type(policy_fn))
            
            # Test that policy_fn returns a callable
            if policy_callable:
                sharding_fn = policy_fn()
                policy_callable = callable(sharding_fn)
                
            logging.info(f"{LOG_PREFIX} [Worker-{rank}] Policy '{model_name}' is callable: {policy_callable}")
        except Exception as e:
            logging.error(f"{LOG_PREFIX} [Worker-{rank}] Policy test error: {e}")
            policy_callable = False
            policy_type = str(type(e))
    
    return {
        "is_registered": is_registered,
        "policy_callable": policy_callable,
        "policy_type": policy_type,
        "available_policies": available_policies
    }


def test_fsdp2_modelpatcher() -> dict:
    """Test FSDP2ModelPatcher functionality (Phase 2.2 test).
    
    Creates a small test model and validates:
    - FSDP2ModelPatcher extends ModelPatcher
    - Can create instance
    - Tracks original model size
    - Reports sharded size correctly
    - Has proper attributes (rank, world_size, model_type)
    
    Returns:
        Dict with test results
    """
    from comfy.parallel_attention.fsdp2_model_patcher import FSDP2ModelPatcher
    import comfy.model_patcher
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    logging.info(f"{LOG_PREFIX} [Worker-{rank}] Testing FSDP2ModelPatcher functionality...")
    
    # Create small test model
    test_model = torch.nn.Sequential(
        torch.nn.Linear(100, 100),
        torch.nn.ReLU(),
        torch.nn.Linear(100, 10)
    )
    
    # Calculate model size
    model_size = sum(p.numel() * p.element_size() for p in test_model.parameters())
    logging.info(f"{LOG_PREFIX} [Worker-{rank}] Test model size: {model_size / 1024:.2f}KB")
    
    # Create FSDP2ModelPatcher
    patcher = FSDP2ModelPatcher(
        model=test_model,
        load_device=torch.device(f'cuda:{rank}'),
        offload_device=torch.device('cpu'),
        size=model_size,
        model_type='flux'
    )
    
    # Test 1: Extends ModelPatcher
    extends_modelpatcher = isinstance(patcher, comfy.model_patcher.ModelPatcher)
    logging.info(f"{LOG_PREFIX} [Worker-{rank}]   Extends ModelPatcher: {extends_modelpatcher}")
    
    # Test 2: Tracks original size
    has_original_size = hasattr(patcher, 'original_model_size')
    correct_size = patcher.original_model_size == model_size if has_original_size else False
    logging.info(f"{LOG_PREFIX} [Worker-{rank}]   Tracks original size: {correct_size}")
    
    # Test 3: Reports sharded size correctly
    if world_size > 1:
        # Simulate wrapped state
        patcher.is_fsdp2_wrapped = True
        sharded_size = patcher.model_memory_required(torch.device(f'cuda:{rank}'))
        expected_sharded = model_size // world_size
        correct_sharded_size = sharded_size == expected_sharded
        logging.info(f"{LOG_PREFIX} [Worker-{rank}]   Sharded size correct: {correct_sharded_size} ({sharded_size} == {expected_sharded})")
    else:
        correct_sharded_size = True
        logging.info(f"{LOG_PREFIX} [Worker-{rank}]   Sharded size: skipped (single GPU)")
    
    # Test 4: Has proper attributes
    has_model_type = hasattr(patcher, 'model_type') and patcher.model_type == 'flux'
    has_rank = hasattr(patcher, 'rank') and patcher.rank == rank
    has_world_size = hasattr(patcher, 'world_size') and patcher.world_size == world_size
    has_is_wrapped_flag = hasattr(patcher, 'is_fsdp2_wrapped')
    
    logging.info(f"{LOG_PREFIX} [Worker-{rank}]   Has attributes: type={has_model_type}, rank={has_rank}, ws={has_world_size}, flag={has_is_wrapped_flag}")
    
    all_passed = all([
        extends_modelpatcher,
        has_original_size,
        correct_size,
        correct_sharded_size,
        has_model_type,
        has_rank,
        has_world_size,
        has_is_wrapped_flag
    ])
    
    passed_count = sum([
        extends_modelpatcher,
        has_original_size,
        correct_size,
        correct_sharded_size,
        has_model_type,
        has_rank,
        has_world_size,
        has_is_wrapped_flag
    ])
    
    logging.info(f"{LOG_PREFIX} [Worker-{rank}] FSDP2ModelPatcher tests: {passed_count}/8 passed")
    
    return {
        "extends_modelpatcher": extends_modelpatcher,
        "has_original_size": has_original_size,
        "correct_size": correct_size,
        "correct_sharded_size": correct_sharded_size,
        "has_model_type": has_model_type,
        "has_rank": has_rank,
        "has_world_size": has_world_size,
        "has_is_wrapped_flag": has_is_wrapped_flag,
        "all_passed": all_passed,
        "passed_checks": f"{passed_count}/8"
    }


def load_fsdp2_model(checkpoint_path: str, model_config_dict: dict) -> dict:
    """Load model with FSDP2 sharding (Phase 2.2.1).
    
    Complete loading flow:
    1. Load state_dict via ComfyUI utils
    2. Strip diffusion_model prefix (Raylight pattern)
    3. Reconstruct model config from dict
    4. Create model on meta device
    5. Create FSDP2ModelPatcher
    6. Apply FSDP2 wrapping
    7. Load weights with DCP
    8. Measure VRAM
    
    Args:
        checkpoint_path: Path to .safetensors checkpoint
        model_config_dict: Serialized model config {class_name, unet_config}
        
    Returns:
        Dict with success status, VRAM metrics, key count
    """
    import comfy.utils
    import comfy.model_detection
    import comfy.supported_models
    from comfy.parallel_attention.fsdp2_model_patcher import FSDP2ModelPatcher
    from comfy.parallel_attention.parallel_state import get_device_mesh
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    # DIAGNOSTIC: Log what we received
    logging.info(f"{LOG_PREFIX} [Worker-{rank}] === load_fsdp2_model CALLED ===")
    logging.info(f"{LOG_PREFIX} [Worker-{rank}]   checkpoint_path type: {type(checkpoint_path)}")
    logging.info(f"{LOG_PREFIX} [Worker-{rank}]   checkpoint_path value: {checkpoint_path}")
    logging.info(f"{LOG_PREFIX} [Worker-{rank}]   model_config_dict type: {type(model_config_dict)}")
    if isinstance(model_config_dict, dict):
        logging.info(f"{LOG_PREFIX} [Worker-{rank}]   model_config_dict keys: {list(model_config_dict.keys())}")
        for key, value in model_config_dict.items():
            logging.info(f"{LOG_PREFIX} [Worker-{rank}]     {key}: {type(value).__name__}")
    else:
        logging.error(f"{LOG_PREFIX} [Worker-{rank}]   ❌ model_config_dict is NOT a dict!")
    
    logging.info(f"{LOG_PREFIX} [Worker-{rank}] Loading FSDP2 model from {checkpoint_path}")
    
    # 1. Load state_dict (ComfyUI standard)
    state_dict = comfy.utils.load_torch_file(checkpoint_path)
    logging.info(f"{LOG_PREFIX} [Worker-{rank}] State dict loaded: {len(state_dict)} keys")
    
    # 2. Detect prefix but DON'T strip yet
    # We need the prefix for DCP loading
    diffusion_model_prefix = comfy.model_detection.unet_prefix_from_state_dict(state_dict)
    logging.info(f"{LOG_PREFIX} [Worker-{rank}] Detected prefix: {diffusion_model_prefix}")
    
    # 3. Reconstruct model config from serialized dict
    class_name = model_config_dict['class_name']
    unet_config = model_config_dict['unet_config']
    
    # Find model class in ComfyUI's supported models
    model_config = None
    for model_class in comfy.supported_models.models:
        if model_class.__name__ == class_name:
            model_config = model_class(unet_config)
            break
    
    if model_config is None:
        raise RuntimeError(f"Unknown model config class: {class_name}")
    
    logging.info(f"{LOG_PREFIX} [Worker-{rank}] Model config: {class_name}")
    
    # 4. Create model on meta device
    # Meta device = 0 bytes, structure only
    # Pass prefix to get_model so it creates the right structure
    logging.info(f"{LOG_PREFIX} [Worker-{rank}] Creating model on meta device...")
    with torch.device('meta'):
        model = model_config.get_model(state_dict, diffusion_model_prefix)
    
    # Calculate model size from state dict
    model_size = sum(v.numel() * v.element_size() for v in state_dict.values())
    logging.info(f"{LOG_PREFIX} [Worker-{rank}] Model size: {model_size / (1024**3):.2f}GB")
    
    # 5. Create FSDP2ModelPatcher
    # Ensure parallel state initialized
    from comfy.parallel_attention.parallel_state import is_initialized, initialize_parallel_state
    if not is_initialized():
        # Initialize with full sequence parallelism (no data parallel)
        initialize_parallel_state(sp_size=world_size, dp_size=1)
    
    device_mesh = get_device_mesh()
    load_device = torch.device(f'cuda:{rank}')
    offload_device = torch.device('cpu')
    
    # Detect model type for policy lookup
    model_type = _detect_model_type(class_name)
    
    patcher = FSDP2ModelPatcher(
        model=model,
        load_device=load_device,
        offload_device=offload_device,
        size=model_size,
        model_type=model_type
    )
    
    logging.info(f"{LOG_PREFIX} [Worker-{rank}] FSDP2ModelPatcher created")
    
    # 6. Apply FSDP2 wrapping
    # This wraps each block with fully_shard() per policy
    logging.info(f"{LOG_PREFIX} [Worker-{rank}] Applying FSDP2 wrapping...")
    patcher._apply_fsdp2_wrapping()
    patcher.is_fsdp2_wrapped = True
    
    # 7. Load weights with DCP
    # DCP handles distributed loading efficiently
    logging.info(f"{LOG_PREFIX} [Worker-{rank}] Loading weights via DCP...")
    
    # Measure VRAM before loading
    torch.cuda.reset_peak_memory_stats(rank)
    vram_before_gb = torch.cuda.memory_allocated(rank) / (1024**3)
    
    # Use model.load_state_dict instead of DCP set_model_state_dict
    # The model already has the correct structure with prefix
    patcher.model.load_state_dict(state_dict, strict=True)
    
    logging.info(f"{LOG_PREFIX} [Worker-{rank}] Weights loaded successfully")
    
    # 8. Measure VRAM after loading
    torch.cuda.synchronize(rank)
    vram_allocated_gb = torch.cuda.memory_allocated(rank) / (1024**3)
    vram_reserved_gb = torch.cuda.memory_reserved(rank) / (1024**3)
    vram_peak_gb = torch.cuda.max_memory_allocated(rank) / (1024**3)
    
    sharded_size_gb = patcher.model_size() / (1024**3)
    original_size_gb = patcher.original_model_size / (1024**3)
    
    logging.info(f"{LOG_PREFIX} [Worker-{rank}] Model loaded successfully")
    logging.info(f"{LOG_PREFIX} [Worker-{rank}]   VRAM allocated: {vram_allocated_gb:.2f}GB")
    logging.info(f"{LOG_PREFIX} [Worker-{rank}]   VRAM reserved: {vram_reserved_gb:.2f}GB")
    logging.info(f"{LOG_PREFIX} [Worker-{rank}]   VRAM peak: {vram_peak_gb:.2f}GB")
    logging.info(f"{LOG_PREFIX} [Worker-{rank}]   Sharded size: {sharded_size_gb:.2f}GB")
    logging.info(f"{LOG_PREFIX} [Worker-{rank}]   Original size: {original_size_gb:.2f}GB")
    
    return {
        "success": True,
        "vram_before_gb": vram_before_gb,
        "vram_allocated_gb": vram_allocated_gb,
        "vram_reserved_gb": vram_reserved_gb,
        "vram_peak_gb": vram_peak_gb,
        "sharded_size_gb": sharded_size_gb,
        "original_size_gb": original_size_gb,
        "num_keys": len(state_dict),
        "rank": rank,
        "world_size": world_size
    }


def _detect_model_type(class_name: str) -> str:
    """Detect model type from class name for FSDP2 policy lookup.
    
    Args:
        class_name: Model config class name (e.g., "FLUX", "Wan2", "QwenImage")
        
    Returns:
        Model type string for policy registry ("flux", "wan", "qwen_image")
    """
    class_name_lower = class_name.lower()
    
    if 'flux' in class_name_lower:
        return 'flux'
    elif 'wan' in class_name_lower:
        return 'wan'
    elif 'qwen' in class_name_lower:
        return 'qwen_image'
    else:
        raise RuntimeError(
            f"Unknown model type: {class_name}. "
            f"Supported: flux, wan, qwen_image"
        )
