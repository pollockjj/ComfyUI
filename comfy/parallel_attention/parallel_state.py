"""Parallel state management with DeviceMesh for FSDP2 and sequence parallel."""

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh, DeviceMesh
from typing import Optional
import logging

LOG_PREFIX = "⚡ [Parallel-Attention]"

# Global DeviceMesh instance
_DEVICE_MESH: Optional[DeviceMesh] = None

# Mesh dimension names
MESH_DIM_DP = "dp"  # Data parallel dimension
MESH_DIM_SP = "sp"  # Sequence parallel dimension

def initialize_parallel_state(sp_size: int, dp_size: int = 1):
    """Initialize DeviceMesh for hybrid parallelism.
    
    Creates a 2D DeviceMesh with dimensions (dp_size, sp_size):
    - DP (data parallel): Model replicas across DP groups
    - SP (sequence parallel): Sequence sharding within SP groups
    
    Args:
        sp_size: Sequence parallel group size (must divide world_size)
        dp_size: Data parallel group size (must divide world_size)
    
    Raises:
        RuntimeError: If torch.distributed not initialized
        ValueError: If sp_size * dp_size != world_size
        
    Example:
        # Initialize for world_size=4 with SP=2, DP=2
        initialize_parallel_state(sp_size=2, dp_size=2)
        
        # Workers organized as:
        # DP groups: [0,1] and [2,3]
        # SP groups: [0,2] and [1,3]
    """
    global _DEVICE_MESH
    
    if not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized before parallel_state")
    
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    
    if sp_size * dp_size != world_size:
        raise ValueError(
            f"sp_size ({sp_size}) * dp_size ({dp_size}) must equal world_size ({world_size})"
        )
    
    logging.info(f"{LOG_PREFIX} [ParallelState-{rank}] Initializing DeviceMesh: sp_size={sp_size}, dp_size={dp_size}")
    
    # Create 2D DeviceMesh: (dp_size, sp_size)
    # PyTorch handles all process group creation automatically
    _DEVICE_MESH = init_device_mesh(
        "cuda",
        (dp_size, sp_size),
        mesh_dim_names=(MESH_DIM_DP, MESH_DIM_SP)
    )
    
    # Log mesh topology for debugging
    logging.info(f"{LOG_PREFIX} [ParallelState-{rank}] DeviceMesh created: {_DEVICE_MESH}")
    logging.info(f"{LOG_PREFIX} [ParallelState-{rank}] Mesh shape: dp={dp_size}, sp={sp_size}")
    logging.info(f"{LOG_PREFIX} [ParallelState-{rank}] DP rank: {get_dp_rank()}, SP rank: {get_sp_rank()}")

def get_device_mesh() -> DeviceMesh:
    """Return the global DeviceMesh instance.
    
    Returns:
        DeviceMesh: 2D mesh with (dp, sp) dimensions
        
    Raises:
        RuntimeError: If parallel_state not initialized
        
    Example:
        mesh = get_device_mesh()
        fsdp_model = FSDP(model, device_mesh=mesh)
    """
    if _DEVICE_MESH is None:
        raise RuntimeError("parallel_state not initialized. Call initialize_parallel_state() first")
    return _DEVICE_MESH

def get_sp_group() -> dist.ProcessGroup:
    """Return the sequence parallel process group.
    
    Returns:
        ProcessGroup: Group for sequence parallel communication
        
    Raises:
        RuntimeError: If parallel_state not initialized
        
    Example:
        sp_group = get_sp_group()
        dist.all_gather(tensors, tensor, group=sp_group)
    """
    mesh = get_device_mesh()
    return mesh[MESH_DIM_SP].get_group()

def get_dp_group() -> dist.ProcessGroup:
    """Return the data parallel process group.
    
    Returns:
        ProcessGroup: Group for data parallel communication
        
    Raises:
        RuntimeError: If parallel_state not initialized
        
    Example:
        dp_group = get_dp_group()
        dist.all_reduce(gradients, group=dp_group)
    """
    mesh = get_device_mesh()
    return mesh[MESH_DIM_DP].get_group()

def get_sp_rank() -> int:
    """Return rank within sequence parallel group.
    
    Returns:
        int: Rank in SP group (0 to sp_size-1)
        
    Raises:
        RuntimeError: If parallel_state not initialized
        
    Example:
        if get_sp_rank() == 0:
            # First rank in SP group
            print("SP leader")
    """
    mesh = get_device_mesh()
    return mesh.get_local_rank(1)  # SP is dimension 1

def get_dp_rank() -> int:
    """Return rank within data parallel group.
    
    Returns:
        int: Rank in DP group (0 to dp_size-1)
        
    Raises:
        RuntimeError: If parallel_state not initialized
        
    Example:
        if get_dp_rank() == 0:
            # First rank in DP group
            print("DP leader")
    """
    mesh = get_device_mesh()
    return mesh.get_local_rank(0)  # DP is dimension 0

def get_sp_size() -> int:
    """Return sequence parallel group size.
    
    Returns:
        int: Number of ranks in SP group
        
    Raises:
        RuntimeError: If parallel_state not initialized
    """
    mesh = get_device_mesh()
    return mesh.size(1)  # SP is dimension 1

def get_dp_size() -> int:
    """Return data parallel group size.
    
    Returns:
        int: Number of ranks in DP group
        
    Raises:
        RuntimeError: If parallel_state not initialized
    """
    mesh = get_device_mesh()
    return mesh.size(0)  # DP is dimension 0

def is_initialized() -> bool:
    """Check if parallel state is initialized.
    
    Returns:
        bool: True if DeviceMesh initialized, False otherwise
    """
    return _DEVICE_MESH is not None

def get_mesh_coordinates() -> tuple:
    """Return current rank's coordinates in the mesh.
    
    Returns:
        tuple: (dp_rank, sp_rank)
        
    Raises:
        RuntimeError: If parallel_state not initialized
        
    Example:
        dp_rank, sp_rank = get_mesh_coordinates()
        print(f"Position: DP={dp_rank}, SP={sp_rank}")
    """
    return (get_dp_rank(), get_sp_rank())
