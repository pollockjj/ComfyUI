"""Minimal distributed communication layer for Ulysses attention.

Pure torch.distributed implementation - no GroupCoordinator, no vLLM abstractions.
"""

import logging
import torch
import torch.distributed as dist
from typing import Optional

LOG_PREFIX = "⚡ [FastVideo-Ulysses][Communicator]"
logger = logging.getLogger(__name__)

# Global state
_SP_GROUP: Optional[dist.ProcessGroup] = None


def initialize_sp_group(sp_degree: int) -> None:
    """Initialize sequence parallel process group.
    
    Args:
        sp_degree: Number of ranks in the sequence parallel group (e.g., 2 for 2 GPUs)
    """
    global _SP_GROUP
    
    if _SP_GROUP is not None:
        logger.warning(f"{LOG_PREFIX} SP group already initialized, skipping")
        return
    
    if not dist.is_initialized():
        raise RuntimeError(f"{LOG_PREFIX} torch.distributed not initialized")
    
    world_size = dist.get_world_size()
    
    if sp_degree > world_size:
        raise ValueError(
            f"{LOG_PREFIX} sp_degree ({sp_degree}) cannot exceed world_size ({world_size})"
        )
    
    # Create SP group with consecutive ranks [0, 1, ..., sp_degree-1]
    ranks = list(range(sp_degree))
    _SP_GROUP = dist.new_group(ranks)
    
    rank = dist.get_rank()
    if rank in ranks:
        logger.info(
            f"{LOG_PREFIX} Initialized SP group: ranks={ranks}, "
            f"my_rank={rank}, sp_world_size={sp_degree}"
        )


def get_sp_group() -> dist.ProcessGroup:
    """Get the sequence parallel process group."""
    if _SP_GROUP is None:
        raise RuntimeError(f"{LOG_PREFIX} SP group not initialized. Call initialize_sp_group() first")
    return _SP_GROUP


def get_sp_rank() -> int:
    """Get rank within the SP group."""
    return dist.get_rank(get_sp_group())


def get_sp_world_size() -> int:
    """Get world size of the SP group."""
    return dist.get_world_size(get_sp_group())


def all_to_all_4d(
    tensor: torch.Tensor,
    scatter_dim: int,
    gather_dim: int,
) -> torch.Tensor:
    """Perform all-to-all communication on a 4D tensor.
    
    This is the core of Ulysses attention: redistribute heads <-> sequence.
    
    Args:
        tensor: 4D tensor [batch, seq, heads, dim] or [batch, heads, seq, dim]
        scatter_dim: Dimension to scatter (split across ranks)
        gather_dim: Dimension to gather (concat from ranks)
    
    Returns:
        Redistributed tensor with scatter_dim shrunk and gather_dim grown
    """
    group = get_sp_group()
    world_size = get_sp_world_size()
    
    if world_size == 1:
        return tensor
    
    # Split tensor along scatter_dim
    input_list = list(torch.chunk(tensor, world_size, dim=scatter_dim))
    
    # Prepare output tensors
    output_list = [torch.empty_like(input_list[0]) for _ in range(world_size)]
    
    # Perform all-to-all
    dist.all_to_all(output_list, input_list, group=group)
    
    # Concatenate along gather_dim
    output = torch.cat(output_list, dim=gather_dim)
    
    return output


def all_gather_nd(tensor: torch.Tensor, dim: int) -> torch.Tensor:
    """All-gather a tensor along a specific dimension.
    
    Used for gathering replicated text tokens across heads.
    
    Args:
        tensor: Input tensor
        dim: Dimension to gather along
    
    Returns:
        Gathered tensor
    """
    group = get_sp_group()
    world_size = get_sp_world_size()
    
    if world_size == 1:
        return tensor
    
    # Prepare output list
    output_list = [torch.empty_like(tensor) for _ in range(world_size)]
    
    # Perform all-gather
    dist.all_gather(output_list, tensor, group=group)
    
    # Concatenate along specified dimension
    output = torch.cat(output_list, dim=dim)
    
    return output
