"""TorchSP Ulysses communicator - pure PyTorch distributed primitives.

Provides sequence-parallel communication operations using torch.distributed.
Process group ownership follows BPS model: worker creates group, backend consumes it.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.distributed as dist
from torch import Tensor

LOGGER = logging.getLogger(__name__)
LOG_PREFIX = "⚡ [Parallel-Attention][TorchSP-Communicator]"

# Module-level state (set by worker during initialization)
_SP_GROUP: Optional[dist.ProcessGroup] = None
_SP_RANK: int = 0
_SP_WORLD_SIZE: int = 1


def initialize_torch_sp_group(ranks: list[int]) -> dist.ProcessGroup:
    """Initialize TorchSP sequence parallel process group.
    
    Called by worker during setup. Creates a new process group for the given ranks.
    
    Args:
        ranks: List of global ranks to include in the sequence parallel group.
               For 2-GPU setup with ulysses_degree=2, this is [0, 1].
    
    Returns:
        The created process group handle.
    
    Raises:
        RuntimeError: If group already initialized or distributed not initialized.
    """
    global _SP_GROUP, _SP_RANK, _SP_WORLD_SIZE
    
    if _SP_GROUP is not None:
        raise RuntimeError("TorchSP group already initialized")
    
    if not dist.is_initialized():
        raise RuntimeError("torch.distributed not initialized - worker must call init first")
    
    current_rank = dist.get_rank()
    
    # Create process group for sequence parallelism
    _SP_GROUP = dist.new_group(ranks=ranks, backend="nccl")
    
    # Determine local rank within this group
    if current_rank in ranks:
        _SP_RANK = ranks.index(current_rank)
        _SP_WORLD_SIZE = len(ranks)
    else:
        # Not in this group (shouldn't happen with current 2-GPU setup)
        _SP_RANK = 0
        _SP_WORLD_SIZE = 1
    
    LOGGER.info(
        "%s Initialized TorchSP group: ranks=%s, local_rank=%d, world_size=%d",
        LOG_PREFIX, ranks, _SP_RANK, _SP_WORLD_SIZE
    )
    
    return _SP_GROUP


def get_sp_group() -> Optional[dist.ProcessGroup]:
    """Get the sequence parallel process group (or None if not initialized)."""
    return _SP_GROUP


def get_sp_rank() -> int:
    """Get the sequence parallel rank within the group."""
    return _SP_RANK


def get_sp_world_size() -> int:
    """Get the sequence parallel world size."""
    return _SP_WORLD_SIZE


def all_to_all_4d(
    tensor: Tensor,
    scatter_dim: int,
    gather_dim: int,
) -> Tensor:
    """Execute all-to-all collective for 4D tensors.
    
    Scatters along scatter_dim and gathers along gather_dim.
    
    Args:
        tensor: Input tensor with shape [B, S, H, D]
        scatter_dim: Dimension to scatter (split and distribute chunks)
        gather_dim: Dimension to gather (concatenate received chunks)
    
    Returns:
        Transformed tensor with same rank but redistributed data.
    
    Example:
        Input:  [2, 1024, 16, 64] with scatter_dim=2, gather_dim=1, world_size=2
        Scatter: Split H=16 into 2 chunks of H=8
        All-to-all: Each rank gets H=8 from each peer
        Gather: Concatenate along S dimension
        Output: [2, 2048, 8, 64]
    """
    if _SP_GROUP is None:
        raise RuntimeError("TorchSP group not initialized - call initialize_torch_sp_group() first")
    
    world_size = _SP_WORLD_SIZE
    if world_size == 1:
        return tensor
    
    # Get input shape
    input_shape = list(tensor.shape)
    
    # Verify scatter dimension is divisible by world size
    if input_shape[scatter_dim] % world_size != 0:
        raise ValueError(
            f"Scatter dimension {scatter_dim} (size={input_shape[scatter_dim]}) "
            f"must be divisible by world_size={world_size}"
        )
    
    # Split along scatter dimension
    chunk_size = input_shape[scatter_dim] // world_size
    chunks = [chunk.contiguous() for chunk in torch.chunk(tensor, world_size, dim=scatter_dim)]
    
    # Prepare output buffers
    output_chunks = [torch.empty_like(chunks[0]) for _ in range(world_size)]
    
    # Execute all-to-all
    dist.all_to_all(output_chunks, chunks, group=_SP_GROUP)
    
    # Concatenate along gather dimension
    output = torch.cat(output_chunks, dim=gather_dim)
    
    LOGGER.debug(
        "%s all_to_all_4d: input_shape=%s, scatter_dim=%d, gather_dim=%d, output_shape=%s",
        LOG_PREFIX, input_shape, scatter_dim, gather_dim, list(output.shape)
    )
    
    return output


def all_gather_nd(
    tensor: Tensor,
    dim: int,
) -> Tensor:
    """Execute all-gather collective for N-dimensional tensors.
    
    Gathers tensors from all ranks and concatenates along the specified dimension.
    
    Args:
        tensor: Input tensor to gather
        dim: Dimension along which to concatenate gathered tensors
    
    Returns:
        Concatenated tensor with size multiplied by world_size along dim.
    
    Example:
        Input:  [2, 512, 8, 64] with dim=1, world_size=2
        Gather: Each rank collects tensors from all peers
        Concat: Concatenate along S dimension
        Output: [2, 1024, 8, 64]
    """
    if _SP_GROUP is None:
        raise RuntimeError("TorchSP group not initialized - call initialize_torch_sp_group() first")
    
    world_size = _SP_WORLD_SIZE
    if world_size == 1:
        return tensor
    
    # Prepare output buffers
    output_tensors = [torch.empty_like(tensor) for _ in range(world_size)]
    
    # Execute all-gather
    dist.all_gather(output_tensors, tensor, group=_SP_GROUP)
    
    # Concatenate along specified dimension
    output = torch.cat(output_tensors, dim=dim)
    
    LOGGER.debug(
        "%s all_gather_nd: input_shape=%s, dim=%d, output_shape=%s",
        LOG_PREFIX, list(tensor.shape), dim, list(output.shape)
    )
    
    return output
