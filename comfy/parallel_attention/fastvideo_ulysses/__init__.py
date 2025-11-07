"""FastVideo-style pure Ulysses attention implementation.

Zero-dependency (except torch.distributed and flash-attn) bare-metal implementation
of Ulysses-style sequence parallelism for ComfyUI.
"""

from comfy.parallel_attention.fastvideo_ulysses.communicator import (
    initialize_sp_group,
    get_sp_group,
    get_sp_rank,
    get_sp_world_size,
    all_to_all_4d,
    all_gather_nd,
)

from comfy.parallel_attention.fastvideo_ulysses.attention import (
    UlyssesAttention,
)

__all__ = [
    "initialize_sp_group",
    "get_sp_group",
    "get_sp_rank",
    "get_sp_world_size",
    "all_to_all_4d",
    "all_gather_nd",
    "UlyssesAttention",
]
