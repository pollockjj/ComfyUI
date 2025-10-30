"""Parallel Attention - Distributed inference for ComfyUI.

FSDP2 parameter sharding + USP sequence parallelism for large models.
"""

from .executor import MultiprocExecutor
from .fsdp2_policies import FSDP2PolicyRegistry
from .fsdp2_model_patcher import FSDP2ModelPatcher
from .distributed_model_wrapper import DistributedModelWrapper
from .parallel_state import (
    initialize_parallel_state,
    get_device_mesh,
    get_sp_group,
    get_dp_group,
    get_sp_rank,
    get_dp_rank,
    get_sp_size,
    get_dp_size,
    is_initialized
)

__all__ = [
    'MultiprocExecutor',
    'FSDP2PolicyRegistry',
    'FSDP2ModelPatcher',
    'DistributedModelWrapper',
    'initialize_parallel_state',
    'get_device_mesh',
    'get_sp_group',
    'get_dp_group',
    'get_sp_rank',
    'get_dp_rank',
    'get_sp_size',
    'get_dp_size',
    'is_initialized'
]
