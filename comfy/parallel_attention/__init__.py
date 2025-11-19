"""Parallel Attention - Core Infrastructure

Exports:
- FSDP2Executor (DeviceMesh-based multiprocess executor)
- parallel_state (DeviceMesh management)
- FSDP2PolicyRegistry (policy system)
- ShardingConfig/BlockConfig (data structures)
"""

from .fsdp2_executor import FSDP2Executor
from .fsdp2_policies import FSDP2PolicyRegistry
from .fsdp2_config import ShardingConfig, BlockConfig
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
    'FSDP2Executor',
    'FSDP2PolicyRegistry',
    'ShardingConfig',
    'BlockConfig',
    'initialize_parallel_state',
    'get_device_mesh',
    'get_sp_group',
    'get_dp_group',
    'get_sp_rank',
    'get_dp_rank',
    'get_sp_size',
    'get_dp_size',
    'is_initialized',
]
