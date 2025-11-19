"""FSDP2 sharding configuration data structures.

Separates policy configuration from execution logic.
Enables core code reuse across FSDP2, USP, CFG parallelism strategies.

Based on Phase 2.7 refactor plan for extensibility.
"""

from dataclasses import dataclass
from typing import Set, List
import torch.nn as nn


@dataclass
class BlockConfig:
    """Configuration for a single shardable block group.
    
    Attributes:
        module_path: Dot-path to block list (e.g., "diffusion_model.single_blocks")
        block_count: Number of blocks in list
        shard_each: Whether to shard each block independently (True) or as one unit (False)
    
    Example:
        BlockConfig("diffusion_model.single_blocks", 38, shard_each=True)
    """
    module_path: str
    block_count: int
    shard_each: bool = True


@dataclass
class ShardingConfig:
    """Complete FSDP2 sharding configuration for a model.
    
    Defines which blocks to shard and which parameters to ignore.
    Separates configuration from execution - policies return this,
    engine executes based on it.
    
    Attributes:
        model_name: Model identifier ("flux", "wan", "qwen_image")
        blocks: List of block configurations to shard
        shardable_param_patterns: Param name prefixes that WILL be sharded (EXCLUSIVE logic)
        root_wrap: Whether to wrap root module after blocks
    
    Example:
        config = ShardingConfig(
            model_name="flux",
            blocks=[
                BlockConfig("diffusion_model.single_blocks", 38),
                BlockConfig("diffusion_model.double_blocks", 19),
            ],
            shardable_param_patterns=["single_blocks.", "double_blocks."],
            root_wrap=True
        )
    """
    model_name: str
    blocks: List[BlockConfig]
    shardable_param_patterns: List[str]
    root_wrap: bool = True
    
    def get_ignored_params(self, diffusion_model: nn.Module) -> Set[nn.Parameter]:
        """Collect parameters to exclude from sharding (EXCLUSIVE logic).
        
        Iterates over diffusion_model.named_parameters() and ignores everything
        EXCEPT params matching shardable_param_patterns.
        
        Args:
            diffusion_model: Inner model (transformer only, no wrapper)
            
        Returns:
            Set of parameters to exclude from FSDP2 sharding
        """
        ignored = set()
        
        for name, param in diffusion_model.named_parameters():
            # If name does NOT start with any shardable pattern → ignore it
            is_shardable = any(name.startswith(prefix) for prefix in self.shardable_param_patterns)
            if not is_shardable:
                ignored.add(param)
        
        return ignored
