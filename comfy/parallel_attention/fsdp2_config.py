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
        ignored_param_patterns: Name patterns for params NOT to shard
        root_wrap: Whether to wrap root module after blocks
    
    Example:
        config = ShardingConfig(
            model_name="flux",
            blocks=[
                BlockConfig("diffusion_model.single_blocks", 38),
                BlockConfig("diffusion_model.double_blocks", 19),
            ],
            ignored_param_patterns=["img_in", "txt_in", "time_in", "final_layer"],
            root_wrap=True
        )
    """
    model_name: str
    blocks: List[BlockConfig]
    ignored_param_patterns: List[str]
    root_wrap: bool = True
    
    def get_ignored_params(self, model: nn.Module) -> Set[nn.Parameter]:
        """Collect parameters NOT in shard_prefixes (EXCLUSIVE logic).
        
        Raylight pattern: Ignore everything EXCEPT specified block prefixes.
        This catches model_sampling and all other non-transformer components.
        
        Args:
            model: Model to scan for parameters
            
        Returns:
            Set of parameters to exclude from sharding
            
        Example:
            # For Flux: shard_prefixes = ["single_blocks.", "double_blocks."]
            # Returns all params NOT starting with those prefixes
        """
        ignored = set()
        for name, param in model.named_parameters():
            # If name doesn't start with any shard prefix → ignore it
            is_shardable = any(name.startswith(prefix) for prefix in self.ignored_param_patterns)
            if not is_shardable:
                ignored.add(param)
        return ignored
