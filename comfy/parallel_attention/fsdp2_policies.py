"""FSDP2 wrapping policies for model-specific sharding strategies.

Registry pattern for model-specific FSDP2 parameter sharding.
Each policy returns configuration that defines how to shard a model.

Refactored in Phase 2.7 for core code reuse and extensibility.
"""

from typing import Callable, Dict
import logging

from comfy.parallel_attention.fsdp2_config import ShardingConfig, BlockConfig

LOG_PREFIX = "⚡ [Parallel-Attention]"


class FSDP2PolicyRegistry:
    """Registry for model-specific FSDP2 sharding policies.
    
    Stores and retrieves sharding configuration for different model architectures.
    Policies return ShardingConfig (config) not executables (functions).
    
    Refactored in Phase 2.7 for separation of concerns.
    """
    
    _policies: Dict[str, Callable] = {}
    
    @classmethod
    def register(cls, model_name: str):
        """Decorator to register an FSDP2 policy function.
        
        Args:
            model_name: Model identifier (e.g., "flux", "wan", "qwen_image")
            
        Returns:
            Decorator function
            
        Example:
            @FSDP2PolicyRegistry.register("flux")
            def flux_fsdp2_policy() -> ShardingConfig:
                return ShardingConfig(...)
        """
        def decorator(policy_fn: Callable) -> Callable:
            cls._policies[model_name] = policy_fn
            return policy_fn
        return decorator
    
    @classmethod
    def get_policy(cls, model_name: str) -> ShardingConfig:
        """Retrieve FSDP2 sharding configuration for a model.
        
        Args:
            model_name: Model identifier
            
        Returns:
            ShardingConfig instance
            
        Raises:
            ValueError: If model_name not registered
        """
        if model_name not in cls._policies:
            available = ", ".join(cls._policies.keys()) if cls._policies else "(none)"
            raise ValueError(
                f"No FSDP2 policy registered for '{model_name}'. "
                f"Available policies: {available}"
            )
        policy_fn = cls._policies[model_name]
        return policy_fn()  # Execute function, return ShardingConfig
    
    @classmethod
    def is_registered(cls, model_name: str) -> bool:
        """Check if a model has a registered policy.
        
        Args:
            model_name: Model identifier
            
        Returns:
            True if registered, False otherwise
        """
        return model_name in cls._policies
    
    @classmethod
    def list_registered(cls) -> list:
        """List all registered model names.
        
        Returns:
            List of registered model identifiers
        """
        return list(cls._policies.keys())


@FSDP2PolicyRegistry.register("flux")
def flux_fsdp2_policy() -> ShardingConfig:
    """Return FSDP2 sharding configuration for Flux model.
    
    Flux architecture:
    - double_blocks: 19 DoubleStreamBlock
    - single_blocks: 38 SingleStreamBlock
    - Other components: img_in, txt_in, time_in, vector_in, guidance_in, pe_embedder, final_layer
    
    Sharding strategy:
    - Shard: single_blocks.*, double_blocks.* (760 params)
    - Ignore: Everything else (20 params)
    
    Returns:
        ShardingConfig with block paths and shardable patterns
    """
    return ShardingConfig(
        model_name="flux",
        blocks=[
            BlockConfig(
                module_path="single_blocks",
                block_count=38,
                shard_each=True
            ),
            BlockConfig(
                module_path="double_blocks",
                block_count=19,
                shard_each=True
            ),
        ],
        shardable_param_patterns=[
            "single_blocks.",
            "double_blocks."
        ],
        root_wrap=True
    )


@FSDP2PolicyRegistry.register("wan")
def wan_fsdp2_policy() -> ShardingConfig:
    """Return FSDP2 sharding configuration for Wan model.
    
    Wan architecture:
    - blocks: 30 transformer blocks (Wan2.2)
    - Other components: patch_embed, pos_embed, final_layer
    
    Sharding strategy (FastVideo EXCLUSIVE):
    - Shard: blocks.0, blocks.1, ..., blocks.29 (numbered blocks only)
    - Ignore: Everything else (patch_embed, pos_embed, final_layer, any non-numbered blocks)
    
    FastVideo policy: lambda n, m: "blocks" in n and str.isdigit(n.split(".")[-1])
    
    Returns:
        ShardingConfig with block paths and shardable patterns
    """
    return ShardingConfig(
        model_name="wan",
        blocks=[
            BlockConfig(
                module_path="blocks",
                block_count=30,
                shard_each=True
            ),
        ],
        shardable_param_patterns=[
            "blocks.0.", "blocks.1.", "blocks.2.", "blocks.3.", "blocks.4.",
            "blocks.5.", "blocks.6.", "blocks.7.", "blocks.8.", "blocks.9.",
            "blocks.10.", "blocks.11.", "blocks.12.", "blocks.13.", "blocks.14.",
            "blocks.15.", "blocks.16.", "blocks.17.", "blocks.18.", "blocks.19.",
            "blocks.20.", "blocks.21.", "blocks.22.", "blocks.23.", "blocks.24.",
            "blocks.25.", "blocks.26.", "blocks.27.", "blocks.28.", "blocks.29."
        ],
        root_wrap=True
    )


@FSDP2PolicyRegistry.register("qwen_image")
def qwen_image_fsdp2_policy() -> ShardingConfig:
    """Return FSDP2 sharding configuration for Qwen Image model.
    
    Qwen Image architecture:
    - transformer_blocks: 60 blocks (default)
    - Other components: embeddings, final_layer
    
    Sharding strategy:
    - Shard: transformer_blocks.*
    - Ignore: Everything else (embeddings, final_layer)
    
    Returns:
        ShardingConfig with block paths and shardable patterns
    """
    return ShardingConfig(
        model_name="qwen_image",
        blocks=[
            BlockConfig(
                module_path="blocks",
                block_count=60,
                shard_each=True
            ),
        ],
        shardable_param_patterns=[
            "transformer_blocks."
        ],
        root_wrap=True
    )
