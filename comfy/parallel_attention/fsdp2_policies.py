"""FSDP2 wrapping policies for model-specific sharding strategies.

Registry pattern for model-specific FSDP2 parameter sharding.
Each policy defines how to shard a particular model architecture.

Based on Raylight diffusion_models/{flux,wan,qwen_image}/fsdp.py
"""

from torch.distributed.fsdp import fully_shard
from torch.distributed.fsdp.api import MixedPrecision
from typing import Callable, Dict
import logging

LOG_PREFIX = "⚡ [Parallel-Attention]"


def detect_dtype_mismatch(module, ref_dtype):
    """Detect parameters with dtype mismatch (e.g., FP8 weights).
    
    From Raylight: raylight/distributed_modules/utils.py
    
    Args:
        module: PyTorch module to check
        ref_dtype: Reference dtype (e.g., torch.float16)
        
    Returns:
        Set of parameters with mismatched dtype
    """
    ignored_param = set()
    for name, param in module.named_parameters(recurse=True):
        if param.dtype != ref_dtype:
            ignored_param.add(param)
    return ignored_param


class FSDP2PolicyRegistry:
    """Registry for model-specific FSDP2 wrapping policies.
    
    Stores and retrieves sharding strategies for different model architectures.
    Each policy is a function that applies fully_shard() to model components.
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
            def flux_fsdp2_policy():
                def shard_flux(model, state_dict):
                    # Apply sharding
                    pass
                return shard_flux
        """
        def decorator(policy_fn: Callable) -> Callable:
            cls._policies[model_name] = policy_fn
            logging.info(f"{LOG_PREFIX} [FSDP2Registry] Registered policy: {model_name}")
            return policy_fn
        return decorator
    
    @classmethod
    def get_policy(cls, model_name: str) -> Callable:
        """Retrieve a registered FSDP2 policy function.
        
        Args:
            model_name: Model identifier
            
        Returns:
            Policy function that returns sharding callable
            
        Raises:
            ValueError: If model_name not registered
        """
        if model_name not in cls._policies:
            available = ", ".join(cls._policies.keys()) if cls._policies else "(none)"
            raise ValueError(
                f"No FSDP2 policy registered for '{model_name}'. "
                f"Available policies: {available}"
            )
        return cls._policies[model_name]
    
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
def flux_fsdp2_policy() -> Callable:
    """Return FSDP2 sharding function for Flux model.
    
    Flux architecture:
    - double_blocks: List of DoubleStreamBlock (19 blocks)
    - single_blocks: List of SingleStreamBlock (38 blocks)
    - Other components: img_in, txt_in, time_in, final_layer, etc.
    
    Sharding strategy:
    - Shard each double_block independently
    - Shard each single_block independently
    - Ignore non-transformer components (embeddings, projections)
    
    Based on: raylight/diffusion_models/flux/fsdp.py
    
    Returns:
        Callable that takes (model, state_dict) and applies FSDP2 sharding
    """
    def shard_flux(model, state_dict):
        """Apply FSDP2 sharding to Flux model.
        
        Args:
            model: Flux model instance
            state_dict: Model state dict (for dtype detection)
            
        Returns:
            Model with FSDP2 sharding applied
        """
        diffusion_model = model.diffusion_model
        
        # Collect params to ignore (everything except single_blocks + double_blocks)
        ignored_params = set()
        for name, param in diffusion_model.named_parameters():
            if (not name.startswith("single_blocks.")) and (not name.startswith("double_blocks.")):
                ignored_params.add(param)
        
        logging.info(
            f"{LOG_PREFIX} [FSDP2-Flux] Ignoring {len(ignored_params)} non-transformer params "
            f"(img_in, txt_in, time_in, etc.)"
        )
        
        # Get reference dtype for mismatch detection (handles FP8 scaled models)
        ref_dtype = diffusion_model.double_blocks[0].img_attn.qkv.weight.dtype
        logging.info(f"{LOG_PREFIX} [FSDP2-Flux] Reference dtype: {ref_dtype}")
        
        # Shard single_blocks (38 blocks)
        logging.info(f"{LOG_PREFIX} [FSDP2-Flux] Sharding {len(diffusion_model.single_blocks)} single_blocks...")
        for i, block in enumerate(diffusion_model.single_blocks):
            # Detect dtype mismatches (e.g., FP8 weights in FP16 model)
            ignored_block_params = detect_dtype_mismatch(block, ref_dtype)
            
            diffusion_model.single_blocks[i] = fully_shard(
                module=block,
                mp_policy=MixedPrecision(),
                reshard_after_forward=True,
                ignored_params=ignored_block_params,
            )
        
        # Shard double_blocks (19 blocks)
        logging.info(f"{LOG_PREFIX} [FSDP2-Flux] Sharding {len(diffusion_model.double_blocks)} double_blocks...")
        for i, block in enumerate(diffusion_model.double_blocks):
            ignored_block_params = detect_dtype_mismatch(block, ref_dtype)
            
            diffusion_model.double_blocks[i] = fully_shard(
                module=block,
                mp_policy=MixedPrecision(),
                reshard_after_forward=True,
                ignored_params=ignored_block_params,
            )
        
        # Root wrap with ignored params
        logging.info(f"{LOG_PREFIX} [FSDP2-Flux] Applying root wrap with {len(ignored_params)} ignored params...")
        fully_shard(
            diffusion_model,
            ignored_params=ignored_params,
            mp_policy=MixedPrecision(),
            reshard_after_forward=True
        )
        
        model.diffusion_model = diffusion_model
        logging.info(f"{LOG_PREFIX} [FSDP2-Flux] Sharding complete (57 blocks wrapped)")
        
        return model
    
    return shard_flux


@FSDP2PolicyRegistry.register("wan")
def wan_fsdp2_policy() -> Callable:
    """Return FSDP2 sharding function for Wan model.
    
    Wan architecture:
    - blocks: List of transformer blocks (30 blocks for Wan2.2)
    - Other components: patch_embed, pos_embed, final_layer, etc.
    
    Sharding strategy:
    - Shard each transformer block independently
    - Ignore embeddings and final layer
    
    Based on: raylight/diffusion_models/wan/fsdp.py
    
    Returns:
        Callable that takes (model, state_dict) and applies FSDP2 sharding
    """
    def shard_wan(model, state_dict):
        """Apply FSDP2 sharding to Wan model.
        
        Args:
            model: Wan model instance
            state_dict: Model state dict (for dtype detection)
            
        Returns:
            Model with FSDP2 sharding applied
        """
        diffusion_model = model.diffusion_model
        
        # Collect ignored params (everything except blocks)
        ignored_params = set()
        for name, param in diffusion_model.named_parameters():
            if not name.startswith("blocks."):
                ignored_params.add(param)
        
        logging.info(
            f"{LOG_PREFIX} [FSDP2-Wan] Ignoring {len(ignored_params)} non-transformer params "
            f"(patch_embed, pos_embed, final_layer)"
        )
        
        # Get reference dtype
        ref_dtype = diffusion_model.blocks[0].self_attn.v.weight.dtype
        logging.info(f"{LOG_PREFIX} [FSDP2-Wan] Reference dtype: {ref_dtype}")
        
        # Shard blocks
        logging.info(f"{LOG_PREFIX} [FSDP2-Wan] Sharding {len(diffusion_model.blocks)} blocks...")
        for i, block in enumerate(diffusion_model.blocks):
            ignored_block_params = detect_dtype_mismatch(block, ref_dtype)
            
            diffusion_model.blocks[i] = fully_shard(
                module=block,
                mp_policy=MixedPrecision(),
                reshard_after_forward=True,
                ignored_params=ignored_block_params,
            )
        
        # Root wrap
        logging.info(f"{LOG_PREFIX} [FSDP2-Wan] Applying root wrap with {len(ignored_params)} ignored params...")
        fully_shard(
            diffusion_model,
            ignored_params=ignored_params,
            mp_policy=MixedPrecision(),
            reshard_after_forward=True
        )
        
        model.diffusion_model = diffusion_model
        logging.info(f"{LOG_PREFIX} [FSDP2-Wan] Sharding complete ({len(diffusion_model.blocks)} blocks wrapped)")
        
        return model
    
    return shard_wan


@FSDP2PolicyRegistry.register("qwen_image")
def qwen_image_fsdp2_policy() -> Callable:
    """Return FSDP2 sharding function for Qwen Image model.
    
    Qwen Image architecture:
    - transformer_blocks: List of QwenImageTransformerBlock (60 blocks default)
    - Other components: embeddings, final layer, etc.
    
    Sharding strategy:
    - Shard each transformer block independently
    - Ignore embeddings and final layer
    
    Based on: raylight/diffusion_models/qwen_image/fsdp.py
    
    Returns:
        Callable that takes (model, state_dict) and applies FSDP2 sharding
    """
    def shard_qwen(model, state_dict):
        """Apply FSDP2 sharding to Qwen Image model.
        
        Args:
            model: Qwen Image model instance
            state_dict: Model state dict (for dtype detection)
            
        Returns:
            Model with FSDP2 sharding applied
        """
        diffusion_model = model.diffusion_model
        
        # Collect ignored params (everything except transformer_blocks)
        ignored_params = set()
        for name, param in diffusion_model.named_parameters():
            if not name.startswith("transformer_blocks."):
                ignored_params.add(param)
        
        logging.info(
            f"{LOG_PREFIX} [FSDP2-Qwen] Ignoring {len(ignored_params)} non-transformer params"
        )
        
        # Get reference dtype
        ref_dtype = diffusion_model.transformer_blocks[0].attn.to_q.weight.dtype
        logging.info(f"{LOG_PREFIX} [FSDP2-Qwen] Reference dtype: {ref_dtype}")
        
        # Shard transformer blocks
        logging.info(f"{LOG_PREFIX} [FSDP2-Qwen] Sharding {len(diffusion_model.transformer_blocks)} transformer_blocks...")
        for i, block in enumerate(diffusion_model.transformer_blocks):
            ignored_block_params = detect_dtype_mismatch(block, ref_dtype)
            
            diffusion_model.transformer_blocks[i] = fully_shard(
                module=block,
                mp_policy=MixedPrecision(),
                reshard_after_forward=True,
                ignored_params=ignored_block_params,
            )
        
        # Root wrap
        logging.info(f"{LOG_PREFIX} [FSDP2-Qwen] Applying root wrap with {len(ignored_params)} ignored params...")
        fully_shard(
            diffusion_model,
            ignored_params=ignored_params,
            mp_policy=MixedPrecision(),
            reshard_after_forward=True
        )
        
        model.diffusion_model = diffusion_model
        logging.info(f"{LOG_PREFIX} [FSDP2-Qwen] Sharding complete ({len(diffusion_model.transformer_blocks)} blocks wrapped)")
        
        return model
    
    return shard_qwen
