"""Parallel Attention configuration and control nodes."""

import torch
import logging

LOG_PREFIX = "⚡ [Parallel-Attention]"


class ParallelAttentionConfig:
    """Configure parallel attention options for a model.
    
    Requires --use-parallel-attention CLI flag to be set.
    Extends base config attached by comfy/sd.py.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
            }
        }
    
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "configure"
    CATEGORY = "parallel_attention"
    
    def configure(self, model):
        """Set parallel config hardware parameters."""
        if not hasattr(model.model, 'parallel_config'):
            logging.warning(f"{LOG_PREFIX} Config node skipped: --use-parallel-attention flag not set")
            logging.warning(f"{LOG_PREFIX} Start ComfyUI with: python main.py --use-parallel-attention")
            return (model,)
        
        if not model.model.parallel_config.get('enabled'):
            logging.warning(f"{LOG_PREFIX} Config disabled")
            return (model,)
        
        # Set hardware parameters (later: expose as node inputs)
        model.model.parallel_config['world_size'] = 2
        model.model.parallel_config['backend'] = "nccl" if torch.cuda.is_available() else "gloo"
        model.model.parallel_config['phase'] = "0.1"
        model.model.parallel_config['message'] = "Phase 0.1: Node config active"
        
        logging.info(f"{LOG_PREFIX} Node config applied")
        logging.info(f"{LOG_PREFIX} Config: {model.model.parallel_config}")
        
        return (model,)


NODE_CLASS_MAPPINGS = {
    "ParallelAttentionConfig": ParallelAttentionConfig,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ParallelAttentionConfig": "Parallel Attention Config",
}