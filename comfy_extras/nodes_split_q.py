"""
Split-Q Parallel Attention Configuration Node

Provides a ComfyUI node that sets an intent flag for enabling Split-Q dual-GPU sampling.
The node only records user intent; runtime attributes are attached post-pre_run() by CFGGuider.
"""

import logging


class SplitQIntentConfig:
    """
    Sets the split_q_requested intent flag in transformer_options.
    Runtime attributes (replica models, devices) are attached post-pre_run() by CFGGuider.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "enable_split_q": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "flag_model"
    CATEGORY = "advanced/split_q"
    
    def flag_model(self, model, enable_split_q):
        """
        Sets the split_q_requested intent flag in transformer_options.
        Does NOT attach replica models or device attributes - those are applied
        post-pre_run() by CFGGuider.outer_sample().
        """
        clone = model.clone()
        clone.model_options = dict(clone.model_options)
        clone.model_options.setdefault("transformer_options", {})["split_q_requested"] = enable_split_q
        
        logging.info("⚡ [split-q][Node] enable_split_q=%s", enable_split_q)
        
        return (clone,)


NODE_CLASS_MAPPINGS = {
    "SplitQIntentConfig": SplitQIntentConfig,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SplitQIntentConfig": "Split-Q Parallel Attention Config",
}
