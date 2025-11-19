"""
CGDP Intent Configuration Node
UI node that sets the cgdp_requested flag in model_options.
"""


class CGDPIntentConfig:
    """
    Node to enable Coarse-Grained Data Parallelism (CGDP) for dual-GPU inference.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "enable_cgdp": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch_model"
    CATEGORY = "advanced/cgdp"
    
    def patch_model(self, model, enable_cgdp):
        """
        Patch model_options to set cgdp_requested flag.
        """
        if enable_cgdp:
            # Clone model to avoid mutating original
            cloned_model = model.clone()
            
            # Set CGDP intent flag
            if "transformer_options" not in cloned_model.model_options:
                cloned_model.model_options["transformer_options"] = {}
            
            cloned_model.model_options["transformer_options"]["cgdp_requested"] = True
            
            return (cloned_model,)
        else:
            return (model,)


NODE_CLASS_MAPPINGS = {
    "CGDPIntentConfig": CGDPIntentConfig,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CGDPIntentConfig": "CGDP Enable",
}
