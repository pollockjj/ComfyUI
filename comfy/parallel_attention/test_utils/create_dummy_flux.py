"""Generate minimal Flux dummy model for FSDP2 testing.

Creates a tiny Flux checkpoint with correct structure but minimal parameters.
Follows ComfyUI Flux architecture patterns for detection compatibility.
"""

import torch
from pathlib import Path
import logging

LOG_PREFIX = "⚡ [Parallel-Attention]"


def create_minimal_flux_state_dict():
    """Create minimal Flux state dict (~1MB instead of 22GB).
    
    Strategy:
    - Keep architecture structure (layer names match real Flux)
    - Reduce dimensions to minimum (hidden_size=64 instead of 3072)
    - Only 2 double_blocks, 2 single_blocks (instead of 19+38)
    - ComfyUI's model detection will recognize this as Flux
    
    Returns:
        Dict[str, torch.Tensor]: State dict with diffusion_model prefix
    """
    
    # Minimal dimensions for fast testing
    hidden_size = 64  # vs 3072 in real Flux
    in_channels = 16  # Flux uses 16 in channels
    context_dim = 64  # vs 4096
    num_heads = 4  # vs 24
    
    state_dict = {}
    
    # 1. Input projections (required for Flux detection)
    state_dict["diffusion_model.img_in.weight"] = torch.randn(hidden_size, in_channels, 1, 1)
    state_dict["diffusion_model.img_in.bias"] = torch.randn(hidden_size)
    
    state_dict["diffusion_model.time_in.in_layer.weight"] = torch.randn(hidden_size, 256)
    state_dict["diffusion_model.time_in.in_layer.bias"] = torch.randn(hidden_size)
    state_dict["diffusion_model.time_in.out_layer.weight"] = torch.randn(hidden_size, hidden_size)
    state_dict["diffusion_model.time_in.out_layer.bias"] = torch.randn(hidden_size)
    
    state_dict["diffusion_model.vector_in.in_layer.weight"] = torch.randn(hidden_size, context_dim)
    state_dict["diffusion_model.vector_in.in_layer.bias"] = torch.randn(hidden_size)
    state_dict["diffusion_model.vector_in.out_layer.weight"] = torch.randn(hidden_size, hidden_size)
    state_dict["diffusion_model.vector_in.out_layer.bias"] = torch.randn(hidden_size)
    
    state_dict["diffusion_model.guidance_in.in_layer.weight"] = torch.randn(hidden_size, 256)
    state_dict["diffusion_model.guidance_in.in_layer.bias"] = torch.randn(hidden_size)
    state_dict["diffusion_model.guidance_in.out_layer.weight"] = torch.randn(hidden_size, hidden_size)
    state_dict["diffusion_model.guidance_in.out_layer.bias"] = torch.randn(hidden_size)
    
    state_dict["diffusion_model.txt_in.weight"] = torch.randn(hidden_size, context_dim)
    state_dict["diffusion_model.txt_in.bias"] = torch.randn(hidden_size)
    
    # 2. Double blocks (only 2 instead of 19)
    for i in range(2):
        prefix = f"diffusion_model.double_blocks.{i}"
        
        # Image attention
        state_dict[f"{prefix}.img_mod.lin.weight"] = torch.randn(6 * hidden_size, hidden_size)
        state_dict[f"{prefix}.img_mod.lin.bias"] = torch.randn(6 * hidden_size)
        state_dict[f"{prefix}.img_attn.qkv.weight"] = torch.randn(3 * hidden_size, hidden_size)
        state_dict[f"{prefix}.img_attn.qkv.bias"] = torch.randn(3 * hidden_size)
        state_dict[f"{prefix}.img_attn.norm.query_norm.scale"] = torch.randn(num_heads)
        state_dict[f"{prefix}.img_attn.norm.key_norm.scale"] = torch.randn(num_heads)
        state_dict[f"{prefix}.img_attn.proj.weight"] = torch.randn(hidden_size, hidden_size)
        state_dict[f"{prefix}.img_attn.proj.bias"] = torch.randn(hidden_size)
        
        # Image MLP
        state_dict[f"{prefix}.img_mlp.0.weight"] = torch.randn(4 * hidden_size, hidden_size)
        state_dict[f"{prefix}.img_mlp.0.bias"] = torch.randn(4 * hidden_size)
        state_dict[f"{prefix}.img_mlp.2.weight"] = torch.randn(hidden_size, 4 * hidden_size)
        state_dict[f"{prefix}.img_mlp.2.bias"] = torch.randn(hidden_size)
        
        # Text attention
        state_dict[f"{prefix}.txt_mod.lin.weight"] = torch.randn(6 * hidden_size, hidden_size)
        state_dict[f"{prefix}.txt_mod.lin.bias"] = torch.randn(6 * hidden_size)
        state_dict[f"{prefix}.txt_attn.qkv.weight"] = torch.randn(3 * hidden_size, hidden_size)
        state_dict[f"{prefix}.txt_attn.qkv.bias"] = torch.randn(3 * hidden_size)
        state_dict[f"{prefix}.txt_attn.norm.query_norm.scale"] = torch.randn(num_heads)
        state_dict[f"{prefix}.txt_attn.norm.key_norm.scale"] = torch.randn(num_heads)
        state_dict[f"{prefix}.txt_attn.proj.weight"] = torch.randn(hidden_size, hidden_size)
        state_dict[f"{prefix}.txt_attn.proj.bias"] = torch.randn(hidden_size)
        
        # Text MLP
        state_dict[f"{prefix}.txt_mlp.0.weight"] = torch.randn(4 * hidden_size, hidden_size)
        state_dict[f"{prefix}.txt_mlp.0.bias"] = torch.randn(4 * hidden_size)
        state_dict[f"{prefix}.txt_mlp.2.weight"] = torch.randn(hidden_size, 4 * hidden_size)
        state_dict[f"{prefix}.txt_mlp.2.bias"] = torch.randn(hidden_size)
    
    # 3. Single blocks (only 2 instead of 38)
    for i in range(2):
        prefix = f"diffusion_model.single_blocks.{i}"
        state_dict[f"{prefix}.linear1.weight"] = torch.randn(3 * hidden_size + 4 * hidden_size, hidden_size)
        state_dict[f"{prefix}.linear1.bias"] = torch.randn(3 * hidden_size + 4 * hidden_size)
        state_dict[f"{prefix}.linear2.weight"] = torch.randn(hidden_size, 4 * hidden_size)
        state_dict[f"{prefix}.linear2.bias"] = torch.randn(hidden_size)
        state_dict[f"{prefix}.norm.query_norm.scale"] = torch.randn(num_heads)
        state_dict[f"{prefix}.norm.key_norm.scale"] = torch.randn(num_heads)
        state_dict[f"{prefix}.modulation.lin.weight"] = torch.randn(3 * hidden_size, hidden_size)
        state_dict[f"{prefix}.modulation.lin.bias"] = torch.randn(3 * hidden_size)
    
    # 4. Final layer
    state_dict["diffusion_model.final_layer.linear.weight"] = torch.randn(hidden_size * 2, hidden_size)
    state_dict["diffusion_model.final_layer.linear.bias"] = torch.randn(hidden_size * 2)
    state_dict["diffusion_model.final_layer.adaLN_modulation.1.weight"] = torch.randn(2 * hidden_size, hidden_size)
    state_dict["diffusion_model.final_layer.adaLN_modulation.1.bias"] = torch.randn(2 * hidden_size)
    
    return state_dict


def create_dummy_flux_checkpoint(output_path: str = None):
    """Create and save dummy Flux checkpoint.
    
    Args:
        output_path: Path to save .safetensors file. 
                    Default: models/diffusion_models/flux_dummy_test.safetensors
    
    Returns:
        Path to saved checkpoint
    """
    if output_path is None:
        output_path = "models/diffusion_models/flux_dummy_test.safetensors"
    
    output_path = Path(output_path)
    
    logging.info(f"{LOG_PREFIX} [DummyFlux] Creating minimal Flux checkpoint...")
    
    state_dict = create_minimal_flux_state_dict()
    
    # Calculate size
    total_params = sum(v.numel() for v in state_dict.values())
    total_bytes = sum(v.numel() * v.element_size() for v in state_dict.values())
    
    logging.info(f"{LOG_PREFIX} [DummyFlux]   Keys: {len(state_dict)}")
    logging.info(f"{LOG_PREFIX} [DummyFlux]   Params: {total_params:,}")
    logging.info(f"{LOG_PREFIX} [DummyFlux]   Size: {total_bytes / (1024**2):.2f}MB")
    
    # Save using safetensors
    try:
        from safetensors.torch import save_file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_file(state_dict, str(output_path))
        
        file_size_mb = output_path.stat().st_size / (1024**2)
        logging.info(f"{LOG_PREFIX} [DummyFlux]   ✅ Saved: {output_path}")
        logging.info(f"{LOG_PREFIX} [DummyFlux]   File size: {file_size_mb:.2f}MB")
        
        return str(output_path)
        
    except ImportError:
        logging.error(f"{LOG_PREFIX} [DummyFlux]   ❌ safetensors not installed")
        logging.error(f"{LOG_PREFIX} [DummyFlux]   Install: pip install safetensors")
        raise


if __name__ == "__main__":
    # Standalone execution for manual testing
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    create_dummy_flux_checkpoint()
