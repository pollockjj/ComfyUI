"""Ring-Attention module using xfuser USP (Logic Seam pattern).

Per core_plan.md Section II.D: "The launcher node will use add_object_patch
to replace attention blocks with a custom ParallelAttentionModule whose
forward pass contains the actual TP logic."
"""

import torch
import torch.nn as nn
import logging

LOG_PREFIX = "⚡ [RingAttentionModule]"

# Global counter for first N calls
_forward_call_count = 0
_MAX_LOGGED_CALLS = 10


class RingAttentionModule(nn.Module):
    """Wraps Flux SelfAttention with xfuser USP for Ring-Attention.
    
    Installed via add_object_patch (Logic Seam). Preserves original module's
    weights (qkv, norm, proj) but replaces attention computation with
    xfuser's USP when sp_world_size > 1.
    """
    
    def __init__(self, original_module: nn.Module, use_usp: bool = True):
        """Initialize Ring-Attention wrapper.
        
        Args:
            original_module: Original Flux SelfAttention module
            use_usp: Enable xfuser USP (False = passthrough for testing)
        """
        super().__init__()
        
        # Preserve original module (holds trained weights)
        self.original = original_module
        self.num_heads = original_module.num_heads
        self.use_usp = use_usp
        
        # Proxy all attributes to original module (for Flux compatibility)
        # Flux's DoubleStreamBlock calls self.img_attn.qkv() directly
        for attr_name in ['qkv', 'norm', 'proj']:
            if hasattr(original_module, attr_name):
                attr = getattr(original_module, attr_name)
                setattr(self, attr_name, attr)
                
                # Enable comfy_cast_weights on FSDP2-wrapped Linear layers
                # This tells ComfyUI's ops.py to skip manual casting since FSDP2 handles it
                if hasattr(attr, 'comfy_cast_weights'):
                    attr.comfy_cast_weights = True
        
        # Also set on self for safety
        self.comfy_cast_weights = True
        
        # Import xfuser - REQUIRED for Ring-Attention
        if use_usp:
            from xfuser.model_executor.layers.usp import USP
            from xfuser.core.distributed import get_sequence_parallel_world_size
            self._usp_fn = USP
            self._get_sp_size = get_sequence_parallel_world_size
    
    def forward(self, x: torch.Tensor, pe: torch.Tensor) -> torch.Tensor:
        """Forward with optional Ring-Attention.
        
        Args:
            x: Input [batch, seq, hidden]
            pe: RoPE embeddings
        
        Returns:
            Attention output [batch, seq, hidden]
        """
        global _forward_call_count
        _forward_call_count += 1
        
        # Check if USP is needed
        sp_size = self._get_sp_size() if self.use_usp else 1
        
        # PRINT TO STDOUT - BYPASS ALL LOGGING
        if _forward_call_count <= _MAX_LOGGED_CALLS:
            msg = (
                f"🔥🔥🔥 RINGATTENTION FORWARD CALL #{_forward_call_count}: "
                f"use_usp={self.use_usp}, sp_size={sp_size}, x.shape={x.shape}, "
                f"device={x.device}"
            )
            print(msg, flush=True)
            logging.warning(msg)
        
        if not self.use_usp or sp_size == 1:
            # Single GPU or USP disabled - use original
            if _forward_call_count <= _MAX_LOGGED_CALLS:
                msg = f"🔥 CALL #{_forward_call_count} → Using ORIGINAL attention (sp_size={sp_size})"
                print(msg, flush=True)
                logging.warning(f"{LOG_PREFIX} {msg}")
            return self._forward_original(x, pe)
        
        # Multi-GPU with USP enabled
        if _forward_call_count <= _MAX_LOGGED_CALLS:
            msg = f"🔥 CALL #{_forward_call_count} → Using USP Ring-Attention (sp_size={sp_size})"
            print(msg, flush=True)
            logging.warning(f"{LOG_PREFIX} {msg}")
        return self._forward_usp(x, pe)
    
    def _forward_original(self, x: torch.Tensor, pe: torch.Tensor) -> torch.Tensor:
        """Original Flux attention (copied from SelfAttention)."""
        global _forward_call_count
        from comfy.ldm.flux.math import attention, apply_rope
        
        if _forward_call_count <= _MAX_LOGGED_CALLS:
            msg = f"🔥 CALL #{_forward_call_count} _forward_original: Executing standard Flux attention"
            print(msg, flush=True)
            logging.warning(f"{LOG_PREFIX} {msg}")
        
        qkv = self.original.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q, k = self.original.norm(q, k)
        q, k = apply_rope(q, k, pe)
        
        x = attention(q, k, v, pe=pe)
        x = self.original.proj(x)
        
        if _forward_call_count <= _MAX_LOGGED_CALLS:
            msg = f"🔥 CALL #{_forward_call_count} _forward_original: Completed, output.shape={x.shape}"
            print(msg, flush=True)
            logging.warning(f"{LOG_PREFIX} {msg}")
        return x
    
    def _forward_usp(self, x: torch.Tensor, pe: torch.Tensor) -> torch.Tensor:
        """xfuser USP Ring-Attention path."""
        global _forward_call_count
        from comfy.ldm.flux.math import apply_rope
        
        if _forward_call_count <= _MAX_LOGGED_CALLS:
            msg = f"🔥 CALL #{_forward_call_count} _forward_usp ENTRY: x.shape={x.shape}, device={x.device}"
            print(msg, flush=True)
            logging.warning(f"{LOG_PREFIX} {msg}")
        
        # Extract Q/K/V
        qkv = self.original.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q, k = self.original.norm(q, k)
        
        if _forward_call_count <= _MAX_LOGGED_CALLS:
            msg = f"🔥 CALL #{_forward_call_count} Q/K/V extracted: q.shape={q.shape}"
            print(msg, flush=True)
            logging.warning(f"{LOG_PREFIX} {msg}")
        
        # Apply RoPE
        q, k = apply_rope(q, k, pe)
        
        if _forward_call_count <= _MAX_LOGGED_CALLS:
            msg = f"🔥 CALL #{_forward_call_count} RoPE applied"
            print(msg, flush=True)
            logging.warning(f"{LOG_PREFIX} {msg}")
        
        # Reshape: [B, S, H*D] -> [B, H, S, D]
        batch, seq, _ = q.shape
        head_dim = q.shape[-1] // self.num_heads
        
        q = q.view(batch, seq, self.num_heads, head_dim).transpose(1, 2)
        k = k.view(batch, seq, self.num_heads, head_dim).transpose(1, 2)
        v = v.view(batch, seq, self.num_heads, head_dim).transpose(1, 2)
        
        # Call xfuser USP
        if _forward_call_count <= _MAX_LOGGED_CALLS:
            msg = f"🔥 CALL #{_forward_call_count} 🚀 Calling xfuser USP: q.shape={q.shape}"
            print(msg, flush=True)
            logging.warning(f"{LOG_PREFIX} {msg}")
        try:
            output = self._usp_fn(
                query=q,
                key=k,
                value=v,
                dropout_p=0.0,
                is_causal=False
            )
            if _forward_call_count <= _MAX_LOGGED_CALLS:
                msg = f"🔥 CALL #{_forward_call_count} ✅ USP returned: output.shape={output.shape}"
                print(msg, flush=True)
                logging.warning(f"{LOG_PREFIX} {msg}")
        except Exception as e:
            msg = f"❌ USP call failed: {e}"
            print(msg, flush=True)
            logging.error(f"{LOG_PREFIX} {msg}")
            raise
        
        # Reshape back: [B, H, S, D] -> [B, S, H*D]
        output = output.transpose(1, 2).contiguous().view(batch, seq, -1)
        
        if _forward_call_count <= _MAX_LOGGED_CALLS:
            msg = f"🔥 CALL #{_forward_call_count} Reshaped output: {output.shape}"
            print(msg, flush=True)
            logging.warning(f"{LOG_PREFIX} {msg}")
        
        # Output projection
        output = self.original.proj(output)
        
        if _forward_call_count <= _MAX_LOGGED_CALLS:
            msg = f"🔥 CALL #{_forward_call_count} _forward_usp COMPLETE: output.shape={output.shape}"
            print(msg, flush=True)
            logging.warning(f"{LOG_PREFIX} {msg}")
        
        return output
