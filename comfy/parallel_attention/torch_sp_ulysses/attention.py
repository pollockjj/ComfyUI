"""Pure Ulysses attention implementation with pluggable backends."""

import logging
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Tuple

from comfy.parallel_attention.torch_sp_ulysses.communicator import (
    get_sp_rank,
    get_sp_world_size,
    all_to_all_4d,
    all_gather_nd,
)

LOG_PREFIX = "⚡ [Parallel-Attention][TorchSP-Ulysses][Attention]"
logger = logging.getLogger(__name__)


@dataclass
class UlyssesAttentionMetadata:
    """Minimal metadata for attention. Placeholder for V1."""
    pass


class UlyssesAttention:
    """Pure Ulysses-style sequence parallel attention.
    
    Architecture:
    1. All-to-all: scatter heads → gather sequence (for image tokens)
    2. Local shard replicated tokens by heads (for text tokens)  
    3. Local flash attention
    4. All-gather replicated output
    5. All-to-all: scatter sequence → gather heads (reverse step 1)
    """
    
    def __init__(self, num_heads: int, head_dim: int, backend: str = "sdpa"):
        """
        Args:
            num_heads: Total number of attention heads
            head_dim: Dimension of each head
            backend: Attention backend - "flash", "sdpa", or "math"
        """
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.backend = backend
        self.world_size = None  # Will be set on first forward
        self.rank = None
        
        logger.info(
            f"{LOG_PREFIX} Initialized (num_heads={num_heads}, head_dim={head_dim}, backend={backend})"
        )
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        replicated_q: Optional[torch.Tensor] = None,
        replicated_k: Optional[torch.Tensor] = None,
        replicated_v: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass with sequence parallelism.
        
        Args:
            q, k, v: [batch, seq_len, num_heads, head_dim] - image tokens (distributed)
            replicated_q/k/v: [batch, text_len, num_heads, head_dim] - text tokens (replicated)
        
        Returns:
            output: [batch, seq_len, num_heads, head_dim] - image output
            replicated_output: [batch, text_len, num_heads, head_dim] - text output (if provided)
        """
        # Lazy init
        if self.world_size is None:
            self.world_size = get_sp_world_size()
            self.rank = get_sp_rank()
        
        batch_size, seq_len, num_heads, head_dim = q.shape
        
        logger.info(f"{LOG_PREFIX} Input shapes: q={q.shape}, k={k.shape}, v={v.shape}")
        
        # Step 1: All-to-all for main QKV (scatter heads, gather sequence)
        # Before: [batch, seq_len, num_heads, head_dim]
        # After:  [batch, seq_len * world_size, num_heads / world_size, head_dim]
        qkv = torch.stack([q, k, v], dim=0)  # [3, batch, seq, heads, dim]
        logger.info(f"{LOG_PREFIX} Before all-to-all: qkv.shape={qkv.shape}")
        qkv = all_to_all_4d(qkv, scatter_dim=3, gather_dim=2)  # Scatter heads (dim=3), gather seq (dim=2)
        logger.info(f"{LOG_PREFIX} After all-to-all: qkv.shape={qkv.shape}")
        q, k, v = qkv.unbind(0)
        
        # Step 2: Handle replicated tokens (text) if provided
        replicated_output = None
        if replicated_q is not None:
            assert replicated_k is not None and replicated_v is not None
            
            # Shard replicated tokens locally by head dimension
            # Each rank gets a slice of heads
            heads_per_rank = self.num_heads // self.world_size
            head_start = self.rank * heads_per_rank
            head_end = (self.rank + 1) * heads_per_rank
            
            replicated_q = replicated_q[:, :, head_start:head_end, :]
            replicated_k = replicated_k[:, :, head_start:head_end, :]
            replicated_v = replicated_v[:, :, head_start:head_end, :]
            
            # Concatenate with distributed tokens
            # q, k, v are now [batch, seq * world_size, heads / world_size, dim]
            # replicated are [batch, text_len, heads / world_size, dim]
            q = torch.cat([q, replicated_q], dim=1)
            k = torch.cat([k, replicated_k], dim=1)
            v = torch.cat([v, replicated_v], dim=1)
        
        # Step 3: Local attention with pluggable backend
        output = self._local_attention(q, k, v)
        
        # Step 4: Separate distributed and replicated outputs
        if replicated_q is not None:
            distributed_seq_len = seq_len * self.world_size
            distributed_output = output[:, :distributed_seq_len, :, :]
            replicated_output = output[:, distributed_seq_len:, :, :]
            
            # All-gather replicated output (gather heads)
            replicated_output = all_gather_nd(replicated_output.contiguous(), dim=2)
        else:
            distributed_output = output
        
        # Step 5: All-to-all to reverse initial redistribution
        # Before: [batch, seq * world_size, heads / world_size, dim]
        # After:  [batch, seq, heads, dim]
        distributed_output = all_to_all_4d(
            distributed_output, 
            scatter_dim=1,  # Scatter sequence (was gathered)
            gather_dim=2    # Gather heads (was scattered)
        )
        
        logger.info(f"{LOG_PREFIX} After reverse all-to-all: distributed_output.shape={distributed_output.shape}")
        
        return distributed_output, replicated_output
    
    def _local_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """Execute local attention using selected backend.
        
        Args:
            q, k, v: [batch, seq_len, num_heads, head_dim]
        
        Returns:
            output: [batch, seq_len, num_heads, head_dim]
        """
        if self.backend == "flash":
            # FlashAttention 2/3 (fastest, requires flash-attn package)
            from flash_attn import flash_attn_func
            return flash_attn_func(q, k, v, causal=False)
        
        elif self.backend == "sdpa":
            # PyTorch scaled_dot_product_attention (PyTorch 2.0+)
            # Automatically uses FlashAttention/Memory-Efficient attention when available
            logger.info(
                f"{LOG_PREFIX} Calling SDPA: q.shape={q.shape}, k.shape={k.shape}, v.shape={v.shape}"
            )
            result = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, 
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False
            )
            logger.info(f"{LOG_PREFIX} SDPA output shape: {result.shape}")
            return result
        
        elif self.backend == "math":
            # Manual attention (fallback, slowest but always works)
            return self._manual_attention(q, k, v)
        
        else:
            raise ValueError(f"Unknown attention backend: {self.backend}")
    
    def _manual_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """Manual attention implementation (fallback).
        
        Args:
            q, k, v: [batch, seq_len, num_heads, head_dim]
        
        Returns:
            output: [batch, seq_len, num_heads, head_dim]
        """
        # q, k, v: [batch, seq_len, num_heads, head_dim]
        scale = 1.0 / (self.head_dim ** 0.5)
        
        # Compute attention scores
        # [batch, num_heads, seq_len, head_dim] @ [batch, num_heads, head_dim, seq_len]
        # = [batch, num_heads, seq_len, seq_len]
        q = q.transpose(1, 2)  # [batch, num_heads, seq_len, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
        
        # Apply attention to values
        output = torch.matmul(attn_weights, v)  # [batch, num_heads, seq_len, head_dim]
        output = output.transpose(1, 2)  # [batch, seq_len, num_heads, head_dim]
        
        return output
