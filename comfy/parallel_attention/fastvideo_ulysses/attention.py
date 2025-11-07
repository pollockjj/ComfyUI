"""Pure Ulysses attention implementation using flash-attn."""

import logging
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Tuple

from flash_attn import flash_attn_func

from comfy.parallel_attention.fastvideo_ulysses.communicator import (
    get_sp_rank,
    get_sp_world_size,
    all_to_all_4d,
    all_gather_nd,
)

LOG_PREFIX = "⚡ [FastVideo-Ulysses][Attention]"
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
    
    def __init__(self, num_heads: int, head_dim: int):
        """
        Args:
            num_heads: Total number of attention heads
            head_dim: Dimension of each head
        """
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.world_size = None  # Will be set on first forward
        self.rank = None
        
        logger.info(
            f"{LOG_PREFIX} Initialized (num_heads={num_heads}, head_dim={head_dim})"
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
        
        # Step 1: All-to-all for main QKV (scatter heads, gather sequence)
        # Before: [batch, seq_len, num_heads, head_dim]
        # After:  [batch, seq_len * world_size, num_heads / world_size, head_dim]
        qkv = torch.stack([q, k, v], dim=0)  # [3, batch, seq, heads, dim]
        qkv = all_to_all_4d(qkv, scatter_dim=3, gather_dim=2)  # Scatter heads (dim=3), gather seq (dim=2)
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
        
        # Step 3: Local flash attention
        # flash_attn_func expects [batch, seq_len, num_heads, head_dim]
        output = flash_attn_func(q, k, v, causal=False)
        
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
            scatter_dim=2,  # Scatter sequence (was gathered)
            gather_dim=3    # Gather heads (was scattered)
        )
        
        return distributed_output, replicated_output
