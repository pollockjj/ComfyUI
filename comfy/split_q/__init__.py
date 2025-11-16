"""
Split-Q Parallel Attention Module

Single-process, dual-GPU attention implementation using CUDA streams.
No torch.distributed, no multiprocessing - pure CUDA async execution.

Architecture:
- Q tensor split along sequence dimension
- K/V tensors fully replicated on both GPUs
- Async compute via CUDA streams
- NVLink gather for final concatenation
"""

__version__ = "0.1.0"

from .split_q_attention import attention_split

__all__ = ['attention_split']
