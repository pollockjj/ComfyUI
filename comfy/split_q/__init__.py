"""
Split-Q module for dual-GPU parallel attention
"""

from .split_q_attention import attention_split

__all__ = ['attention_split']
