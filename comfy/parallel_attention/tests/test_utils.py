"""Utility functions for writing tests."""

import torch
from typing import Tuple


def create_test_tensors(
    batch: int,
    seq: int,
    heads: int,
    dim: int,
    device: torch.device,
    use_bfloat16: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create test Q, K, V tensors in FastVideo convention [batch, seq, heads, dim].
    
    Args:
        batch: Batch size
        seq: Sequence length
        heads: Number of attention heads
        dim: Head dimension
        device: Target device
        use_bfloat16: Use bfloat16 if available, else float32
    
    Returns:
        (q, k, v) tensors with shape [batch, seq, heads, dim]
    """
    dtype = torch.bfloat16 if (use_bfloat16 and torch.cuda.is_available()) else torch.float32
    
    q = torch.randn(batch, seq, heads, dim, device=device, dtype=dtype)
    k = torch.randn(batch, seq, heads, dim, device=device, dtype=dtype)
    v = torch.randn(batch, seq, heads, dim, device=device, dtype=dtype)
    
    return q, k, v


def create_flux_inputs(
    device: torch.device,
    img_seq: int = 2048,
    txt_seq: int = 128,
    hidden_dim: int = 3072,
    use_bfloat16: bool = True
) -> dict:
    """Create minimal inputs for Flux double-stream block forward.
    
    Args:
        device: Target device
        img_seq: Image sequence length
        txt_seq: Text sequence length
        hidden_dim: Hidden dimension size
        use_bfloat16: Use bfloat16 if available, else float32
    
    Returns:
        Dict with keys: img, txt, vec, pe
    """
    dtype = torch.bfloat16 if (use_bfloat16 and torch.cuda.is_available()) else torch.float32
    
    return {
        'img': torch.randn(1, img_seq, hidden_dim, device=device, dtype=dtype),
        'txt': torch.randn(1, txt_seq, hidden_dim, device=device, dtype=dtype),
        'vec': torch.randn(1, hidden_dim, device=device, dtype=dtype),
        'pe': torch.randn(1, 1, img_seq + txt_seq, 64, 2, 2, device=device, dtype=dtype)
    }


def assert_shape_equals(
    tensor: torch.Tensor,
    expected_shape: Tuple[int, ...],
    name: str = "tensor"
) -> Tuple[bool, str]:
    """Assert tensor has expected shape.
    
    Args:
        tensor: Tensor to check
        expected_shape: Expected shape tuple
        name: Tensor name for error messages
    
    Returns:
        (success, message) tuple
    """
    if tensor.shape != expected_shape:
        return False, (
            f"{name} shape mismatch: expected {expected_shape}, got {tensor.shape}"
        )
    return True, f"{name} shape correct: {tensor.shape}"


def assert_dtype_equals(
    tensor: torch.Tensor,
    expected_dtype: torch.dtype,
    name: str = "tensor"
) -> Tuple[bool, str]:
    """Assert tensor has expected dtype.
    
    Args:
        tensor: Tensor to check
        expected_dtype: Expected dtype
        name: Tensor name for error messages
    
    Returns:
        (success, message) tuple
    """
    if tensor.dtype != expected_dtype:
        return False, (
            f"{name} dtype mismatch: expected {expected_dtype}, got {tensor.dtype}"
        )
    return True, f"{name} dtype correct: {tensor.dtype}"
