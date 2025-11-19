"""Unit tests for TorchSP communicator primitives.

Tests the all_to_all_4d and all_gather_nd implementations using
PyTorch distributed primitives.
"""

import time
import torch
import torch.distributed as dist
from comfy.parallel_attention.tests.test_registry import register_test


@register_test("test_torch_sp_communicator_initialization")
def test_torch_sp_communicator_initialization(fsdp_model, config, rank, device):
    """Verify TorchSP communicator initialization in distributed context."""
    start = time.time()
    
    try:
        from comfy.parallel_attention.backends.torch_sp_ulysses import communicator
        
        # Check if group was initialized by worker
        sp_group = communicator.get_sp_group()
        sp_rank = communicator.get_sp_rank()
        sp_world_size = communicator.get_sp_world_size()
        
        if sp_group is None:
            return "SKIP", "TorchSP group not initialized - test requires TorchSP backend", time.time() - start
        
        # Verify rank/world_size make sense
        expected_world_size = config.get('ulysses_degree', 2)
        if sp_world_size != expected_world_size:
            return "FAIL", f"World size mismatch: {sp_world_size} != {expected_world_size}", time.time() - start
        
        if not (0 <= sp_rank < sp_world_size):
            return "FAIL", f"Invalid rank: {sp_rank} (world_size={sp_world_size})", time.time() - start
        
        return "PASS", f"TorchSP initialized: rank={sp_rank}/{sp_world_size}", time.time() - start
        
    except Exception as e:
        return "ERROR", f"{type(e).__name__}: {str(e)}", time.time() - start


@register_test("test_torch_sp_all_to_all_shape")
def test_torch_sp_all_to_all_shape(fsdp_model, config, rank, device):
    """Test all_to_all_4d shape transformations."""
    start = time.time()
    
    try:
        from comfy.parallel_attention.backends.torch_sp_ulysses import communicator
        
        # Check if TorchSP initialized
        if communicator.get_sp_group() is None:
            return "SKIP", "TorchSP not initialized - test requires TorchSP backend", time.time() - start
        
        world_size = communicator.get_sp_world_size()
        
        # Create test tensor: [batch=2, seq=1024, heads=16, dim=64]
        batch, seq, heads, dim = 2, 1024, 16, 64
        tensor = torch.randn(batch, seq, heads, dim, device=device, dtype=torch.float32)
        
        # all_to_all: scatter heads, gather seq
        # Input: [2, 1024, 16, 64]
        # After scatter heads (dim 2) into 2 chunks: each rank has H=8
        # After gather seq (dim 1): each rank gets 2x seq -> S=2048
        # Output: [2, 2048, 8, 64]
        output = communicator.all_to_all_4d(tensor, scatter_dim=2, gather_dim=1)
        
        expected_shape = (batch, seq * world_size, heads // world_size, dim)
        if output.shape != expected_shape:
            return "FAIL", f"Shape mismatch: {output.shape} != {expected_shape}", time.time() - start
        
        return "PASS", f"all_to_all_4d: {tensor.shape} → {output.shape}", time.time() - start
        
    except Exception as e:
        return "ERROR", f"{type(e).__name__}: {str(e)}", time.time() - start


@register_test("test_torch_sp_all_gather_shape")
def test_torch_sp_all_gather_shape(fsdp_model, config, rank, device):
    """Test all_gather_nd shape transformations."""
    start = time.time()
    
    try:
        from comfy.parallel_attention.backends.torch_sp_ulysses import communicator
        
        # Check if TorchSP initialized
        if communicator.get_sp_group() is None:
            return "SKIP", "TorchSP not initialized - test requires TorchSP backend", time.time() - start
        
        world_size = communicator.get_sp_world_size()
        
        # Create test tensor: [batch=2, seq=512, heads=8, dim=64]
        batch, seq, heads, dim = 2, 512, 8, 64
        tensor = torch.randn(batch, seq, heads, dim, device=device, dtype=torch.float32)
        
        # all_gather: concatenate along seq dimension (dim=1)
        # Input: [2, 512, 8, 64]
        # After gather from 2 ranks along dim 1: S = 512 * 2 = 1024
        # Output: [2, 1024, 8, 64]
        output = communicator.all_gather_nd(tensor, dim=1)
        
        expected_shape = (batch, seq * world_size, heads, dim)
        if output.shape != expected_shape:
            return "FAIL", f"Shape mismatch: {output.shape} != {expected_shape}", time.time() - start
        
        return "PASS", f"all_gather_nd: {tensor.shape} → {output.shape}", time.time() - start
        
    except Exception as e:
        return "ERROR", f"{type(e).__name__}: {str(e)}", time.time() - start
