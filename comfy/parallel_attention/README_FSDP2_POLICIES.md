# FSDP2 Wrapping Policies

## Overview

FSDP2 policies determine how PyTorch shards model parameters across GPUs.
Each wrapped unit becomes an independent sharding boundary.

## Policy Registry

Use `FSDP2PolicyRegistry` to register and retrieve model-specific policies:

```python
from comfy.parallel_attention import FSDP2PolicyRegistry

# Get policy
policy_fn = FSDP2PolicyRegistry.get_policy("flux")
policy = policy_fn()

# Use with FSDP2
from torch.distributed.fsdp import fully_shard
# FSDPModelPatcher applies fully_shard() to each matching module
```

## Flux Policy

**Wrapped Layers:**
- `DoubleStreamBlock` (19 instances, ~350MB each)
- `SingleStreamBlock` (38 instances, ~250MB each)

**Result:**
- 57 FSDP sharding units
- 22GB model → ~11GB per GPU (2-way sharding)

**Design Rationale:**
- Natural sharding boundaries (self-contained attention+MLP)
- Matches Raylight production implementation
- Minimizes communication overhead

## Qwen Image Policy

**Wrapped Layers:**
- `QwenImageTransformerBlock` (60 instances)

**Result:**
- 60 FSDP sharding units
- Dual-stream attention (img + txt) with modulation

## Wan Policy

**Wrapped Layers:**
- `WanAttentionBlock` (32 instances default, configurable)

**Result:**
- 32 FSDP sharding units (default)
- Self-attention + cross-attention + FFN per block

## Adding New Policies

```python
@FSDP2PolicyRegistry.register("my_model")
def my_model_fsdp2_policy():
    from my_model import TransformerBlock
    
    return partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={TransformerBlock}
    )
```

## Testing

Run tests in ComfyUI:
1. Add "Test Distributed Runtime" node
2. Set `test_type` to "fsdp_policy" or "all"
3. Execute workflow

Expected output:
```
⚡ [Parallel-Attention] [Test] Test 5: FSDP2 Policy Registry
⚡ [Parallel-Attention] [Test] FSDP2 Policy test passed:
⚡ [Parallel-Attention] [Test]   Model: flux
⚡ [Parallel-Attention] [Test]   Registered: True
⚡ [Parallel-Attention] [Test]   Available policies: ['flux', 'qwen_image', 'wan']
```
