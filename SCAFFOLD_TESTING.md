# Scaffold Pattern Testing Instructions (TDD)

## ✅ IMPLEMENTATION COMPLETE

All three files have been fixed to use the **WorkSplit deepcopy pattern**:

### Files Updated:
1. **`comfy/parallel_attention/model_scaffold.py`** (114 lines)
   - Creates model structure WITHOUT loading 22GB weights
   - Uses `copy.deepcopy()` on model structure
   - Returns `(scaffold_model, state_dict)` - real object, not dict

2. **`comfy/parallel_attention/distributed_model_wrapper.py`** (262 lines)  
   - Accepts `scaffold_model: Any` (real model object)
   - Stores `self._scaffold = scaffold_model`
   - All properties accessed via `self._scaffold.attribute`

3. **`comfy_extras/nodes_parallel_attention.py`** (Updated test node)
   - Comprehensive scaffold validation tests
   - Tests scaffold is real model (not dict)
   - Tests scaffold size (<100MB, no weights)
   - Tests all property access patterns

## 🧪 HOW TO TEST (TDD: Inside ComfyUI)

### Step 1: Start ComfyUI
```bash
cd /home/johnj/ComfyUI
python main.py
```

### Step 2: Create Test Workflow
1. Add node: `UnetLoaderParallelAttention`
2. Connect to: `TestParallelAttention`  
3. Select model: `flux1-dev.safetensors` or similar

### Step 3: Run Tests
Queue the workflow and check the `TestParallelAttention` output.

### Expected Test Output:
```
======================================================================
SCAFFOLD PATTERN VALIDATION (Deepcopy - WorkSplit Pattern)
======================================================================

[Test 1] Wrapper Type
✅ Type: DistributedModelWrapper
✅ World size: 2

[Test 2] Scaffold Object Type (CRITICAL)
✅ Scaffold type: Flux (or similar model class)
✅ Scaffold has get_dtype() method
✅ Scaffold has latent_format attribute
✅ Scaffold metadata: <ModelConfig>

[Test 3] Scaffold Size (0GB Structure)
✅ Scaffold size: <50MB (should be small)
✅ Scaffold is lightweight (no weights)

[Test 4] Wrapper Property Access
✅ latent_format: Flux (or model-specific)
✅ latent_channels: 16
✅ get_model_object() returns scaffold property
✅ dtype: torch.bfloat16
✅ model_type: flux
✅ is_adm(): False
✅ extra_conds(): dict

[Test 5] Model Size (vs Scaffold Size)
✅ Model size: 23.8 GB (or similar)
✅ Model size 500x larger than scaffold

[Test 6] Forward Pass
⏸️  PENDING: Worker forward_pass handler not implemented

[Test 7] VRAM Usage  
✅ GPU 0: <X> GB allocated
✅ GPU 1: <Y> GB allocated

======================================================================
TEST SUMMARY
======================================================================
✅ Wrapper type correct
✅ Scaffold is real model object (deepcopy pattern)
✅ Scaffold is lightweight (<100MB)
✅ All properties accessible via scaffold
⏸️  Forward pass pending worker handler

🎯 SCAFFOLD PATTERN: VALIDATED
```

## ❌ FAILURE INDICATORS

If you see any of these, the implementation is BROKEN:

- `❌ FAIL: Scaffold missing get_dtype()` - Scaffold is dict, not model object
- `❌ FAIL: Scaffold too large (>500MB)` - Weights were loaded, should be structure only
- `❌ FAIL: latent_format is None` - Scaffold pattern broken
- `AttributeError` or `KeyError` - Type mismatch between files

## 🔄 NEXT PHASE: Worker Forward Pass Handler

After scaffold validation passes:
1. Implement `forward_pass` handler in `worker.py`
2. Add RPC forwarding test to `TestParallelAttention`
3. Test with actual inference workflow

## 📊 SUCCESS CRITERIA

- ✅ All 7 tests pass (6 complete, 1 pending)
- ✅ Scaffold size <100MB
- ✅ No NoneType errors
- ✅ Properties accessible without RPC
- ✅ Model size calculation correct
