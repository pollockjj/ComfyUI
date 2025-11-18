"""
Split-Q ModelPatcher Integration Hooks
State injection and attention override orchestration.
"""

import torch
import logging
import comfy.ldm.modules.attention
from .state import SplitQState
from .attention import _parallel_attention_compute, _serial_attention_compute

_logger = logging.getLogger('comfy')


def hook_into_modelpatcher(model_patcher, split_q_enabled, validation_mode=False):
    """
    Called from CFGGuider.outer_sample() (post-pre_run) or custom node.
    Injects SplitQState and sets optimized_attention_override.
    
    Architecture (Blueprint Section 2.2):
    - This is the "constructor" - runs ONCE per sampling
    - Creates persistent CUDA streams/events
    - Injects state into transformer_options
    - Sets attention override function
    
    Args:
        model_patcher: ComfyUI ModelPatcher instance
        split_q_enabled: bool, enable Split-Q parallelism
        validation_mode: bool, run parallel+serial and compare (Phase 3)
    """
    
    # CHECKPOINT: Hardware pre-flight check
    if split_q_enabled and torch.cuda.device_count() < 2:
        _logger.warning("⚡ [split-q][hooks] Split-Q requires >= 2 GPUs")
        _logger.warning(f"⚡ [split-q][hooks] Detected {torch.cuda.device_count()} GPU(s), disabling Split-Q")
        split_q_enabled = False
    
    # Get or create transformer_options dict
    transformer_options = model_patcher.model_options.setdefault("transformer_options", {})
    
    # Get or create SplitQState object
    state = transformer_options.get("split_q_state")
    if not isinstance(state, SplitQState):
        _logger.info("⚡ [split-q][hooks] Initializing new SplitQState...")
        state = SplitQState(enabled=split_q_enabled, validation_mode=validation_mode)
        transformer_options["split_q_state"] = state
    else:
        # State exists from previous call, update flags
        _logger.info("⚡ [split-q][hooks] Reusing existing SplitQState, updating flags")
        state.enabled = split_q_enabled
        state.validation_mode = validation_mode
    
    if not state.enabled:
        # If disabling, restore original function if we have it
        if state.original_attn_func:
            _logger.info("⚡ [split-q][hooks] Restoring original attention function")
            transformer_options["optimized_attention_override"] = state.original_attn_func
        return  # Early exit, do not override
    
    # CHECKPOINT: Store original attention function (if not already stored)
    if state.original_attn_func is None:
        current_override = transformer_options.get("optimized_attention_override")
        if current_override and current_override != split_q_attention_wrapper:
            # Another override is active (xFormers, Sage, etc.)
            _logger.info(f"⚡ [split-q][hooks] Storing existing override: {current_override.__name__}")
            state.original_attn_func = current_override
        else:
            # No override, use baseline attention_split
            _logger.info("⚡ [split-q][hooks] Storing baseline attention: attention_split")
            state.original_attn_func = comfy.ldm.modules.attention.attention_split
    
    # CHECKPOINT: Set the override function
    transformer_options["optimized_attention_override"] = split_q_attention_wrapper
    _logger.info("⚡ [split-q][hooks] Attention override installed, state is ready")
    _logger.info(f"⚡ [split-q][hooks] {state}")


def split_q_attention_wrapper(original_func, *args, **kwargs):
    """
    Attention override function (the "executor").
    Called hundreds of times per sampling run (per attention block).
    
    Architecture (Blueprint Section 2.2):
    - Lightweight dispatch function
    - Reads SplitQState from kwargs
    - Routes to parallel execution or fallback
    - Handles OOM gracefully (Phase 3)
    
    Signature (per comfy/ldm/modules/attention.py:129):
        override(original_func, *args, **kwargs)
        - original_func: The wrapped attention function (e.g., attention_split)
        - args: (q, k, v, heads, mask, ...) attention parameters
        - kwargs: Contains transformer_options with split_q_state
        
    Returns:
        Attention output tensor [batch, seq_len, heads*dim_head]
    """
    # CHECKPOINT: Log entry IMMEDIATELY (first thing, no conditions)
    # _logger.info("⚡ [split-q][wrapper] ENTRY - Override function called")
    # _logger.info(f"⚡ [split-q][wrapper] args length={len(args)}, kwargs keys={list(kwargs.keys())}")
    
    # CHECKPOINT: Extract state from kwargs
    transformer_options = kwargs.get("transformer_options", {})
    state = transformer_options.get("split_q_state")
    # _logger.info(f"⚡ [split-q][wrapper] transformer_options present={bool(transformer_options)}, state present={bool(state)}")
    
    # FALLBACK PATH: No state or not ready
    if not state or not state.is_ready():
        # _logger.info(f"⚡ [split-q][wrapper] FALLBACK - state={'present' if state else 'None'}, ready={state.is_ready() if state else 'N/A'}")
        # Call the original wrapped function
        return original_func(*args, **kwargs)
    
    # _logger.info("⚡ [split-q][wrapper] State ready, proceeding to parallel path")
    
    # Extract attention arguments
    # Signature: attention_func(q, k, v, heads=X, mask=None, **kwargs)
    # Per logs: args=(q,k,v), kwargs={'heads':X, 'mask':Y, 'transformer_options':{...}}
    # _logger.info(f"⚡ [split-q][wrapper] Extracting args: len={len(args)}")
    if len(args) < 3:
        _logger.error(f"⚡ [split-q][wrapper] Invalid args length={len(args)}, expected >=3")
        return original_func(*args, **kwargs)
    
    q, k, v = args[0], args[1], args[2]
    # _logger.info(f"⚡ [split-q][wrapper] Extracted q/k/v: shapes={q.shape},{k.shape},{v.shape}")
    heads = kwargs.get('heads')
    mask = kwargs.get('mask', None)
    # _logger.info(f"⚡ [split-q][wrapper] heads={heads}, mask={'present' if mask is not None else 'None'}")
    
    if heads is None:
        _logger.error("⚡ [split-q][wrapper] 'heads' parameter missing in kwargs")
        return original_func(*args, **kwargs)
    
    # _logger.info("⚡ [split-q][wrapper] Filtering kwargs and calling parallel compute")
    
    # Remove heads and mask from kwargs before passing to parallel compute
    # (we'll pass them as positional args to avoid "multiple values" error)
    kwargs_filtered = {k: v for k, v in kwargs.items() if k not in ['heads', 'mask']}
    
    try:
        # _logger.info(f"⚡ [split-q][wrapper] Entering try block, validation_mode={state.validation_mode}")
        if state.validation_mode:
            # VALIDATION MODE (Phase 3)
            # Run both parallel and serial, compare results
            _logger.debug("⚡ [split-q][wrapper] Validation mode: running parallel+serial")
            
            out_parallel = _parallel_attention_compute(state, q, k, v, heads, mask, **kwargs_filtered)
            out_serial = _serial_attention_compute(state, q, k, v, heads, mask, **kwargs_filtered)
            
            # Compare results (Blueprint Section 6.1)
            is_close = torch.allclose(
                out_parallel.to(state.device_0),
                out_serial,
                atol=1e-8,
                rtol=1e-5
            )
            
            if not is_close:
                diff = (out_parallel.to(state.device_0) - out_serial).abs().max().item()
                _logger.warning(f"⚡ [split-q][wrapper] VALIDATION FAILED! max_diff={diff:.2e}")
                _logger.warning("⚡ [split-q][wrapper] Returning serial result for correctness")
            else:
                _logger.debug("⚡ [split-q][wrapper] Validation passed, results match")
            
            # Always return serial result in validation mode
            return out_serial
        
        else:
            # PRODUCTION MODE (Phase 1)
            # _logger.info("⚡ [split-q][wrapper] Production mode - calling _parallel_attention_compute")
            return _parallel_attention_compute(state, q, k, v, heads, mask, **kwargs_filtered)
    
    except RuntimeError as e:
        # OOM or other CUDA error (Blueprint Section 6.2)
        if "out of memory" in str(e).lower():
            _logger.warning(f"⚡ [split-q][wrapper] OOM detected on cuda:1, disabling Split-Q")
            _logger.warning(f"⚡ [split-q][wrapper] Error: {e}")
            
            # Latch to disabled for all subsequent calls
            state.enabled = False
            torch.cuda.empty_cache()
            
            # Fallback to original function
            return original_func(*args, **kwargs)
        else:
            # Unknown error, disable and re-raise
            _logger.error(f"⚡ [split-q][wrapper] Unexpected runtime error: {e}")
            state.enabled = False
            raise e
