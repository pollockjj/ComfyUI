"""Debug harness to find first divergence in step 0 between two models."""

import logging
import torch

_logger = logging.getLogger(__name__)


def compare_tensors(name, t0, t1, rtol=1e-5, atol=1e-8):
    """Compare two tensors and log differences."""
    if t0 is None and t1 is None:
        _logger.info(f"⚡ [split-q][Debug] {name}: both None ✅")
        return True
    if t0 is None or t1 is None:
        _logger.error(f"⚡ [split-q][Debug] {name}: one is None ❌ t0={t0 is not None} t1={t1 is not None}")
        return False
    
    if t0.shape != t1.shape:
        _logger.error(f"⚡ [split-q][Debug] {name}: shape mismatch ❌ t0={t0.shape} t1={t1.shape}")
        return False
    
    if t0.device != t1.device:
        _logger.warning(f"⚡ [split-q][Debug] {name}: device mismatch t0={t0.device} t1={t1.device}, moving to cpu")
        t0 = t0.cpu()
        t1 = t1.cpu()
    
    equal = torch.allclose(t0, t1, rtol=rtol, atol=atol)
    
    if equal:
        _logger.info(f"⚡ [split-q][Debug] {name}: IDENTICAL ✅")
    else:
        diff = (t0 - t1).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        _logger.error(f"⚡ [split-q][Debug] {name}: DIVERGED ❌ max_diff={max_diff:.6e} mean_diff={mean_diff:.6e}")
    
    return equal


def debug_single_step_compare(model_0, model_1, seed, steps, cfg, sampler_name, scheduler, 
                               positive, negative, latent, denoise=1.0):
    """Run exactly ONE step on each model and compare at every checkpoint."""
    import comfy.sample
    import comfy.samplers
    
    _logger.info("⚡ [split-q][Debug] ========== SINGLE STEP COMPARISON ==========")
    
    # Prepare identical inputs
    latent_image = latent["samples"]
    device_0 = model_0.load_device
    device_1 = model_1.load_device
    
    latent_image = comfy.sample.fix_empty_latent_channels(model_0, latent_image)
    batch_inds = latent.get("batch_index", None)
    noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)
    
    _logger.info(f"⚡ [split-q][Debug] Initial noise: shape={noise.shape} device={noise.device}")
    _logger.info(f"⚡ [split-q][Debug] Initial latent: shape={latent_image.shape} device={latent_image.device}")
    
    # Use nodes.common_ksampler with steps=1 for single step
    from nodes import common_ksampler
    
    _logger.info("⚡ [split-q][Debug] Running model_0 for 1 step...")
    result_0 = common_ksampler(
        model_0, seed, steps=1, cfg=cfg, sampler_name=sampler_name,
        scheduler=scheduler, positive=positive, negative=negative,
        latent=latent, denoise=denoise
    )
    
    # Reset noise for model_1 (must use same seed)
    noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)
    
    _logger.info("⚡ [split-q][Debug] Running model_1 for 1 step...")
    result_1 = common_ksampler(
        model_1, seed, steps=1, cfg=cfg, sampler_name=sampler_name,
        scheduler=scheduler, positive=positive, negative=negative,
        latent=latent, denoise=denoise
    )
    
    # Compare final outputs
    samples_0 = result_0[0]["samples"]
    samples_1 = result_1[0]["samples"]
    
    _logger.info(f"⚡ [split-q][Debug] Output 0: shape={samples_0.shape} device={samples_0.device}")
    _logger.info(f"⚡ [split-q][Debug] Output 1: shape={samples_1.shape} device={samples_1.device}")
    
    identical = compare_tensors("step_0_output", samples_0, samples_1)
    
    if identical:
        _logger.info("⚡ [split-q][Debug] ========== STEP 0: IDENTICAL ✅ ==========")
    else:
        _logger.error("⚡ [split-q][Debug] ========== STEP 0: DIVERGED ❌ ==========")
        _logger.error("⚡ [split-q][Debug] BINARY SEARCH REQUIRED - need to instrument model forward pass")
    
    return result_0, result_1, identical
