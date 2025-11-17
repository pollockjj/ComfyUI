"""Utilities for Split-Q sampling orchestration.

These helpers keep both WAN replicas in lockstep while we experiment with
serial and (eventually) parallel execution strategies.  The functions here
are intentionally thin wrappers around ``nodes.common_ksampler`` so they stay
in sync with the canonical sampler behavior.
"""

from __future__ import annotations

import logging
import hashlib
import torch
from typing import Tuple

import torch

from nodes import common_ksampler

_logger = logging.getLogger(__name__)


def _clone_latent(latent: dict | None) -> dict | None:
    """Deep-copy the latent dict so samplers do not mutate shared tensors."""
    if latent is None:
        return None
    cloned: dict = {}
    for key, value in latent.items():
        if torch.is_tensor(value):
            cloned[key] = value.clone()
        else:
            cloned[key] = value
    return cloned


def _run_single(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, **kwargs):
    """Execute a single common_ksampler call with an isolated latent."""
    latent_copy = _clone_latent(latent)
    return common_ksampler(
        model,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        positive,
        negative,
        latent_copy,
        **kwargs,
    )


def _extract_samples(latent_result: Tuple[dict]) -> torch.Tensor:
    """Helper to pull the tensor we compare for equality."""
    if not latent_result:
        raise ValueError("Empty latent result tuple passed to Split-Q sampler")
    samples = latent_result[0].get("samples")
    if not torch.is_tensor(samples):
        raise ValueError("Split-Q sampler expected tensor samples in latent output")
    return samples


def serial_split_q_sample(
    model_primary,
    model_secondary,
    *,
    seed,
    steps,
    cfg,
    sampler_name,
    scheduler,
    positive,
    negative,
    latent,
    denoise=1.0,
    capture_golden=False,
    **kwargs,
):
    """Execute sampling on both models serially and validate byte-identical outputs.
    
    Args:
        capture_golden: If True, capture latent at each step via callback.
    """
    from comfy.split_q.golden_reference import save_golden_step, clear_golden_references
    import comfy.sample
    import comfy.utils
    
    if capture_golden:
        _logger.info("⚡ [split-q][Sampler] PHASE 1: Capturing golden reference via callback")
        clear_golden_references()
        
        # Prepare inputs
        latent_image = latent["samples"]
        latent_image = comfy.sample.fix_empty_latent_channels(model_primary, latent_image)
        batch_inds = latent.get("batch_index", None)
        noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)
        noise_mask = latent.get("noise_mask", None)
        
        # Callback to capture intermediate latents
        captured_latents = {}
        def capture_callback(step_i, denoised, x, total_steps):
            # Save denoised latent at this step
            captured_latents[step_i] = denoised.clone().detach()
            _logger.info(f"⚡ [split-q][Sampler] Captured step {step_i}/{total_steps}")
        
        # Call comfy.sample.sample directly with callback
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
        samples = comfy.sample.sample(
            model_primary, noise, steps, cfg, sampler_name, scheduler, 
            positive, negative, latent_image,
            denoise=denoise, disable_noise=False, start_step=None, last_step=None,
            force_full_denoise=False, noise_mask=noise_mask, 
            callback=capture_callback, disable_pbar=disable_pbar, seed=seed
        )
        
        # Save all captured latents as golden
        for step_i, step_latent in captured_latents.items():
            save_golden_step(step_i, step_latent, label="golden", metadata={
                "seed": seed,
                "step": step_i,
                "total_steps": steps,
                "sampler": sampler_name,
                "scheduler": scheduler
            })
        
        out = latent.copy()
        out["samples"] = samples
        
        _logger.info(f"⚡ [split-q][Sampler] Golden reference captured: {len(captured_latents)} steps")
        return (out,), None
    
    # Normal serial validation
    from nodes import common_ksampler
    
    _logger.info("⚡ [split-q][Sampler] serial execution starting")

    result_primary = common_ksampler(
        model_primary,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        positive,
        negative,
        latent,
        denoise=denoise,
    )

    result_secondary = common_ksampler(
        model_secondary,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        positive,
        negative,
        latent,
        denoise=denoise,
    )

    samples_primary = result_primary[0]["samples"]
    samples_secondary = result_secondary[0]["samples"]

    samples_primary_cpu = samples_primary.to("cpu")
    samples_secondary_cpu = samples_secondary.to("cpu")

    if not torch.equal(samples_primary_cpu, samples_secondary_cpu):
        delta = (samples_primary_cpu - samples_secondary_cpu).abs().max().item()
        _logger.error(
            "⚡ [split-q][Sampler][FAIL] replica outputs diverged: max_abs_diff=%.6f",
            delta,
        )
        raise ValueError("Split-Q replicas produced different latents")

    _logger.info("⚡ [split-q][Sampler][PASS] replica outputs are byte-identical")
    return result_primary, result_secondary


def interleaved_split_q_sample(
    model_primary,
    model_secondary,
    *,
    seed,
    steps,
    cfg,
    sampler_name,
    scheduler,
    positive,
    negative,
    latent,
    denoise=1.0,
    **kwargs,
):
    """ISOLATED-INTERLEAVED: Run both models independently, validate in interleaved order."""
    from comfy.split_q.golden_reference import validate_against_golden
    import comfy.sample
    import comfy.utils
    
    _logger.info("⚡ [split-q][Sampler] ISOLATED-INTERLEAVED: Starting dual execution")
    
    # Prepare inputs (shared initial conditions)
    latent_image = latent["samples"]
    latent_image = comfy.sample.fix_empty_latent_channels(model_primary, latent_image)
    batch_inds = latent.get("batch_index", None)
    noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)
    noise_mask = latent.get("noise_mask", None)
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
    
    # Interceptor for model_0
    class StepInterceptor:
        def __init__(self, name):
            self.name = name
            self.step_latents = {}
        
        def callback(self, step_i, denoised, x, total_steps):
            self.step_latents[step_i] = denoised.clone().detach()
            _logger.info(f"⚡ [split-q][{self.name}] Captured step {step_i}/{total_steps}")
    
    # Execute model_0 (cuda:0)
    _logger.info("⚡ [split-q][Sampler] Executing model_0 on cuda:0")
    interceptor_0 = StepInterceptor("model_0")
    samples_0 = comfy.sample.sample(
        model_primary, noise, steps, cfg, sampler_name, scheduler,
        positive, negative, latent_image,
        denoise=denoise, disable_noise=False, start_step=None, last_step=None,
        force_full_denoise=False, noise_mask=noise_mask,
        callback=interceptor_0.callback, disable_pbar=disable_pbar, seed=seed
    )
    
    # Execute model_1 (cuda:1)
    _logger.info("⚡ [split-q][Sampler] Executing model_1 on cuda:1")
    interceptor_1 = StepInterceptor("model_1")
    samples_1 = comfy.sample.sample(
        model_secondary, noise, steps, cfg, sampler_name, scheduler,
        positive, negative, latent_image,
        denoise=denoise, disable_noise=False, start_step=None, last_step=None,
        force_full_denoise=False, noise_mask=noise_mask,
        callback=interceptor_1.callback, disable_pbar=disable_pbar, seed=seed
    )
    
    # Interleaved validation
    _logger.info("⚡ [split-q][Sampler] Starting interleaved validation (0/0/1/1/2/2/.../29/29)")
    for step_i in range(steps):
        # Validate model_0 step
        _logger.info(f"⚡ [split-q][Sampler] Validating model_0 step {step_i}")
        validate_against_golden(step_i, interceptor_0.step_latents[step_i], label="golden", halt_on_mismatch=True)
        
        # Validate model_1 step
        _logger.info(f"⚡ [split-q][Sampler] Validating model_1 step {step_i}")
        validate_against_golden(step_i, interceptor_1.step_latents[step_i], label="golden", halt_on_mismatch=True)
        
        _logger.info(f"⚡ [split-q][Sampler] ✅ Step {step_i}: BOTH models match golden")
    
    # Return model_0 result
    out = latent.copy()
    out["samples"] = samples_0
    
    _logger.info("⚡ [split-q][Sampler] ✅ All 60 validations passed (30 steps × 2 models)")
    return (out,), None
