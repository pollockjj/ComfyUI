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


def execute_single_step(
    model,
    sampler,
    current_latent,
    initial_latent_image,
    step_index,
    total_steps,
    cfg,
    scheduler,
    positive,
    negative,
    seed,
    denoise=1.0,
):
    """
    Execute a single denoising step using sigma scheduling.
    
    CRITICAL: The sampler object MUST be persistent across all steps to maintain state.
    Do NOT create a new sampler inside this function.
    
    Args:
        model: ComfyUI model (ModelPatcher) on specific GPU
        sampler: Persistent sampler object (KSAMPLER instance, created ONCE outside loop)
        current_latent: Latent tensor from previous step (or initial noise for step 0)
        initial_latent_image: Original starting latent (unchanged across all steps)
        step_index: Current step index (0 to total_steps-1)
        total_steps: Total number of denoising steps
        cfg: CFG scale
        scheduler: Scheduler name
        positive: Positive conditioning
        negative: Negative conditioning
        seed: Random seed
        denoise: Denoising strength (default 1.0)
    
    Returns:
        Denoised latent tensor for this step (output becomes input to next step)
    """
    import comfy.samplers
    import comfy.sample
    
    device = current_latent.device
    
    _logger.info(
        f"⚡ [split-q][SingleStep] step={step_index}/{total_steps} "
        f"current_latent.shape={current_latent.shape} device={device}"
    )
    
    # Get full sigma schedule for the total steps
    model_sampling = model.get_model_object("model_sampling")
    sigmas = comfy.samplers.calculate_sigmas(model_sampling, scheduler, total_steps)
    
    # Extract sigma range for this single step [step_index, step_index+1]
    sigma_start = sigmas[step_index]
    sigma_end = sigmas[step_index + 1]
    single_step_sigmas = torch.tensor([sigma_start, sigma_end], device=device)
    
    _logger.info(
        f"⚡ [split-q][SingleStep] step={step_index} "
        f"sigma_start={sigma_start:.6f} sigma_end={sigma_end:.6f}"
    )
    
    # Capture denoised output via callback (matches golden reference capture)
    captured_denoised = {}
    def capture_callback(substep_i, denoised, x, total_substeps):
        # Save denoised prediction (fully denoised estimate at this timestep)
        # NOTE: substep_i is always 0 for single-step execution (only one substep per call)
        captured_denoised[substep_i] = denoised.clone().detach()
        _logger.info(
            f"⚡ [split-q][SingleStep] Captured denoised at substep {substep_i} "
            f"(global step {step_index}), shape={denoised.shape}, device={denoised.device}"
        )
    
    # Execute single step using sample_custom with restricted sigmas
    # Key semantics:
    #   - noise: The current noisy latent being denoised (changes each step)
    #   - latent_image: The original starting latent (constant across all steps)
    #   - callback: Captures denoised prediction (for validation)
    output_latent = comfy.sample.sample_custom(
        model=model,
        noise=current_latent,           # Current step's noisy input
        cfg=cfg,
        sampler=sampler,                # PERSISTENT sampler object (maintains state)
        sigmas=single_step_sigmas,
        positive=positive,
        negative=negative,
        latent_image=initial_latent_image,  # Original latent (unchanged)
        noise_mask=None,
        callback=capture_callback,      # Capture denoised for validation
        disable_pbar=True,  # Disable progress bar for single step
        seed=seed
    )
    
    _logger.info(
        f"⚡ [split-q][SingleStep] step={step_index} complete, "
        f"output_shape={output_latent.shape} device={output_latent.device}"
    )
    
    # Return BOTH the next-step input (output_latent) AND the denoised prediction (for validation)
    # The denoised prediction is what we compare against golden
    # For single-step execution, substep is always 0
    if 0 not in captured_denoised:
        _logger.error(f"⚡ [split-q][SingleStep] FATAL: No denoised captured for step {step_index} (substep 0)")
        _logger.error(f"⚡ [split-q][SingleStep] captured_denoised keys: {list(captured_denoised.keys())}")
        raise ValueError(f"Callback did not capture denoised for step {step_index}")
    
    denoised_prediction = captured_denoised[0]  # Always substep 0 for single-step execution
    
    # CRITICAL: sample_custom moves result to CPU, but we need it on GPU for next step
    # Move output_latent back to the model's device for chaining
    output_latent = output_latent.to(device)
    
    _logger.info(
        f"⚡ [split-q][SingleStep] step={step_index} output moved back to {device} for chaining"
    )
    
    return output_latent, denoised_prediction


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
    """Execute TRUE interleaved sampling: 0/0/1/1/2/2/.../29/29."""
    from comfy.split_q.golden_reference import validate_against_golden
    import comfy.sample
    import comfy.samplers
    
    _logger.info("⚡ [split-q][Sampler] ISOLATED-INTERLEAVED: TRUE interleaved execution starting")
    
    # Prepare initial latent (shared seed, different GPUs)
    latent_image = latent["samples"]
    latent_image = comfy.sample.fix_empty_latent_channels(model_primary, latent_image)
    batch_inds = latent.get("batch_index", None)
    
    # Generate initial noise on CPU first, then clone to each GPU
    initial_noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)
    
    # Clone initial noise to each GPU's VRAM (TOTAL ISOLATION starts here)
    latent_0 = initial_noise.clone().to(model_primary.load_device)
    latent_1 = initial_noise.clone().to(model_secondary.load_device)
    
    _logger.info(
        f"⚡ [split-q][Sampler] Initial latents prepared: "
        f"latent_0.device={latent_0.device} latent_1.device={latent_1.device} "
        f"shape={latent_0.shape}"
    )
    
    # Create PERSISTENT sampler objects (CRITICAL: reused across all steps for state)
    sampler_0 = comfy.samplers.sampler_object(sampler_name)
    sampler_1 = comfy.samplers.sampler_object(sampler_name)
    
    _logger.info(
        f"⚡ [split-q][Sampler] Persistent samplers created: "
        f"sampler_0={type(sampler_0).__name__} sampler_1={type(sampler_1).__name__}"
    )
    
    # Preserve initial latent for latent_image parameter (constant across all steps)
    initial_latent_0 = latent_0.clone()
    initial_latent_1 = latent_1.clone()
    
    # TRUE INTERLEAVED LOOP: Execute one step at a time, alternating GPUs
    for step_i in range(steps):
        _logger.info(f"⚡ [split-q][Sampler] === STEP {step_i}/{steps} ===")
        
        # PHASE 3.0: Test model_0 ONLY first (validate single-step mechanism)
        _logger.info(f"⚡ [split-q][Sampler] Executing model_0 step {step_i}")
        latent_0, denoised_0 = execute_single_step(
            model=model_primary,
            sampler=sampler_0,              # Persistent sampler (maintains state)
            current_latent=latent_0,        # Uses model_0's previous output
            initial_latent_image=initial_latent_0,  # Constant reference
            step_index=step_i,
            total_steps=steps,
            cfg=cfg,
            scheduler=scheduler,
            positive=positive,
            negative=negative,
            seed=seed,
            denoise=denoise,
        )
        
        # Validate model_0 step IMMEDIATELY (using denoised prediction, not noisy output)
        _logger.info(f"⚡ [split-q][Sampler] Validating model_0 step {step_i}")
        validate_against_golden(
            step_i, 
            denoised_0,  # Validate the denoised prediction, not the noisy latent
            label="golden", 
            halt_on_mismatch=True
        )
        _logger.info(f"⚡ [split-q][Sampler] ✅ model_0 step {step_i} PASSED")
        
        # TEMPORARY: Phase 3.1 - Test model_0 through steps 0-2 (chain verification)
        if step_i == 2:
            _logger.warning("⚡ [split-q][Sampler] PHASE 3.1: Exiting after model_0 step 2 (chain test)")
            out = latent.copy()
            out["samples"] = latent_0
            return (out,), None
        
        # Execute step_i on model_1 (cuda:1)
        _logger.info(f"⚡ [split-q][Sampler] Executing model_1 step {step_i}")
        latent_1, denoised_1 = execute_single_step(
            model=model_secondary,
            sampler=sampler_1,              # Persistent sampler (maintains state)
            current_latent=latent_1,        # Uses model_1's previous output
            initial_latent_image=initial_latent_1,  # Constant reference
            step_index=step_i,
            total_steps=steps,
            cfg=cfg,
            scheduler=scheduler,
            positive=positive,
            negative=negative,
            seed=seed,
            denoise=denoise,
        )
        
        # Validate model_1 step IMMEDIATELY (using denoised prediction)
        _logger.info(f"⚡ [split-q][Sampler] Validating model_1 step {step_i}")
        validate_against_golden(
            step_i, 
            denoised_1,  # Validate the denoised prediction
            label="golden", 
            halt_on_mismatch=True
        )
        _logger.info(f"⚡ [split-q][Sampler] ✅ model_1 step {step_i} PASSED")
        
        _logger.info(f"⚡ [split-q][Sampler] ✅ Step {step_i}: BOTH models validated")
    
    # Return model_0 result
    out = latent.copy()
    out["samples"] = latent_0
    
    _logger.info(
        f"⚡ [split-q][Sampler] ✅ All {steps * 2} executions complete "
        f"(interleaved order: 0/0/1/1/2/2/.../29/29)"
    )
    
    return (out,), None
