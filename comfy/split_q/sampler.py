"""Utilities for Split-Q sampling orchestration.

These helpers keep both WAN replicas in lockstep while we experiment with
serial and (eventually) parallel execution strategies.  The functions here
are intentionally thin wrappers around ``nodes.common_ksampler`` so they stay
in sync with the canonical sampler behavior.
"""

from __future__ import annotations

import logging
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
    **kwargs,
):
    """Run both replicas sequentially and verify byte-identical latents.

    Parameters mirror ``nodes.common_ksampler``.  ``kwargs`` are forwarded so the
    helper remains drop-in compatible with future options.
    """

    _logger.info("⚡ [split-q][Sampler] serial execution pass=1/2")
    result_primary = _run_single(
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
        **kwargs,
    )

    _logger.info("⚡ [split-q][Sampler] serial execution pass=2/2")
    result_secondary = _run_single(
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
        **kwargs,
    )

    samples_primary = _extract_samples(result_primary)
    samples_secondary = _extract_samples(result_secondary)

    if not torch.equal(samples_primary, samples_secondary):
        delta = (samples_primary - samples_secondary).abs().max().item()
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
    """Placeholder for future interleaved implementation."""
    _logger.info("⚡ [split-q][Sampler] interleaved mode not yet implemented, falling back to serial")
    return serial_split_q_sample(
        model_primary,
        model_secondary,
        seed=seed,
        steps=steps,
        cfg=cfg,
        sampler_name=sampler_name,
        scheduler=scheduler,
        positive=positive,
        negative=negative,
        latent=latent,
        denoise=denoise,
        **kwargs,
    )
