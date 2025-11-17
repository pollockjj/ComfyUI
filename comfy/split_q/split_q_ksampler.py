"""
Split-Q ksampler - direct copy of common_ksampler for dual-model serial execution.
"""

import torch
import logging
import comfy.sample
import comfy.utils
import latent_preview

_logger = logging.getLogger(__name__)


def split_q_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise=1.0, disable_noise=False, start_step=None, last_step=None, force_full_denoise=False):
    _logger.info("⚡ [split-q][split_q_ksampler] ENTRY: model=%s seed=%d steps=%d sampler=%s", 
                 model.__class__.__name__, seed, steps, sampler_name)
    
    latent_image = latent["samples"]
    latent_image = comfy.sample.fix_empty_latent_channels(model, latent_image)

    if disable_noise:
        noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
    else:
        batch_inds = latent["batch_index"] if "batch_index" in latent else None
        noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)

    noise_mask = None
    if "noise_mask" in latent:
        noise_mask = latent["noise_mask"]

    callback = latent_preview.prepare_callback(model, steps)
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
    samples = comfy.sample.sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
                                  denoise=denoise, disable_noise=disable_noise, start_step=start_step, last_step=last_step,
                                  force_full_denoise=force_full_denoise, noise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=seed)
    out = latent.copy()
    out["samples"] = samples
    
    _logger.info("⚡ [split-q][split_q_ksampler] RETURN: samples.shape=%s samples.device=%s", 
                 samples.shape, samples.device)
    
    return (out, )
