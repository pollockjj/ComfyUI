"""
Split-Q ksampler - direct copy of common_ksampler for dual-model serial execution.
"""

import torch
import logging
import comfy.sample
import comfy.utils
import latent_preview
from comfy.split_q.split_q_sample import split_q_sample

_logger = logging.getLogger(__name__)


def split_q_ksampler(model_0, model_1, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise=1.0, disable_noise=False, start_step=None, last_step=None, force_full_denoise=False):
    _logger.info("⚡ [split-q][split_q_ksampler] ENTRY: model_0=%s model_1=%s seed=%d steps=%d sampler=%s", 
                 model_0.__class__.__name__, model_1.__class__.__name__, seed, steps, sampler_name)
    
    latent_image = latent["samples"]
    latent_image_0 = latent_image_1 = comfy.sample.fix_empty_latent_channels(model_0, latent_image)
    
    _logger.info("⚡ [split-q][split_q_ksampler] Replicated latents: latent_0.shape=%s latent_1.shape=%s", 
                 latent_image_0.shape, latent_image_1.shape)

    if disable_noise:
        noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
    else:
        batch_inds = latent["batch_index"] if "batch_index" in latent else None
        noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)

    noise_mask = None
    if "noise_mask" in latent:
        noise_mask = latent["noise_mask"]

    callback = latent_preview.prepare_callback(model_0, steps)
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
    
    samples_0 = comfy.sample.sample(model_0, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image_0,
                                  denoise=denoise, disable_noise=disable_noise, start_step=start_step, last_step=last_step,
                                  force_full_denoise=force_full_denoise, noise_mask=noise_mask, sigmas=None, callback=callback, disable_pbar=disable_pbar, seed=seed)

    samples_1 = split_q_sample(model_1, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image_1,
                                  denoise=denoise, disable_noise=disable_noise, start_step=start_step, last_step=last_step,
                                  force_full_denoise=force_full_denoise, noise_mask=noise_mask, sigmas=None, callback=callback, disable_pbar=disable_pbar, seed=seed)
    
    out_0 = latent.copy()
    out_0["samples"] = samples_0
    out_split_0 = latent.copy()
    out_split_0["samples"] = samples_0
    out_split_1 = latent.copy()
    out_split_1["samples"] = samples_1
    
    _logger.info("⚡ [split-q][split_q_ksampler] RETURN: control.shape=%s split_0.shape=%s split_1.shape=%s", 
                 samples_0.shape, samples_0.shape, samples_1.shape)
    
    return (out_split_0, out_split_1)
