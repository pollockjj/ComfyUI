"""
Split-Q sample - EXACT copy of comfy.sample.sample for validation.
"""

import logging
import comfy.samplers
import comfy.model_management

_logger = logging.getLogger(__name__)


def split_q_sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=1.0, disable_noise=False, start_step=None, last_step=None, force_full_denoise=False, noise_mask=None, sigmas=None, callback=None, disable_pbar=False, seed=None):
    _logger.info("⚡ [split-q][split_q_sample] ENTRY: model=%s steps=%d sampler=%s", 
                 model.__class__.__name__, steps, sampler_name)
    
    sampler = comfy.samplers.KSampler(model, steps=steps, device=model.load_device, sampler=sampler_name, scheduler=scheduler, denoise=denoise, model_options=model.model_options)

    samples = sampler.sample(noise, positive, negative, cfg=cfg, latent_image=latent_image, start_step=start_step, last_step=last_step, force_full_denoise=force_full_denoise, denoise_mask=noise_mask, sigmas=sigmas, callback=callback, disable_pbar=disable_pbar, seed=seed)
    samples = samples.to(comfy.model_management.intermediate_device())
    
    _logger.info("⚡ [split-q][split_q_sample] RETURN: samples.shape=%s", samples.shape)
    
    return samples
