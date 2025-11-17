"""
Split-Q sample - EXACT copy of comfy.sample.sample for validation.
"""

import logging
import comfy.samplers
import comfy.model_management

_logger = logging.getLogger(__name__)


def split_q_sample(model_0, model_1, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=1.0, disable_noise=False, start_step=None, last_step=None, force_full_denoise=False, noise_mask=None, sigmas=None, callback=None, disable_pbar=False, seed=None):
    _logger.info("⚡ [split-q][split_q_sample] ENTRY: model_0=%s model_1=%s steps=%d sampler=%s", 
                 model_0.__class__.__name__, model_1.__class__.__name__, steps, sampler_name)
    
    # Model 0: complete 0-29
    sampler_0 = comfy.samplers.KSampler(model_0, steps=steps, device=model_0.load_device, sampler=sampler_name, scheduler=scheduler, denoise=denoise, model_options=model_0.model_options)
    samples_0 = sampler_0.sample(noise, positive, negative, cfg=cfg, latent_image=latent_image, start_step=start_step, last_step=last_step, force_full_denoise=force_full_denoise, denoise_mask=noise_mask, sigmas=sigmas, callback=callback, disable_pbar=disable_pbar, seed=seed)
    samples_0 = samples_0.to(comfy.model_management.intermediate_device())
    
    # Model 1: complete 0-29 (after model_0 finishes)
    sampler_1 = comfy.samplers.KSampler(model_1, steps=steps, device=model_1.load_device, sampler=sampler_name, scheduler=scheduler, denoise=denoise, model_options=model_1.model_options)
    samples_1 = sampler_1.sample(noise, positive, negative, cfg=cfg, latent_image=latent_image, start_step=start_step, last_step=last_step, force_full_denoise=force_full_denoise, denoise_mask=noise_mask, sigmas=sigmas, callback=callback, disable_pbar=disable_pbar, seed=seed)
    samples_1 = samples_1.to(comfy.model_management.intermediate_device())
    
    _logger.info("⚡ [split-q][split_q_sample] RETURN: samples_0.shape=%s samples_1.shape=%s", 
                 samples_0.shape, samples_1.shape)
    
    return samples_0, samples_1
