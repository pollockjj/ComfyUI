"""
Split-Q sample - Dual-GPU parallel sampling with model replication.
"""

import copy
import torch
import logging
import gc
import math
import comfy.samplers
import comfy.model_management

_logger = logging.getLogger(__name__)


def _clone_model_to_device(model_1, device_1):
    vram_needed = sum(p.numel() * p.element_size() for p in model_1.model.parameters()) / 1e9
    vram_needed += sum(b.numel() * b.element_size() for b in model_1.model.buffers()) / 1e9
    vram_free = torch.cuda.mem_get_info(device_1.index)[0] / 1e9
    
    if vram_free < vram_needed * 1.5:
        raise RuntimeError(f"Insufficient VRAM on {device_1}: need {vram_needed:.2f}GB, have {vram_free:.2f}GB")
    
    comfy.model_management.free_memory(vram_needed * 1.5, device_1, keep_loaded=[])
    comfy.model_management.soft_empty_cache()
    
    model_1_clone = model_1.clone()
    unet_replica = copy.deepcopy(model_1_clone.model).to(device_1)
    torch.cuda.synchronize(device_1)
    
    params_on_wrong_device = [name for name, p in unet_replica.named_parameters() if p.device != device_1]
    if params_on_wrong_device:
        raise RuntimeError(f"Parameters not on {device_1}: {params_on_wrong_device[:5]}")
    
    model_1_clone.model = unet_replica
    model_1_clone.load_device = device_1
    
    if id(model_1.model) == id(model_1_clone.model):
        raise RuntimeError("Clone failed: UNet objects are identical")
    
    del model_1
    gc.collect()
    comfy.model_management.soft_empty_cache()
    
    return model_1_clone


class SplitQKSampler:
    SCHEDULERS = comfy.samplers.SCHEDULER_NAMES
    SAMPLERS = comfy.samplers.SAMPLER_NAMES
    DISCARD_PENULTIMATE_SIGMA_SAMPLERS = set(('dpm_2', 'dpm_2_ancestral', 'uni_pc', 'uni_pc_bh2'))

    def __init__(self, model, steps, device, sampler=None, scheduler=None, denoise=None, model_options={}):
        _logger.info("🔥 [SplitQKSampler] __init__ called")
        self.model = model
        self.device = device
        if scheduler not in self.SCHEDULERS:
            scheduler = self.SCHEDULERS[0]
        if sampler not in self.SAMPLERS:
            sampler = self.SAMPLERS[0]
        self.scheduler = scheduler
        self.sampler = sampler
        self.set_steps(steps, denoise)
        self.denoise = denoise
        self.model_options = model_options

    def calculate_sigmas(self, steps):
        sigmas = None

        discard_penultimate_sigma = False
        if self.sampler in self.DISCARD_PENULTIMATE_SIGMA_SAMPLERS:
            steps += 1
            discard_penultimate_sigma = True

        sigmas = comfy.samplers.calculate_sigmas(self.model.get_model_object("model_sampling"), self.scheduler, steps)

        if discard_penultimate_sigma:
            sigmas = torch.cat([sigmas[:-2], sigmas[-1:]])
        return sigmas

    def set_steps(self, steps, denoise=None):
        self.steps = steps
        if denoise is None or denoise > 0.9999:
            self.sigmas = self.calculate_sigmas(steps).to(self.device)
        else:
            if denoise <= 0.0:
                self.sigmas = torch.FloatTensor([])
            else:
                new_steps = int(steps/denoise)
                sigmas = self.calculate_sigmas(new_steps).to(self.device)
                self.sigmas = sigmas[-(steps + 1):]

    def sample(self, noise, positive, negative, cfg, latent_image=None, start_step=None, last_step=None, force_full_denoise=False, denoise_mask=None, sigmas=None, callback=None, disable_pbar=False, seed=None):
        _logger.info("🔥 [SplitQKSampler] sample() called")
        if sigmas is None:
            sigmas = self.sigmas

        if last_step is not None and last_step < (len(sigmas) - 1):
            sigmas = sigmas[:last_step + 1]
            if force_full_denoise:
                sigmas[-1] = 0

        if start_step is not None:
            if start_step < (len(sigmas) - 1):
                sigmas = sigmas[start_step:]
            else:
                if latent_image is not None:
                    return latent_image
                else:
                    return torch.zeros_like(noise)

        sampler = comfy.samplers.sampler_object(self.sampler)

        return comfy.samplers.sample(self.model, noise, positive, negative, cfg, self.device, sampler, sigmas, self.model_options, latent_image=latent_image, denoise_mask=denoise_mask, callback=callback, disable_pbar=disable_pbar, seed=seed)


def split_q_sample(model_0, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=1.0, disable_noise=False, start_step=None, last_step=None, force_full_denoise=False, noise_mask=None, sigmas=None, callback=None, disable_pbar=False, seed=None):
    _logger.info("⚡ [split-q][split_q_sample] ENTRY: model_0=%s steps=%d sampler=%s", 
                 model_0.__class__.__name__, steps, sampler_name)
    
    _logger.info("🔥 [split-q] Using pre-configured replica from SplitQAttentionConfig")
    model_1_to_clone = model_0.model.split_q_replica
    device_1 = model_0.model.split_q_device_replica
    
    model_1 = _clone_model_to_device(model_1_to_clone, device_1)
    
    sampler_0 = comfy.samplers.KSampler(model_0, steps=steps, device=model_0.load_device, sampler=sampler_name, scheduler=scheduler, denoise=denoise, model_options=model_0.model_options)
    sampler_1 = SplitQKSampler(model_1, steps=steps, device=model_1.load_device, sampler=sampler_name, scheduler=scheduler, denoise=denoise, model_options=model_1.model_options)
    
    samples_0 = sampler_0.sample(noise, positive, negative, cfg=cfg, latent_image=latent_image, start_step=start_step, last_step=last_step, force_full_denoise=force_full_denoise, denoise_mask=noise_mask, sigmas=sigmas, callback=callback, disable_pbar=disable_pbar, seed=seed)
    samples_1 = sampler_1.sample(noise, positive, negative, cfg=cfg, latent_image=latent_image, start_step=start_step, last_step=last_step, force_full_denoise=force_full_denoise, denoise_mask=noise_mask, sigmas=sigmas, callback=callback, disable_pbar=disable_pbar, seed=seed)
    
    samples_0 = samples_0.to(comfy.model_management.intermediate_device())
    samples_1 = samples_1.to(comfy.model_management.intermediate_device())
    
    _logger.info("⚡ [split-q][split_q_sample] RETURN: samples_0.shape=%s samples_1.shape=%s", 
                 samples_0.shape, samples_1.shape)
    
    return samples_0, samples_1
