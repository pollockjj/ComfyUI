"""
Split-Q sample - Dual-GPU parallel sampling with model replication.
"""

import copy
import torch
import logging
import gc
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


def split_q_sample(model_0, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=1.0, disable_noise=False, start_step=None, last_step=None, force_full_denoise=False, noise_mask=None, sigmas=None, callback=None, disable_pbar=False, seed=None):
    _logger.info("⚡ [split-q][split_q_sample] ENTRY: model_0=%s steps=%d sampler=%s", 
                 model_0.__class__.__name__, steps, sampler_name)
    
    _logger.info("🔥 [split-q] Using pre-configured replica from SplitQAttentionConfig")
    model_1_to_clone = model_0.model.split_q_replica
    device_1 = model_0.model.split_q_device_replica
    
    model_1_clone = _clone_model_to_device(model_1_to_clone, device_1)
    
    sampler_0 = comfy.samplers.KSampler(model_0, steps=steps, device=model_0.load_device, sampler=sampler_name, scheduler=scheduler, denoise=denoise, model_options=model_0.model_options)
    samples_0 = sampler_0.sample(noise, positive, negative, cfg=cfg, latent_image=latent_image, start_step=start_step, last_step=last_step, force_full_denoise=force_full_denoise, denoise_mask=noise_mask, sigmas=sigmas, callback=callback, disable_pbar=disable_pbar, seed=seed)
    samples_0 = samples_0.to(comfy.model_management.intermediate_device())
    
    sampler_1 = comfy.samplers.KSampler(model_1_clone, steps=steps, device=model_1_clone.load_device, sampler=sampler_name, scheduler=scheduler, denoise=denoise, model_options=model_1_clone.model_options)
    samples_1 = sampler_1.sample(noise, positive, negative, cfg=cfg, latent_image=latent_image, start_step=start_step, last_step=last_step, force_full_denoise=force_full_denoise, denoise_mask=noise_mask, sigmas=sigmas, callback=callback, disable_pbar=disable_pbar, seed=seed)
    samples_1 = samples_1.to(comfy.model_management.intermediate_device())
    
    _logger.info("⚡ [split-q][split_q_sample] RETURN: samples_0.shape=%s samples_1.shape=%s", 
                 samples_0.shape, samples_1.shape)
    
    return samples_0, samples_1
