"""
Split-Q sample - EXACT copy of comfy.sample.sample for validation.
"""

import copy
import torch
import logging
import comfy.samplers
import comfy.model_management

_logger = logging.getLogger(__name__)


def split_q_sample(model_0, model_1, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=1.0, disable_noise=False, start_step=None, last_step=None, force_full_denoise=False, noise_mask=None, sigmas=None, callback=None, disable_pbar=False, seed=None):
    _logger.info("⚡ [split-q][split_q_sample] ENTRY: model_0=%s model_1=%s steps=%d sampler=%s", 
                 model_0.__class__.__name__, model_1.__class__.__name__, steps, sampler_name)
    
    # === SLICE 1: Pre-Clone Instrumentation ===
    _logger.info("⚡ [split-q][clone] PRE-CLONE: model_0.device=%s model_1.device=%s", 
                 model_0.load_device, model_1.load_device)
    _logger.info("⚡ [split-q][clone] PRE-CLONE: model_1.model id=%d", id(model_1.model))
    _logger.info("⚡ [split-q][clone] PRE-CLONE: patches=%d attachments=%d", 
                 len(model_1.patches), len(model_1.attachments))
    
    # === SLICE 2: VRAM Safety Check ===
    device_1 = torch.device('cuda:1')
    
    # Calculate VRAM needed (parameters + buffers)
    vram_needed = sum(p.numel() * p.element_size() for p in model_1.model.parameters()) / 1e9
    vram_needed += sum(b.numel() * b.element_size() for b in model_1.model.buffers()) / 1e9
    
    # Check available VRAM with 1.5x safety margin
    vram_free = torch.cuda.mem_get_info(1)[0] / 1e9
    _logger.info("⚡ [split-q][clone] VRAM check: need=%.2fGB free=%.2fGB on %s", 
                 vram_needed, vram_free, device_1)
    
    if vram_free < vram_needed * 1.5:
        raise RuntimeError(f"⚡ [split-q][clone][FATAL] Insufficient VRAM on {device_1}: "
                          f"need {vram_needed:.2f}GB, have {vram_free:.2f}GB")
    
    # === SLICE 3: Clone and Deep Copy ===
    # Clone metadata wrapper
    model_1_clone = model_1.clone()
    _logger.info("⚡ [split-q][clone] Metadata clone created, patches=%d attachments=%d", 
                 len(model_1_clone.patches), len(model_1_clone.attachments))
    
    # Deep copy UNet directly to cuda:1 (single operation)
    _logger.info("⚡ [split-q][clone] Deep copying UNet to %s...", device_1)
    vram_before = torch.cuda.memory_allocated(1) / 1e9
    
    try:
        unet_replica = copy.deepcopy(model_1_clone.model).to(device_1)
        torch.cuda.synchronize(device_1)
    except RuntimeError as e:
        _logger.error("⚡ [split-q][clone][FATAL] Deep copy failed: %s", e)
        raise
    
    vram_after = torch.cuda.memory_allocated(1) / 1e9
    _logger.info("⚡ [split-q][clone] Deep copy complete: VRAM delta=%.2fGB", vram_after - vram_before)
    
    # === SLICE 4: Verify and Reassign ===
    # Verify replica is fully on cuda:1
    params_on_wrong_device = [name for name, p in unet_replica.named_parameters() if p.device != device_1]
    if params_on_wrong_device:
        raise RuntimeError(f"⚡ [split-q][clone][FATAL] Parameters not on {device_1}: {params_on_wrong_device[:5]}")
    
    _logger.info("⚡ [split-q][clone] Device verification passed: all parameters on %s", device_1)
    
    # Reassign replica UNet and update device
    model_1_clone.model = unet_replica
    model_1_clone.load_device = device_1
    
    # === SLICE 5: Post-Clone Validation ===
    _logger.info("⚡ [split-q][clone] POST-CLONE: model_1_clone.device=%s", model_1_clone.load_device)
    _logger.info("⚡ [split-q][clone] POST-CLONE: model_1_clone.model id=%d (original=%d)", 
                 id(model_1_clone.model), id(model_1.model))
    
    # Verify independence
    if id(model_1.model) == id(model_1_clone.model):
        raise RuntimeError("⚡ [split-q][clone][FATAL] Clone failed: UNet objects are identical")
    
    _logger.info("⚡ [split-q][clone] Independence verified: different UNet objects")
    
    # Model 0: complete 0-29
    sampler_0 = comfy.samplers.KSampler(model_0, steps=steps, device=model_0.load_device, sampler=sampler_name, scheduler=scheduler, denoise=denoise, model_options=model_0.model_options)
    samples_0 = sampler_0.sample(noise, positive, negative, cfg=cfg, latent_image=latent_image, start_step=start_step, last_step=last_step, force_full_denoise=force_full_denoise, denoise_mask=noise_mask, sigmas=sigmas, callback=callback, disable_pbar=disable_pbar, seed=seed)
    samples_0 = samples_0.to(comfy.model_management.intermediate_device())
    
    # Model 1: complete 0-29 (after model_0 finishes)
    sampler_1 = comfy.samplers.KSampler(model_1_clone, steps=steps, device=model_1_clone.load_device, sampler=sampler_name, scheduler=scheduler, denoise=denoise, model_options=model_1_clone.model_options)
    samples_1 = sampler_1.sample(noise, positive, negative, cfg=cfg, latent_image=latent_image, start_step=start_step, last_step=last_step, force_full_denoise=force_full_denoise, denoise_mask=noise_mask, sigmas=sigmas, callback=callback, disable_pbar=disable_pbar, seed=seed)
    samples_1 = samples_1.to(comfy.model_management.intermediate_device())
    
    _logger.info("⚡ [split-q][split_q_sample] RETURN: samples_0.shape=%s samples_1.shape=%s", 
                 samples_0.shape, samples_1.shape)
    
    return samples_0, samples_1
