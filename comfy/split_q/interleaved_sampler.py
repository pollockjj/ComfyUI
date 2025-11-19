"""Interleaved Split-Q sampler wrapping k-diffusion samplers for dual-GPU execution."""

import logging
import torch
import copy

_logger = logging.getLogger(__name__)


class InterleavedSplitQSampler:
    """Wraps a k-diffusion sampler to execute on two models in parallel."""
    
    def __init__(self, base_sampler, cfg_guider_primary, cfg_guider_secondary, loaded_models_secondary):
        self.base_sampler = base_sampler
        self.cfg_guider_primary = cfg_guider_primary
        self.cfg_guider_secondary = cfg_guider_secondary
        self.loaded_models_secondary = loaded_models_secondary
        
    def sample(self, model_wrap, sigmas, extra_args, callback, noise, 
               latent_image=None, denoise_mask=None, disable_pbar=False):
        """Execute sampling on both models and validate byte-identical outputs."""
        
        # Clone inputs for independent trajectories
        device_0 = self.cfg_guider_primary.model_patcher.load_device
        device_1 = self.cfg_guider_secondary.model_patcher.load_device
        
        noise_0 = noise.clone().to(device_0)
        noise_1 = noise.clone().to(device_1)
        
        latent_0 = latent_image.clone().to(device_0) if latent_image is not None else None
        latent_1 = latent_image.clone().to(device_1) if latent_image is not None else None
        
        _logger.info("⚡ [split-q][InterleavedSampler] executing model_0 on %s", device_0)
        samples_0 = self.base_sampler.sample(
            model_wrap, sigmas, extra_args, callback,
            noise_0, latent_0, denoise_mask, disable_pbar
        )
        
        _logger.info("⚡ [split-q][InterleavedSampler] executing model_1 on %s", device_1)
        # Pass CFGGuider directly to sampler, not the wrapper
        # The sampler will create its own KSamplerX0Inpaint wrapper
        samples_1 = self.base_sampler.sample(
            self.cfg_guider_secondary, sigmas.clone().to(device_1), extra_args, callback,
            noise_1, latent_1, denoise_mask, disable_pbar
        )
        
        # Validate byte-identical
        samples_0_cpu = samples_0.to('cpu')
        samples_1_cpu = samples_1.to('cpu')
        
        if not torch.equal(samples_0_cpu, samples_1_cpu):
            delta = (samples_0_cpu - samples_1_cpu).abs().max().item()
            _logger.error(
                "⚡ [split-q][InterleavedSampler][FAIL] outputs diverged: max_abs_diff=%.6f",
                delta,
            )
            raise ValueError("Split-Q interleaved replicas produced different latents")
        
        _logger.info("⚡ [split-q][InterleavedSampler][PASS] outputs are byte-identical")
        return samples_0
    
    def _create_secondary_wrap(self, model_wrap_primary):
        """Create a copy of model_wrap pointing to the secondary model."""
        # This is a shallow copy - we only need to swap the inner_model reference
        model_wrap_secondary = copy.copy(model_wrap_primary)
        
        # Replace inner_model with secondary CFGGuider
        # The CFGGuider holds the actual ModelPatcher reference
        inner_guider_primary = model_wrap_primary.inner_model
        
        # Create new CFGGuider for secondary model
        import comfy.samplers
        secondary_guider = comfy.samplers.CFGGuider(self.model_secondary)
        
        # Copy conds from primary
        secondary_guider.original_conds = {}
        for k in inner_guider_primary.original_conds:
            secondary_guider.original_conds[k] = inner_guider_primary.original_conds[k][:]
        
        secondary_guider.cfg = inner_guider_primary.cfg
        
        model_wrap_secondary.inner_model = secondary_guider
        
        return model_wrap_secondary
