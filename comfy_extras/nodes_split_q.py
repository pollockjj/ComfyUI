"""
Split-Q KSampler node mirroring the built-in sampler but with dual model inputs.
"""

import logging
import torch
import comfy.samplers
from comfy.split_q.validation import collect_model_metadata, format_metadata_table, attach_peer_reference
from nodes import common_ksampler
from comfy.split_q.split_q_ksampler import split_q_ksampler


class KSamplerSplitQ:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_0": ("MODEL", {"tooltip": "Primary model used for sampling (cuda:0)."}),
                "model_1": ("MODEL", {"tooltip": "Replica model (cuda:1). Currently validated only."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True, "tooltip": "Random seed used for noise generation."}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000, "tooltip": "Number of denoising steps."}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01, "tooltip": "Classifier-Free Guidance scale."}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"tooltip": "Sampling algorithm."}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"tooltip": "Noise schedule."}),
                "positive": ("CONDITIONING", {"tooltip": "Positive conditioning to include."}),
                "negative": ("CONDITIONING", {"tooltip": "Negative conditioning to exclude."}),
                "latent_image": ("LATENT", {"tooltip": "Latent to denoise."}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Denoising strength."}),
        }
    }

    RETURN_TYPES = ("LATENT",)
    OUTPUT_TOOLTIPS = ("The denoised latent computed with model_0.",)
    FUNCTION = "sample"
    CATEGORY = "sampling"
    DESCRIPTION = "Standard KSampler behavior with Split-Q dual-model validation."

    def sample(self, model_0, model_1, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=1.0):
        
        logging.info("⚡ [split-q][KSamplerSplitQ] First standard model run, then return from split_q_ksampler.")
        result_0 = common_ksampler(model_0, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise)
        result_1 = split_q_ksampler(model_1, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise)

        samples_0 = result_0[0]["samples"].to("cpu")
        samples_1 = result_1[0]["samples"].to("cpu")
        if not torch.equal(samples_0, samples_1):
            delta = (samples_0 - samples_1).abs().max().item()
            logging.info("⚡ [split-q][KSamplerSplitQ][FAIL] replica outputs diverged: max_abs_diff=%.6f", delta)
            raise ValueError("Split-Q replicas produced different latents")
        logging.info("⚡ [split-q][KSamplerSplitQ][PASS] replica outputs are byte-identical")
        
        return result_1


NODE_CLASS_MAPPINGS = {
    "KSamplerSplitQ": KSamplerSplitQ,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KSamplerSplitQ": "KSampler (Split-Q)",
}
