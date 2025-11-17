"""
Split-Q KSampler node mirroring the built-in sampler but with dual model inputs.
"""

import logging

import comfy.samplers
from nodes import common_ksampler


def _summarize_model_options(model):
	options = getattr(model, "model_options", None)
	if not isinstance(options, dict):
		return ()
	summary = []
	for key, value in options.items():
		summary.append((key, type(value).__name__))
	return tuple(sorted(summary))


def _model_signature(model):
	"""Capture enough attributes to detect mismatched replicas."""
	if model is None:
		return None
	return (
		getattr(model, "size", None),
		getattr(model, "patches_uuid", None),
		model.__class__.__name__,
		_summarize_model_options(model),
	)


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
		sig0 = _model_signature(model_0)
		sig1 = _model_signature(model_1)
		if sig0 != sig1:
			logging.warning(
				"⚡ [split-q][KSamplerSplitQ] model replicas differ:\n"
				"    model_0=%s\n"
				"    model_1=%s",
				sig0,
				sig1,
			)
		else:
			logging.info(
				"⚡ [split-q][KSamplerSplitQ] model replicas match:\n"
				"    model_0=%s\n"
				"    model_1=%s",
				sig0,
				sig1,
			)

		return common_ksampler(model_0, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise)


NODE_CLASS_MAPPINGS = {
	"KSamplerSplitQ": KSamplerSplitQ,
}

NODE_DISPLAY_NAME_MAPPINGS = {
	"KSamplerSplitQ": "KSampler (Split-Q)",
}
