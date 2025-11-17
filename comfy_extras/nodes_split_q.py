"""
Split-Q KSampler node mirroring the built-in sampler but with dual model inputs.
"""

import logging

import torch

import comfy.samplers
from comfy.split_q.sampler import serial_split_q_sample, interleaved_split_q_sample


def _summarize_model_options(model):
	options = getattr(model, "model_options", None)
	if not isinstance(options, dict):
		return ()
	summary = []
	for key, value in options.items():
		summary.append((key, type(value).__name__))
	return tuple(sorted(summary))


def _model_uuid(model):
	"""Return the unique patches UUID used to differentiate replicas."""
	return getattr(model, "patches_uuid", None)


def _collect_model_metadata(model):
	inner = getattr(model, "model", None)
	inner_class = inner.__class__.__name__ if inner is not None else None
	inner_shapes = _first_two_tensor_shapes(inner)
	return {
		"class": model.__class__.__name__ if model else None,
		"model_type": getattr(model, "model_type", None),
		"size": getattr(model, "size", None),
		"dtype": getattr(model, "dtype", None),
		"inner_class": inner_class,
		"inner_tensor0_shape": inner_shapes[0],
		"inner_tensor1_shape": inner_shapes[1],
		"patches_uuid": _model_uuid(model),
		"transformer_options": _summarize_model_options(model),
	}


def _first_two_tensor_shapes(inner):
	shapes = [None, None]
	if inner is None:
		return tuple(shapes)
	if not hasattr(inner, "parameters"):
		return tuple(shapes)
	try:
		for idx, param in enumerate(inner.parameters()):
			shapes[idx] = tuple(param.shape)
			if idx == 1:
				break
	except Exception:
		pass
	return tuple(shapes)


def _format_metadata_table(meta0, meta1):
	rows = [
		("class", meta0["class"], meta1["class"]),
		("model_type", meta0["model_type"], meta1["model_type"]),
		("size", meta0["size"], meta1["size"]),
		("dtype", meta0["dtype"], meta1["dtype"]),
		("inner_class", meta0["inner_class"], meta1["inner_class"]),
		("inner_tensor0_shape", meta0["inner_tensor0_shape"], meta1["inner_tensor0_shape"]),
		("inner_tensor1_shape", meta0["inner_tensor1_shape"], meta1["inner_tensor1_shape"]),
		("patches_uuid", meta0["patches_uuid"], meta1["patches_uuid"]),
		("transformer_options", meta0["transformer_options"], meta1["transformer_options"]),
	]
	lines = ["| field | model_0 | model_1 |", "| --- | --- | --- |"]
	for field, left, right in rows:
		lines.append(f"| {field} | {left} | {right} |")
	return "\n".join(lines)


def _attach_peer_reference(primary_model, secondary_model, attr_name):
	inner = getattr(primary_model, "model", None)
	if inner is None:
		logging.warning("⚡ [split-q][KSamplerSplitQ] inner model missing for peer attachment: %s", attr_name)
		return
	setattr(inner, attr_name, secondary_model)
	logging.info(
		"⚡ [split-q][KSamplerSplitQ] attached %s -> ModelPatcher(id=%s)",
		attr_name,
		hex(id(secondary_model)),
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
		meta0 = _collect_model_metadata(model_0)
		meta1 = _collect_model_metadata(model_1)
		uuid0 = meta0["patches_uuid"]
		uuid1 = meta1["patches_uuid"]
		fields_to_match = (
			"class",
			"model_type",
			"size",
			"dtype",
			"inner_class",
			"inner_tensor0_shape",
			"inner_tensor1_shape",
			"transformer_options",
		)
		mismatch = [(field, meta0[field], meta1[field]) for field in fields_to_match if meta0[field] != meta1[field]]
		if mismatch:
			table = _format_metadata_table(meta0, meta1)
			logging.error(
				"⚡ [split-q][KSamplerSplitQ][FAIL] replica metadata mismatch detected:\n%s",
				table,
			)
			raise ValueError("Split-Q replicas are not identical across required fields")
		if uuid0 == uuid1:
			table = _format_metadata_table(meta0, meta1)
			logging.error(
				"⚡ [split-q][KSamplerSplitQ][FAIL] patches_uuid collision detected (must differ):\n%s",
				table,
			)
			raise ValueError("Split-Q replicas share identical patches_uuid")
		table = _format_metadata_table(meta0, meta1)
		logging.info(
			"⚡ [split-q][KSamplerSplitQ][PASS] replica validation complete:\n%s",
			table,
		)

		_attach_peer_reference(model_0, model_1, "split_q_model_1")
		_attach_peer_reference(model_1, model_0, "split_q_model_0")

		result_0, _ = serial_split_q_sample(
			model_0,
			model_1,
			seed=seed,
			steps=steps,
			cfg=cfg,
			sampler_name=sampler_name,
			scheduler=scheduler,
			positive=positive,
			negative=negative,
			latent=latent_image,
			denoise=denoise,
		)

		return result_0


NODE_CLASS_MAPPINGS = {
	"KSamplerSplitQ": KSamplerSplitQ,
}

NODE_DISPLAY_NAME_MAPPINGS = {
	"KSamplerSplitQ": "KSampler (Split-Q)",
}
