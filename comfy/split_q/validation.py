"""
Split-Q model validation utilities.
"""

def summarize_model_options(model):
	options = getattr(model, "model_options", None)
	if not isinstance(options, dict):
		return ()
	summary = []
	for key, value in options.items():
		summary.append((key, type(value).__name__))
	return tuple(sorted(summary))


def model_uuid(model):
	"""Return the unique patches UUID used to differentiate replicas."""
	return getattr(model, "patches_uuid", None)


def collect_model_metadata(model):
	inner = getattr(model, "model", None)
	inner_class = inner.__class__.__name__ if inner is not None else None
	inner_shapes = first_two_tensor_shapes(inner)
	return {
		"class": model.__class__.__name__ if model else None,
		"model_type": getattr(model, "model_type", None),
		"size": getattr(model, "size", None),
		"dtype": getattr(model, "dtype", None),
		"inner_class": inner_class,
		"inner_tensor0_shape": inner_shapes[0],
		"inner_tensor1_shape": inner_shapes[1],
		"patches_uuid": model_uuid(model),
		"transformer_options": summarize_model_options(model),
	}


def first_two_tensor_shapes(inner):
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


def format_metadata_table(meta0, meta1):
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


def attach_peer_reference(primary_model, secondary_model, attr_name):
	import logging
	inner = getattr(primary_model, "model", None)
	if inner is None:
		logging.warning("⚡ [split-q][KSamplerSplitQ] inner model missing for peer attachment: %s", attr_name)
		return
	setattr(inner, attr_name, secondary_model)
	logging.info("⚡ [split-q][KSamplerSplitQ] attached %s -> ModelPatcher(id=%s)", attr_name, hex(id(secondary_model)))
