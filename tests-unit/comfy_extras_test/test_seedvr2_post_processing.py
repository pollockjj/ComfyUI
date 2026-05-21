import inspect
from unittest.mock import patch

import torch

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

from comfy_extras import nodes_seedvr  # noqa: E402


def _schema_ids(items):
    return [item.id for item in items]


def test_seedvr2_post_processing_schema():
    schema = nodes_seedvr.SeedVR2PostProcessing.define_schema()

    assert _schema_ids(schema.inputs) == ["decoded", "reference", "method", "resolution"]
    assert schema.inputs[2].options == ["lab", "none"]
    assert schema.inputs[2].default == "lab"
    assert schema.inputs[3].default == 1280
    assert schema.outputs[0].get_io_type() == "IMAGE"


def test_seedvr2_post_processing_lab_uses_explicit_decoded_and_reference():
    decoded = torch.full((1, 3, 9, 11, 3), 0.25)
    reference = torch.full((1, 2, 8, 10, 3), 0.75)
    calls = []

    def _lab(content, style):
        calls.append((content.clone(), style.clone()))
        return torch.zeros_like(content)

    with patch.object(nodes_seedvr, "lab_color_transfer", _lab):
        output = nodes_seedvr.SeedVR2PostProcessing.execute(decoded, reference, "lab", 8).result[0]

    assert tuple(output.shape) == (1, 2, 8, 10, 3)
    assert torch.equal(output, torch.full_like(output, 0.5))
    assert calls[0][0].shape == (2, 3, 8, 10)
    assert calls[0][1].shape == (2, 3, 8, 10)
    assert torch.equal(calls[0][0], torch.full_like(calls[0][0], -0.5))
    assert torch.allclose(calls[0][1], torch.full_like(calls[0][1], 0.5))


def test_seedvr2_post_processing_raw_conversion_does_not_probe_full_tensor_range():
    source = inspect.getsource(nodes_seedvr.SeedVR2PostProcessing._to_seedvr2_raw)

    assert ".amin" not in source
    assert ".item" not in source


def test_seedvr2_post_processing_lab_resizes_full_reference_frame():
    decoded = torch.full((1, 2, 8, 10, 3), 0.25)
    reference = torch.full((1, 2, 16, 20, 3), 0.75)
    resize_calls = []
    lab_calls = []

    def _resize(images, size, interpolation=None, antialias=None):
        resize_calls.append((images.clone(), size, interpolation, antialias))
        return torch.full((2, 3, size[0], size[1]), 0.5)

    def _lab(content, style):
        lab_calls.append((content.clone(), style.clone()))
        return torch.zeros_like(content)

    with patch.object(nodes_seedvr.TVF, "resize", _resize):
        with patch.object(nodes_seedvr, "lab_color_transfer", _lab):
            output = nodes_seedvr.SeedVR2PostProcessing.execute(decoded, reference, "lab", 8).result[0]

    assert tuple(output.shape) == (1, 2, 8, 10, 3)
    assert torch.equal(output, torch.full_like(output, 0.5))
    assert resize_calls[0][0].shape == (2, 3, 16, 20)
    assert resize_calls[0][1] == (8, 10)
    assert lab_calls[0][1].shape == (2, 3, 8, 10)
    assert torch.equal(lab_calls[0][1], torch.zeros_like(lab_calls[0][1]))


def test_seedvr2_post_processing_none_trims_and_crops_without_color_correction():
    decoded = torch.arange(1 * 3 * 9 * 11 * 3, dtype=torch.float32).reshape(1, 3, 9, 11, 3)
    reference = torch.zeros(1, 2, 8, 10, 3)

    with patch.object(nodes_seedvr, "lab_color_transfer") as lab:
        output = nodes_seedvr.SeedVR2PostProcessing.execute(decoded, reference, "none", 8).result[0]

    assert lab.call_count == 0
    assert tuple(output.shape) == (1, 2, 8, 10, 3)
    assert torch.equal(output, decoded[:, :2, :8, :10, :])


def test_seedvr2_post_processing_none_preserves_decoded_spatial_size_when_reference_is_larger():
    decoded = torch.arange(1 * 3 * 8 * 10 * 3, dtype=torch.float32).reshape(1, 3, 8, 10, 3)
    reference = torch.zeros(1, 2, 16, 20, 3)

    with patch.object(nodes_seedvr, "lab_color_transfer") as lab:
        output = nodes_seedvr.SeedVR2PostProcessing.execute(decoded, reference, "none").result[0]

    assert lab.call_count == 0
    assert tuple(output.shape) == (1, 2, 8, 10, 3)
    assert torch.equal(output, decoded[:, :2, :, :, :])


def test_seedvr2_post_processing_preserves_requested_resize_when_reference_is_smaller():
    decoded = torch.ones((1, 1, 720, 1280, 3), dtype=torch.float32)
    reference = torch.ones((1, 1, 360, 640, 3), dtype=torch.float32)

    output = nodes_seedvr.SeedVR2PostProcessing.execute(decoded, reference, "none", 720).result[0]

    assert tuple(output.shape) == (1, 1, 720, 1280, 3)


def test_seedvr2_post_processing_crops_large_raw_reference_to_visible_resize():
    decoded = torch.ones((1, 1, 128, 160, 3), dtype=torch.float32)
    reference = torch.ones((1, 1, 480, 640, 3), dtype=torch.float32)

    output = nodes_seedvr.SeedVR2PostProcessing.execute(decoded, reference, "none", 120).result[0]

    assert tuple(output.shape) == (1, 1, 120, 160, 3)


def test_seedvr2_post_processing_crops_to_explicit_visible_resize_width():
    decoded = torch.ones((1, 1, 128, 224, 3), dtype=torch.float32)
    reference = torch.ones((1, 1, 1080, 1920, 3), dtype=torch.float32)

    output = nodes_seedvr.SeedVR2PostProcessing.execute(decoded, reference, "none", 120).result[0]

    assert tuple(output.shape) == (1, 1, 120, 212, 3)


def test_seedvr2_post_processing_none_preserves_black_bottom_row_content():
    decoded = torch.ones((1, 2, 8, 10, 3), dtype=torch.float32)
    reference = torch.ones((1, 2, 8, 10, 3), dtype=torch.float32)
    reference[:, :, -1, :, :] = -1.0

    output = nodes_seedvr.SeedVR2PostProcessing.execute(decoded, reference, "none").result[0]

    assert tuple(output.shape) == (1, 2, 8, 10, 3)
    assert torch.equal(output, decoded)


def test_seedvr2_post_processing_none_preserves_black_right_column_content():
    decoded = torch.ones((1, 2, 8, 10, 3), dtype=torch.float32)
    reference = torch.ones((1, 2, 8, 10, 3), dtype=torch.float32)
    reference[:, :, :, -1, :] = -1.0

    output = nodes_seedvr.SeedVR2PostProcessing.execute(decoded, reference, "none").result[0]

    assert tuple(output.shape) == (1, 2, 8, 10, 3)
    assert torch.equal(output, decoded)
