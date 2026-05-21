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

    assert _schema_ids(schema.inputs) == ["decoded", "reference", "method"]
    assert schema.inputs[2].options == ["lab", "none"]
    assert schema.inputs[2].default == "lab"
    assert schema.outputs[0].get_io_type() == "IMAGE"


def test_seedvr2_post_processing_lab_uses_explicit_decoded_and_reference():
    decoded = torch.full((1, 3, 9, 11, 3), 0.25)
    reference = torch.full((1, 2, 8, 10, 3), 0.75)
    calls = []

    def _lab(content, style):
        calls.append((content.clone(), style.clone()))
        return torch.zeros_like(content)

    with patch.object(nodes_seedvr, "lab_color_transfer", _lab):
        output = nodes_seedvr.SeedVR2PostProcessing.execute(decoded, reference, "lab").result[0]

    assert tuple(output.shape) == (1, 2, 8, 10, 3)
    assert torch.equal(output, torch.full_like(output, 0.5))
    assert calls[0][0].shape == (2, 3, 8, 10)
    assert calls[0][1].shape == (2, 3, 8, 10)
    assert torch.equal(calls[0][0], torch.full_like(calls[0][0], -0.5))
    assert torch.equal(calls[0][1], torch.full_like(calls[0][1], 0.5))


def test_seedvr2_post_processing_none_trims_and_crops_without_color_correction():
    decoded = torch.arange(1 * 3 * 9 * 11 * 3, dtype=torch.float32).reshape(1, 3, 9, 11, 3)
    reference = torch.zeros(1, 2, 8, 10, 3)

    with patch.object(nodes_seedvr, "lab_color_transfer") as lab:
        output = nodes_seedvr.SeedVR2PostProcessing.execute(decoded, reference, "none").result[0]

    assert lab.call_count == 0
    assert tuple(output.shape) == (1, 2, 8, 10, 3)
    assert torch.equal(output, decoded[:, :2, :8, :10, :])
