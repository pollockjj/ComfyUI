from unittest.mock import MagicMock

import torch

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

import comfy_extras.nodes_seedvr as nodes_seedvr  # noqa: E402


class _NoHiddenStateWrapper:
    def __setattr__(self, name, value):
        if name in {"img_dims", "original_image_video", "tiled_args"}:
            raise AssertionError(f"SeedVR2InputProcessing wrote hidden VAE state {name}")
        super().__setattr__(name, value)


def _make_vae():
    vae = MagicMock()
    vae.is_seedvr2 = MagicMock(return_value=True)
    vae.first_stage_model = _NoHiddenStateWrapper()
    vae.encode = MagicMock(side_effect=AssertionError("SeedVR2InputProcessing called encode"))
    vae.encode_tiled = MagicMock(side_effect=AssertionError("SeedVR2InputProcessing called encode_tiled"))
    vae.decode = MagicMock(side_effect=AssertionError("SeedVR2InputProcessing called decode"))
    vae.decode_tiled = MagicMock(side_effect=AssertionError("SeedVR2InputProcessing called decode_tiled"))
    return vae


def test_input_processing_returns_processed_image_and_same_vae_without_encoding():
    vae = _make_vae()
    images = torch.zeros(1, 3, 16, 16, 3)

    output = nodes_seedvr.SeedVR2InputProcessing.execute(images, vae, 120)

    processed, returned_vae = output.result
    assert returned_vae is vae
    assert tuple(processed.shape) == (1, 5, 128, 128, 3)
    assert processed.min().item() == -1.0
    assert processed.max().item() == -1.0
    vae.encode.assert_not_called()
    vae.encode_tiled.assert_not_called()
    vae.decode.assert_not_called()
    vae.decode_tiled.assert_not_called()


def test_input_processing_treats_4d_image_as_one_video_frame_sequence():
    vae = _make_vae()
    images = torch.zeros(2, 16, 16, 3)

    output = nodes_seedvr.SeedVR2InputProcessing.execute(images, vae, 120)

    processed, returned_vae = output.result
    assert returned_vae is vae
    assert tuple(processed.shape) == (1, 5, 128, 128, 3)


def test_input_processing_schema_and_execute_signature_are_preprocess_only():
    schema = nodes_seedvr.SeedVR2InputProcessing.define_schema()
    assert [item.id for item in schema.inputs] == ["images", "vae", "resolution"]
    assert [item.id for item in schema.outputs] == ["processed", "vae"]


def test_input_processing_rejects_non_seedvr2_vae():
    vae = _make_vae()
    vae.is_seedvr2.return_value = False
    images = torch.zeros(1, 3, 16, 16, 3)

    try:
        nodes_seedvr.SeedVR2InputProcessing.execute(images, vae, 120)
    except ValueError as exc:
        assert str(exc) == "SeedVR2InputProcessing requires a SeedVR2 VAE."
    else:
        raise AssertionError("SeedVR2InputProcessing accepted a non-SeedVR2 VAE")
