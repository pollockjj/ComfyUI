import torch

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

import comfy_extras.nodes_seedvr as nodes_seedvr  # noqa: E402


def test_resize_and_pad_returns_input_pixels_original_image_and_upscaled_shorter_edge():
    images = torch.zeros(1, 3, 16, 16, 3)

    output = nodes_seedvr.SeedVR2ResizeAndPad.execute(images, 120, 1.0)

    input_pixels, original_image, upscaled_shorter_edge = output.result
    assert tuple(input_pixels.shape) == (1, 5, 128, 128, 3)
    assert input_pixels.min().item() == 0.0
    assert input_pixels.max().item() == 0.0
    assert original_image is images
    assert upscaled_shorter_edge == 120


def test_resize_and_pad_multiplier_resolves_upscaled_shorter_edge():
    images = torch.zeros(1, 1, 16, 16, 3)

    output = nodes_seedvr.SeedVR2ResizeAndPad.execute(images, 120, 2.0)

    input_pixels, original_image, upscaled_shorter_edge = output.result
    assert tuple(input_pixels.shape) == (1, 1, 240, 240, 3)
    assert original_image is images
    assert upscaled_shorter_edge == 240


def test_resize_and_pad_rejects_non_positive_multiplier():
    images = torch.zeros(1, 1, 16, 16, 3)

    try:
        nodes_seedvr.SeedVR2ResizeAndPad.execute(images, 120, 0.0)
    except ValueError as e:
        assert "multiplier must be > 0" in str(e)
    else:
        raise AssertionError("non-positive multiplier was not rejected")


def test_resize_and_pad_treats_4d_image_as_one_video_frame_sequence():
    images = torch.zeros(2, 16, 16, 3)

    output = nodes_seedvr.SeedVR2ResizeAndPad.execute(images, 120, 1.0)

    input_pixels, original_image, upscaled_shorter_edge = output.result
    assert tuple(input_pixels.shape) == (1, 5, 128, 128, 3)
    assert original_image is images
    assert upscaled_shorter_edge == 120


def test_resize_and_pad_schema_and_execute_signature_are_preprocess_only():
    schema = nodes_seedvr.SeedVR2ResizeAndPad.define_schema()
    assert [item.id for item in schema.inputs] == ["images", "shorter_edge", "multiplier"]
    assert [item.id for item in schema.outputs] == ["input_pixels", "original_image", "upscaled_shorter_edge"]
