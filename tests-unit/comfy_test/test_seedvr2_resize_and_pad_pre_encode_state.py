import torch

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

import comfy_extras.nodes_seedvr as nodes_seedvr  # noqa: E402


def test_resize_and_pad_returns_input_pixels_and_shorter_edge():
    images = torch.zeros(1, 3, 16, 16, 3)

    output = nodes_seedvr.SeedVR2ResizeAndPad.execute(images, 120)

    input_pixels, shorter_edge = output.result
    assert tuple(input_pixels.shape) == (1, 5, 128, 128, 3)
    assert input_pixels.min().item() == 0.0
    assert input_pixels.max().item() == 0.0
    assert shorter_edge == 120


def test_resize_and_pad_treats_4d_image_as_one_video_frame_sequence():
    images = torch.zeros(2, 16, 16, 3)

    output = nodes_seedvr.SeedVR2ResizeAndPad.execute(images, 120)

    input_pixels, shorter_edge = output.result
    assert tuple(input_pixels.shape) == (1, 5, 128, 128, 3)
    assert shorter_edge == 120


def test_resize_and_pad_schema_and_execute_signature_are_preprocess_only():
    schema = nodes_seedvr.SeedVR2ResizeAndPad.define_schema()
    assert [item.id for item in schema.inputs] == ["images", "shorter_edge"]
    assert [item.id for item in schema.outputs] == ["input_pixels", "shorter_edge"]
