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

    assert _schema_ids(schema.inputs) == ["decoded", "original_image", "upscaled_shorter_edge", "color_correction_method"]
    assert schema.inputs[2].default is None
    assert schema.inputs[2].force_input is True
    assert schema.inputs[3].options == ["lab", "wavelet", "adain", "none"]
    assert schema.inputs[3].default == "lab"
    assert schema.outputs[0].get_io_type() == "IMAGE"


def test_seedvr2_post_processing_lab_derives_reference_from_original_and_upscaled_shorter_edge():
    decoded = torch.full((1, 3, 9, 11, 3), 0.25)
    original = torch.full((1, 2, 16, 20, 3), 0.75)
    calls = []

    def _lab(content, style):
        calls.append((content.clone(), style.clone()))
        return torch.zeros_like(content)

    with patch.object(nodes_seedvr, "lab_color_transfer", _lab):
        output = nodes_seedvr.SeedVR2PostProcessing.execute(decoded, original, 8, "lab").result[0]

    assert tuple(output.shape) == (1, 2, 8, 10, 3)
    assert torch.equal(output, torch.full_like(output, 0.5))
    assert calls[0][0].shape == (2, 3, 8, 10)
    assert calls[0][1].shape == (2, 3, 8, 10)
    assert torch.equal(calls[0][0], torch.full_like(calls[0][0], -0.5))
    assert torch.allclose(calls[0][1], torch.full_like(calls[0][1], 0.5))


def test_seedvr2_post_processing_lab_runs_color_transfer_on_vae_device():
    source = inspect.getsource(nodes_seedvr.SeedVR2PostProcessing.execute)
    helper_source = inspect.getsource(nodes_seedvr.SeedVR2PostProcessing._lab_color_transfer_on_vae_device)

    assert "_lab_color_transfer_on_vae_device" in source
    assert "reference_5d.to(device=decoded_5d.device)" not in source
    assert "comfy.model_management.vae_device()" in helper_source
    assert ".to(device=color_device)" in helper_source
    assert ".to(device=output_device)" in helper_source


def test_seedvr2_post_processing_raw_conversion_does_not_probe_full_tensor_range():
    source = inspect.getsource(nodes_seedvr.SeedVR2PostProcessing._to_seedvr2_raw)

    assert ".amin" not in source
    assert ".item" not in source


def test_seedvr2_post_processing_lab_resizes_full_reference_frame():
    decoded = torch.full((1, 2, 4, 5, 3), 0.25)
    original = torch.full((1, 2, 16, 20, 3), 0.75)
    resize_calls = []
    lab_calls = []

    def _resize(images, size, interpolation=None, antialias=None):
        resize_calls.append((images.clone(), size, interpolation, antialias))
        if isinstance(size, int):
            return torch.full((2, 3, size, round(images.shape[-1] * size / images.shape[-2])), 0.5)
        return torch.full((2, 3, size[0], size[1]), 0.5)

    def _lab(content, style):
        lab_calls.append((content.clone(), style.clone()))
        return torch.zeros_like(content)

    with patch.object(nodes_seedvr.TVF, "resize", _resize):
        with patch.object(nodes_seedvr, "lab_color_transfer", _lab):
            output = nodes_seedvr.SeedVR2PostProcessing.execute(decoded, original, 8, "lab").result[0]

    assert tuple(output.shape) == (1, 2, 4, 4, 3)
    assert torch.equal(output, torch.full_like(output, 0.5))
    assert resize_calls[0][0].shape == (2, 3, 16, 20)
    assert resize_calls[0][1] == 8
    assert resize_calls[1][0].shape == (2, 3, 8, 10)
    assert resize_calls[1][1] == (4, 5)
    assert lab_calls[0][1].shape == (2, 3, 4, 5)
    assert torch.equal(lab_calls[0][1], torch.zeros_like(lab_calls[0][1]))


def test_seedvr2_post_processing_none_trims_and_crops_without_color_correction():
    decoded = torch.arange(1 * 3 * 9 * 11 * 3, dtype=torch.float32).reshape(1, 3, 9, 11, 3)
    original = torch.zeros(1, 2, 16, 20, 3)

    with patch.object(nodes_seedvr, "lab_color_transfer") as lab:
        output = nodes_seedvr.SeedVR2PostProcessing.execute(decoded, original, 8, "none").result[0]

    assert lab.call_count == 0
    assert tuple(output.shape) == (1, 2, 8, 10, 3)
    assert torch.equal(output, decoded[:, :2, :8, :10, :])


def test_seedvr2_post_processing_restores_flattened_padded_batches_before_trimming():
    decoded = torch.arange(10 * 4 * 6 * 1, dtype=torch.float32).reshape(10, 4, 6, 1)
    original = torch.zeros(2, 2, 4, 6, 1)

    output = nodes_seedvr.SeedVR2PostProcessing.execute(decoded, original, 4, "none").result[0]

    expected = torch.cat((decoded[0:2], decoded[5:7]), dim=0)
    assert tuple(output.shape) == (4, 4, 6, 1)
    assert torch.equal(output, expected)


def test_seedvr2_post_processing_none_preserves_decoded_spatial_size_when_reference_is_larger():
    decoded = torch.arange(1 * 3 * 8 * 10 * 3, dtype=torch.float32).reshape(1, 3, 8, 10, 3)
    original = torch.zeros(1, 2, 16, 20, 3)

    with patch.object(nodes_seedvr, "lab_color_transfer") as lab:
        output = nodes_seedvr.SeedVR2PostProcessing.execute(decoded, original, 16, "none").result[0]

    assert lab.call_count == 0
    assert tuple(output.shape) == (1, 2, 8, 10, 3)
    assert torch.equal(output, decoded[:, :2, :, :, :])


def test_seedvr2_post_processing_crops_to_reference_tensor_when_reference_is_smaller():
    decoded = torch.ones((1, 1, 720, 1280, 3), dtype=torch.float32)
    original = torch.ones((1, 1, 360, 640, 3), dtype=torch.float32)

    output = nodes_seedvr.SeedVR2PostProcessing.execute(decoded, original, 360, "none").result[0]

    assert tuple(output.shape) == (1, 1, 360, 640, 3)


def test_seedvr2_post_processing_uses_decoded_size_when_reference_is_larger():
    decoded = torch.ones((1, 1, 128, 160, 3), dtype=torch.float32)
    original = torch.ones((1, 1, 480, 640, 3), dtype=torch.float32)

    output = nodes_seedvr.SeedVR2PostProcessing.execute(decoded, original, 480, "none").result[0]

    assert tuple(output.shape) == (1, 1, 128, 160, 3)


def test_seedvr2_post_processing_derives_crop_from_upscaled_shorter_edge():
    decoded = torch.ones((1, 1, 128, 224, 3), dtype=torch.float32)
    original = torch.ones((1, 1, 1080, 1920, 3), dtype=torch.float32)

    output = nodes_seedvr.SeedVR2PostProcessing.execute(decoded, original, 120, "none").result[0]

    assert tuple(output.shape) == (1, 1, 120, 212, 3)


def test_seedvr2_post_processing_uses_even_crop_from_odd_resized_width():
    decoded = torch.ones((1, 1, 128, 256, 3), dtype=torch.float32)
    original = torch.ones((1, 1, 120, 169, 3), dtype=torch.float32)

    output = nodes_seedvr.SeedVR2PostProcessing.execute(decoded, original, 120, "none").result[0]

    assert tuple(output.shape) == (1, 1, 120, 168, 3)


def test_seedvr2_post_processing_none_preserves_black_bottom_row_content():
    decoded = torch.ones((1, 2, 8, 10, 3), dtype=torch.float32)
    original = torch.ones((1, 2, 8, 10, 3), dtype=torch.float32)
    original[:, :, -1, :, :] = -1.0

    output = nodes_seedvr.SeedVR2PostProcessing.execute(decoded, original, 8, "none").result[0]

    assert tuple(output.shape) == (1, 2, 8, 10, 3)
    assert torch.equal(output, decoded)


def test_seedvr2_post_processing_none_preserves_black_right_column_content():
    decoded = torch.ones((1, 2, 8, 10, 3), dtype=torch.float32)
    original = torch.ones((1, 2, 8, 10, 3), dtype=torch.float32)
    original[:, :, :, -1, :] = -1.0

    output = nodes_seedvr.SeedVR2PostProcessing.execute(decoded, original, 8, "none").result[0]

    assert tuple(output.shape) == (1, 2, 8, 10, 3)
    assert torch.equal(output, decoded)


def test_seedvr2_post_processing_wavelet_dispatch_routes_through_wavelet_color_transfer():
    decoded = torch.full((1, 3, 9, 11, 3), 0.25)
    original = torch.full((1, 2, 16, 20, 3), 0.75)
    wavelet_calls = []
    lab_calls = []

    def _wavelet(content, style):
        wavelet_calls.append((content.clone(), style.clone()))
        return torch.zeros_like(content)

    def _lab(content, style):
        lab_calls.append((content.clone(), style.clone()))
        return torch.zeros_like(content)

    with patch.object(nodes_seedvr, "wavelet_color_transfer", _wavelet):
        with patch.object(nodes_seedvr, "lab_color_transfer", _lab):
            output = nodes_seedvr.SeedVR2PostProcessing.execute(decoded, original, 8, "wavelet").result[0]

    assert len(wavelet_calls) == 1
    assert len(lab_calls) == 0
    assert tuple(output.shape) == (1, 2, 8, 10, 3)
    assert torch.equal(output, torch.full_like(output, 0.5))
    assert wavelet_calls[0][0].shape == (2, 3, 8, 10)
    assert wavelet_calls[0][1].shape == (2, 3, 8, 10)
    assert torch.equal(wavelet_calls[0][0], torch.full_like(wavelet_calls[0][0], -0.5))
    assert torch.allclose(wavelet_calls[0][1], torch.full_like(wavelet_calls[0][1], 0.5))


def test_seedvr2_post_processing_adain_dispatch_routes_through_adain_color_transfer():
    decoded = torch.full((1, 3, 9, 11, 3), 0.25)
    original = torch.full((1, 2, 16, 20, 3), 0.75)
    adain_calls = []
    lab_calls = []

    def _adain(content, style):
        adain_calls.append((content.clone(), style.clone()))
        return torch.zeros_like(content)

    def _lab(content, style):
        lab_calls.append((content.clone(), style.clone()))
        return torch.zeros_like(content)

    with patch.object(nodes_seedvr, "adain_color_transfer", _adain):
        with patch.object(nodes_seedvr, "lab_color_transfer", _lab):
            output = nodes_seedvr.SeedVR2PostProcessing.execute(decoded, original, 8, "adain").result[0]

    assert len(adain_calls) == 1
    assert len(lab_calls) == 0
    assert tuple(output.shape) == (1, 2, 8, 10, 3)
    assert torch.equal(output, torch.full_like(output, 0.5))
    assert adain_calls[0][0].shape == (2, 3, 8, 10)
    assert adain_calls[0][1].shape == (2, 3, 8, 10)


def test_seedvr2_color_transfer_helper_runs_on_vae_device():
    import inspect as _inspect
    helper_source = _inspect.getsource(nodes_seedvr.SeedVR2PostProcessing._color_transfer_on_vae_device)
    assert "comfy.model_management.vae_device()" in helper_source
    assert ".to(device=color_device)" in helper_source
    assert ".to(device=output_device)" in helper_source
    assert "transfer_fn" in helper_source


def test_seedvr2_wavelet_color_transfer_matches_primary_source_reconstruction():
    from comfy.ldm.seedvr import vae as seedvr_vae
    torch.manual_seed(0)
    content = torch.rand(1, 3, 12, 16) * 2.0 - 1.0
    style = torch.rand(1, 3, 12, 16) * 2.0 - 1.0
    out = seedvr_vae.wavelet_color_transfer(content, style)
    expected = seedvr_vae.wavelet_reconstruction(content.clone(), style.clone())
    assert torch.equal(out, expected)


def test_seedvr2_adain_color_transfer_matches_huang_belongie_formula():
    from comfy.ldm.seedvr import vae as seedvr_vae
    torch.manual_seed(0)
    content = torch.rand(2, 3, 5, 7) * 2.0 - 1.0
    style = torch.rand(2, 3, 5, 7) * 2.0 - 1.0
    out = seedvr_vae.adain_color_transfer(content.clone(), style.clone())

    b, c = 2, 3
    cf = content.float().reshape(b, c, -1)
    sf = style.float().reshape(b, c, -1)
    eps = 1e-5
    mu_c = cf.mean(dim=2).reshape(b, c, 1, 1)
    sd_c = (cf.var(dim=2) + eps).sqrt().reshape(b, c, 1, 1)
    mu_s = sf.mean(dim=2).reshape(b, c, 1, 1)
    sd_s = (sf.var(dim=2) + eps).sqrt().reshape(b, c, 1, 1)
    expected = ((content.float() - mu_c) / sd_c) * sd_s + mu_s
    expected = expected.clamp(-1.0, 1.0)
    assert torch.allclose(out, expected, atol=1e-6)


def test_seedvr2_adain_preserves_input_dtype():
    from comfy.ldm.seedvr import vae as seedvr_vae
    content = (torch.rand(1, 3, 4, 4) * 2.0 - 1.0).to(torch.float16)
    style = (torch.rand(1, 3, 4, 4) * 2.0 - 1.0).to(torch.float16)
    out = seedvr_vae.adain_color_transfer(content, style)
    assert out.dtype == torch.float16


def test_seedvr2_adain_resizes_mismatched_style_to_content_shape():
    from comfy.ldm.seedvr import vae as seedvr_vae
    content = torch.rand(1, 3, 8, 10) * 2.0 - 1.0
    style = torch.rand(1, 3, 16, 20) * 2.0 - 1.0
    out = seedvr_vae.adain_color_transfer(content, style)
    assert tuple(out.shape) == (1, 3, 8, 10)


def test_seedvr2_post_processing_unknown_color_correction_method_raises():
    decoded = torch.zeros(1, 2, 4, 4, 3)
    original = torch.zeros(1, 2, 4, 4, 3)
    try:
        nodes_seedvr.SeedVR2PostProcessing.execute(decoded, original, 4, "bogus")
    except ValueError as exc:
        assert "color_correction_method" in str(exc)
    else:
        raise AssertionError("expected ValueError for unknown color_correction_method")
