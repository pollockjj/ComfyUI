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
    assert schema.inputs[2].min == 2
    assert schema.inputs[2].force_input is True
    assert schema.inputs[3].options == ["lab", "wavelet", "adain", "none"]
    assert schema.inputs[3].default == "lab"
    assert schema.outputs[0].get_io_type() == "IMAGE"


def test_seedvr2_post_processing_color_correction_memory_multipliers_are_named():
    assert nodes_seedvr.LAB_SCALE_MULTIPLIER == 13
    assert nodes_seedvr.WAVELET_SCALE_MULTIPLIER == 10
    assert nodes_seedvr.ADAIN_SCALE_MULTIPLIER == 6


def test_seedvr2_post_processing_lab_autochunks_from_memory_estimate(monkeypatch):
    decoded = torch.full((1, 5, 2, 2, 3), 0.25)
    original = torch.full((1, 5, 2, 2, 3), 0.75)
    calls = []

    def _lab(content, style):
        calls.append(content.shape[0])
        return content

    monkeypatch.setattr(nodes_seedvr.comfy.model_management, "vae_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(nodes_seedvr.comfy.model_management, "get_free_memory", lambda device: 1700)

    with patch.object(nodes_seedvr, "lab_color_transfer", _lab):
        output = nodes_seedvr.SeedVR2PostProcessing.execute(decoded, original, 2, "lab").result[0]

    assert calls == [1, 1, 1, 1, 1]
    assert tuple(output.shape) == (1, 5, 2, 2, 3)


def test_seedvr2_post_processing_oom_error_uses_color_correction_method(monkeypatch):
    decoded = torch.full((1, 3, 4, 4), 0.25)
    reference = torch.full((1, 3, 4, 4), 0.75)

    def _lab(content, style):
        raise torch.cuda.OutOfMemoryError("CUDA out of memory")

    monkeypatch.setattr(nodes_seedvr.comfy.model_management, "vae_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(nodes_seedvr.comfy.model_management, "get_free_memory", lambda device: 1_000_000)
    monkeypatch.setattr(nodes_seedvr.comfy.model_management, "soft_empty_cache", lambda: None)

    with patch.object(nodes_seedvr, "lab_color_transfer", _lab):
        try:
            nodes_seedvr.SeedVR2PostProcessing._color_transfer_chunked(
                decoded, reference, torch.device("cpu"), "lab",
            )
        except RuntimeError as exc:
            assert "color_correction_method=lab" in str(exc)
            assert " method=lab" not in str(exc)
        else:
            raise AssertionError("expected RuntimeError for one-frame LAB OOM")


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
    assert len(lab_calls) == 2
    assert lab_calls[0][1].shape == (1, 3, 4, 5)
    assert torch.equal(lab_calls[0][1], torch.zeros_like(lab_calls[0][1]))


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


def test_seedvr2_post_processing_unknown_color_correction_method_raises():
    decoded = torch.zeros(1, 2, 4, 4, 3)
    original = torch.zeros(1, 2, 4, 4, 3)
    try:
        nodes_seedvr.SeedVR2PostProcessing.execute(decoded, original, 4, "bogus")
    except ValueError as exc:
        assert "color_correction_method" in str(exc)
    else:
        raise AssertionError("expected ValueError for unknown color_correction_method")


def test_seedvr2_post_processing_uses_even_crop_from_odd_resized_width():
    decoded = torch.ones((1, 1, 128, 256, 3), dtype=torch.float32)
    original = torch.ones((1, 1, 120, 169, 3), dtype=torch.float32)

    output = nodes_seedvr.SeedVR2PostProcessing.execute(decoded, original, 120, "none").result[0]

    assert tuple(output.shape) == (1, 1, 120, 168, 3)


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
    sd_c = (cf.var(dim=2, correction=0) + eps).sqrt().reshape(b, c, 1, 1)
    mu_s = sf.mean(dim=2).reshape(b, c, 1, 1)
    sd_s = (sf.var(dim=2, correction=0) + eps).sqrt().reshape(b, c, 1, 1)
    expected = ((content.float() - mu_c) / sd_c) * sd_s + mu_s
    expected = expected.clamp(-1.0, 1.0)
    assert torch.allclose(out, expected, atol=1e-6)
