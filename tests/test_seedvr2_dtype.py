from pathlib import Path

import torch

import comfy.supported_models
import comfy.ldm.seedvr.model as seedvr_model


def test_seedvr2_fp16_manual_cast_only_for_bf16_device(monkeypatch):
    bf16_device = object()
    fp16_device = object()

    monkeypatch.setattr(
        comfy.supported_models.comfy.model_management,
        "should_use_bf16",
        lambda device=None: device is bf16_device,
    )

    bf16_config = comfy.supported_models.SeedVR2({"image_model": "seedvr2"})
    bf16_config.set_inference_dtype(torch.float16, None, device=bf16_device)
    assert bf16_config.manual_cast_dtype is torch.bfloat16

    fp16_config = comfy.supported_models.SeedVR2({"image_model": "seedvr2"})
    fp16_config.set_inference_dtype(torch.float16, None, device=fp16_device)
    assert fp16_config.manual_cast_dtype is None


def test_apply_rope1_partial_preserves_full_rotation_input_dtype(monkeypatch):
    def fake_apply_rope1(t, freqs_cis):
        return t.float() + 1.0

    monkeypatch.setattr(seedvr_model, "apply_rope1", fake_apply_rope1)

    t = torch.arange(8, dtype=torch.float16).reshape(1, 2, 4)
    freqs_cis = torch.zeros(1, 2, 2, 2)

    out = seedvr_model._apply_rope1_partial(t, freqs_cis)

    assert out.dtype is torch.float16
    torch.testing.assert_close(out, (t.float() + 1.0).to(torch.float16))


def test_apply_rope1_partial_preserves_partial_rotation_input_dtype(monkeypatch):
    def fake_apply_rope1(t, freqs_cis):
        return t.float() + 1.0

    monkeypatch.setattr(seedvr_model, "apply_rope1", fake_apply_rope1)

    t = torch.arange(12, dtype=torch.float16).reshape(1, 2, 6)
    freqs_cis = torch.zeros(1, 2, 2, 2)

    out = seedvr_model._apply_rope1_partial(t, freqs_cis)

    assert out.dtype is torch.float16
    torch.testing.assert_close(
        out[..., :4],
        (t[..., :4].float() + 1.0).to(torch.float16),
    )
    torch.testing.assert_close(out[..., 4:], t[..., 4:])


def test_seedvr2_text_conditioning_accepts_cfg1_single_branch():
    context = torch.arange(6, dtype=torch.float32).reshape(1, 3, 2)

    txt, txt_shape = seedvr_model.NaDiT._resolve_text_conditioning(object(), context)

    torch.testing.assert_close(txt, context.squeeze(0))
    torch.testing.assert_close(txt_shape, torch.tensor([[3]], device=context.device))


def test_seedvr2_text_conditioning_preserves_two_branch_swap_contract():
    neg = torch.full((1, 3, 2), -1.0)
    pos = torch.full((1, 3, 2), 1.0)
    context = torch.cat([neg, pos], dim=0)

    txt, txt_shape = seedvr_model.NaDiT._resolve_text_conditioning(object(), context)

    torch.testing.assert_close(txt[:3], pos.squeeze(0))
    torch.testing.assert_close(txt[3:], neg.squeeze(0))
    torch.testing.assert_close(txt_shape, torch.tensor([[3], [3]], device=context.device))


def test_seedvr2_cfg1_single_branch_output_is_not_swapped():
    out = torch.arange(6, dtype=torch.float32).reshape(1, 6)

    swapped = seedvr_model.NaDiT._swap_pos_neg_halves(object(), out, [0])

    torch.testing.assert_close(swapped, out)


def test_seedvr2_conditioning_keeps_comfy_cfg1_optimization_enabled():
    source = (Path(__file__).resolve().parents[1] / "comfy_extras" / "nodes_seedvr.py").read_text()

    assert "disable_model_cfg1_optimization()" not in source
