import logging
import warnings

import torch

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

import comfy.ldm.modules.attention as attention
import comfy.sd
import comfy.supported_models
import comfy.ldm.seedvr.model as seedvr_model


def test_set_model_config_inference_dtype_preserves_legacy_signature():
    calls = []

    class LegacyConfig:
        def set_inference_dtype(self, dtype, manual_cast_dtype):
            calls.append((dtype, manual_cast_dtype))

    comfy.sd._set_model_config_inference_dtype(LegacyConfig(), torch.float16, None, object())

    assert calls == [(torch.float16, None)]


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
    original = t.clone()
    freqs_cis = torch.zeros(1, 2, 2, 2)

    out = seedvr_model._apply_rope1_partial(t, freqs_cis)

    assert out.dtype is torch.float16
    torch.testing.assert_close(out, (original.float() + 1.0).to(torch.float16))


def test_seedvr2_text_conditioning_accepts_cfg1_single_branch():
    context = torch.arange(6, dtype=torch.float32).reshape(1, 3, 2)

    txt, txt_shape = seedvr_model.NaDiT._resolve_text_conditioning(object(), context, [0])

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


def test_seedvr2_split_var_attention_matches_nested_var_attention():
    torch.manual_seed(1)
    q = torch.randn(5, 2, 4)
    k = torch.randn(7, 2, 4)
    v = torch.randn(7, 2, 4)
    cu_q = torch.tensor([0, 2, 5], dtype=torch.int32)
    cu_k = torch.tensor([0, 3, 7], dtype=torch.int32)

    torch_fx_logger = logging.getLogger("torch.fx._symbolic_trace")
    old_torch_fx_level = torch_fx_logger.level
    torch_fx_logger.setLevel(logging.ERROR)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="The PyTorch API of nested tensors is in prototype stage.*",
                category=UserWarning,
            )
            nested = attention.var_attention_pytorch(
                q, k, v, heads=2, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
                skip_reshape=True, skip_output_reshape=True,
            )
    finally:
        torch_fx_logger.setLevel(old_torch_fx_level)
    split = attention.var_attention_pytorch_split(
        q, k, v, heads=2, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
        skip_reshape=True, skip_output_reshape=True,
    )

    torch.testing.assert_close(split, nested, rtol=1e-5, atol=1e-5)


def test_seedvr2_vae_decode_memory_covers_full_frame_lab_transfer():
    estimate = comfy.sd._seedvr2_vae_decode_memory_used((1, 16, 26, 120, 160))
    old_estimate = 16 * 120 * 160 * (4 * 8 * 8) * 2

    assert estimate == 101 * 960 * 1280 * 160
    assert estimate > 15 * 1024 ** 3
    assert estimate > old_estimate * 100
