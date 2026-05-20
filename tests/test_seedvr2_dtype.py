import inspect
from pathlib import Path

import torch

import comfy.ldm.modules.attention as attention
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
    original = t.clone()
    freqs_cis = torch.zeros(1, 2, 2, 2)

    out = seedvr_model._apply_rope1_partial(t, freqs_cis)

    assert out.dtype is torch.float16
    torch.testing.assert_close(out, (original.float() + 1.0).to(torch.float16))


def test_apply_rope1_partial_preserves_partial_rotation_input_dtype(monkeypatch):
    def fake_apply_rope1(t, freqs_cis):
        return t.float() + 1.0

    monkeypatch.setattr(seedvr_model, "apply_rope1", fake_apply_rope1)

    t = torch.arange(12, dtype=torch.float16).reshape(1, 2, 6)
    original = t.clone()
    freqs_cis = torch.zeros(1, 2, 2, 2)

    out = seedvr_model._apply_rope1_partial(t, freqs_cis)

    assert out.dtype is torch.float16
    torch.testing.assert_close(
        out[..., :4],
        (original[..., :4].float() + 1.0).to(torch.float16),
    )
    torch.testing.assert_close(out[..., 4:], original[..., 4:])


def test_apply_rope1_partial_chunks_sequence_dimension(monkeypatch):
    calls = []

    def fake_apply_rope1(t, freqs_cis):
        calls.append(t.shape[-2])
        return t.float() + 1.0

    monkeypatch.setattr(seedvr_model, "apply_rope1", fake_apply_rope1)
    monkeypatch.setattr(seedvr_model, "_ROPE1_PARTIAL_CHUNK_TOKENS", 2)

    t = torch.arange(30, dtype=torch.float16).reshape(1, 5, 6)
    original = t.clone()
    freqs_cis = torch.zeros(5, 2, 2, 2)

    out = seedvr_model._apply_rope1_partial(t, freqs_cis)

    assert calls == [2, 2, 1]
    torch.testing.assert_close(out[..., :4], (original[..., :4].float() + 1.0).to(torch.float16))
    torch.testing.assert_close(out[..., 4:], original[..., 4:])


def test_seedvr2_text_conditioning_accepts_cfg1_single_branch():
    context = torch.arange(6, dtype=torch.float32).reshape(1, 3, 2)

    txt, txt_shape = seedvr_model.NaDiT._resolve_text_conditioning(object(), context, [0])

    torch.testing.assert_close(txt, context.squeeze(0))
    torch.testing.assert_close(txt_shape, torch.tensor([[3]], device=context.device))


def test_seedvr2_text_conditioning_accepts_batched_cfg1_single_branch():
    context = torch.arange(12, dtype=torch.float32).reshape(2, 3, 2)

    txt, txt_shape = seedvr_model.NaDiT._resolve_text_conditioning(object(), context, [0])

    torch.testing.assert_close(txt, context.flatten(0, -2))
    torch.testing.assert_close(txt_shape, torch.tensor([[3], [3]], device=context.device))


def test_seedvr2_text_conditioning_preserves_two_branch_swap_contract():
    neg = torch.full((1, 3, 2), -1.0)
    pos = torch.full((1, 3, 2), 1.0)
    context = torch.cat([neg, pos], dim=0)

    txt, txt_shape = seedvr_model.NaDiT._resolve_text_conditioning(object(), context)

    torch.testing.assert_close(txt[:3], pos.squeeze(0))
    torch.testing.assert_close(txt[3:], neg.squeeze(0))
    torch.testing.assert_close(txt_shape, torch.tensor([[3], [3]], device=context.device))


def test_seedvr2_text_conditioning_preserves_batched_two_branch_swap_contract():
    neg = torch.full((2, 3, 2), -1.0)
    pos = torch.full((2, 3, 2), 1.0)
    context = torch.cat([neg, pos], dim=0)

    txt, txt_shape = seedvr_model.NaDiT._resolve_text_conditioning(object(), context, [1, 0])

    torch.testing.assert_close(txt[:6], pos.flatten(0, -2))
    torch.testing.assert_close(txt[6:], neg.flatten(0, -2))
    torch.testing.assert_close(txt_shape, torch.tensor([[3], [3], [3], [3]], device=context.device))


def test_seedvr2_cfg1_single_branch_output_is_not_swapped():
    out = torch.arange(6, dtype=torch.float32).reshape(1, 6)

    swapped = seedvr_model.NaDiT._swap_pos_neg_halves(object(), out, [0])

    torch.testing.assert_close(swapped, out)


def test_seedvr2_conditioning_keeps_comfy_cfg1_optimization_enabled():
    source = (Path(__file__).resolve().parents[1] / "comfy_extras" / "nodes_seedvr.py").read_text()

    assert "disable_model_cfg1_optimization()" not in source


def test_seedvr2_split_var_attention_matches_nested_var_attention():
    torch.manual_seed(1)
    q = torch.randn(5, 2, 4)
    k = torch.randn(7, 2, 4)
    v = torch.randn(7, 2, 4)
    cu_q = torch.tensor([0, 2, 5], dtype=torch.int32)
    cu_k = torch.tensor([0, 3, 7], dtype=torch.int32)

    nested = attention.var_attention_pytorch(
        q, k, v, heads=2, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
        skip_reshape=True, skip_output_reshape=True,
    )
    split = attention.var_attention_pytorch_split(
        q, k, v, heads=2, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
        skip_reshape=True, skip_output_reshape=True,
    )

    torch.testing.assert_close(split, nested, rtol=1e-5, atol=1e-5)


def test_seedvr2_split_var_attention_preserves_flat_output_shape():
    torch.manual_seed(2)
    q = torch.randn(5, 8)
    k = torch.randn(7, 8)
    v = torch.randn(7, 8)
    cu_q = torch.tensor([0, 1, 5], dtype=torch.int32)
    cu_k = torch.tensor([0, 2, 7], dtype=torch.int32)

    nested = attention.var_attention_pytorch(
        q, k, v, heads=2, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
    )
    split = attention.var_attention_pytorch_split(
        q, k, v, heads=2, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
    )

    assert split.shape == q.shape
    torch.testing.assert_close(split, nested, rtol=1e-5, atol=1e-5)


def test_seedvr2_7b_window_attention_routes_to_split_var_attention():
    source = inspect.getsource(seedvr_model.NaSwinAttention.forward)

    assert "var_attention_pytorch_split if self.version_7b else optimized_var_attention" in source
