"""Consolidated SeedVR2 internals regression tests.

Sources (all merged verbatim, helper names disambiguated where colliding):

  * RoPE rewrite — NaMMRotaryEmbedding3d.forward must match the legacy
    apply_rotary_emb wrapper oracle at fp32 (full and partial-rope paths).
  * GroupNorm limit gate — causal_norm_wrapper at vae.py:509 must compare
    memory_occupy against get_norm_limit(), not float('inf').
  * var_attention backend registry + sage/pytorch_split backend contracts.
  * var_attention_pytorch SeedVR2-named guard — present-API shape contract
    and torch RuntimeError pass-through on malformed offsets, with AST-level
    pinning of the guard ordering.

Pre-import CPU-only guard is required because comfy.ldm.seedvr.model and
comfy.ldm.modules.attention transitively pull in comfy.model_management,
which probes torch.cuda.current_device() at import time unless args.cpu is
set first.
"""

from __future__ import annotations

import ast
import inspect
import logging
import textwrap
import warnings
from unittest.mock import patch

import pytest
import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

import comfy.ldm.seedvr.model as seedvr_model  # noqa: E402
import comfy.ldm.seedvr.vae as vae_mod  # noqa: E402
import comfy.ldm.modules.attention as attention  # noqa: E402
import comfy.ops as comfy_ops  # noqa: E402
from comfy.ldm.seedvr.model import (  # noqa: E402
    Cache,
    NaMMRotaryEmbedding3d,
)
from comfy.ldm.seedvr.vae import (  # noqa: E402
    causal_norm_wrapper,
    set_norm_limit,
)
from comfy.ldm.modules.attention import var_attention_pytorch  # noqa: E402


# ---------------------------------------------------------------------------
# RoPE rewrite tests (test_seedvr_rope_rewrite.py)
# ---------------------------------------------------------------------------

# Test rig dimensions. dim=192 → per-axis rope dim = 64 (even, lucidrains
# requirement). vid_shape=(2,4,4) → L_vid = 32. txt_shape=(8,) → L_txt = 8.
_DIM = 192
_HEADS = 4
_VID_T, _VID_H, _VID_W = 2, 4, 4
_TXT_L = 8
_L_VID = _VID_T * _VID_H * _VID_W
_SEED = 0


def _make_inputs(dtype=torch.float32, device="cpu"):
    """Construct the 6 forward inputs + cache. Deterministic via local
    Generator so global RNG state is not mutated.
    """
    g = torch.Generator(device=device).manual_seed(_SEED)
    vid_q = torch.randn(_L_VID, _HEADS, _DIM, dtype=dtype, device=device, generator=g)
    vid_k = torch.randn(_L_VID, _HEADS, _DIM, dtype=dtype, device=device, generator=g)
    txt_q = torch.randn(_TXT_L, _HEADS, _DIM, dtype=dtype, device=device, generator=g)
    txt_k = torch.randn(_TXT_L, _HEADS, _DIM, dtype=dtype, device=device, generator=g)
    vid_shape = torch.tensor([[_VID_T, _VID_H, _VID_W]], dtype=torch.long, device=device)
    txt_shape = torch.tensor([[_TXT_L]], dtype=torch.long, device=device)
    cache = Cache(disable=True)
    return vid_q, vid_k, vid_shape, txt_q, txt_k, txt_shape, cache


def _legacy_get_freqs(rope: NaMMRotaryEmbedding3d, vid_shape, txt_shape):
    """Reproduce the pre-rewrite ``get_freqs`` body verbatim against
    ``self.get_axial_freqs`` (parent ``RotaryEmbeddingBase`` method,
    unchanged by the rewrite).
    """
    max_temporal = 0
    max_height = 0
    max_width = 0
    max_txt_len = 0
    for (f, h, w), l in zip(vid_shape.tolist(), txt_shape[:, 0].tolist()):
        max_temporal = max(max_temporal, l + f)
        max_height = max(max_height, h)
        max_width = max(max_width, w)
        max_txt_len = max(max_txt_len, l)
    with torch.amp.autocast(device_type="cuda", enabled=False):
        vid_freqs_full = rope.get_axial_freqs(
            min(max_temporal + 16, 1024),
            min(max_height + 4, 128),
            min(max_width + 4, 128),
        ).float()
        txt_freqs_full = rope.get_axial_freqs(min(max_txt_len + 16, 1024))
    vid_freq_list, txt_freq_list = [], []
    for (f, h, w), l in zip(vid_shape.tolist(), txt_shape[:, 0].tolist()):
        vid_freq = vid_freqs_full[l : l + f, :h, :w].reshape(-1, vid_freqs_full.size(-1))
        txt_freq = txt_freqs_full[:l].repeat(1, 3).reshape(-1, vid_freqs_full.size(-1))
        vid_freq_list.append(vid_freq)
        txt_freq_list.append(txt_freq)
    return torch.cat(vid_freq_list, dim=0), torch.cat(txt_freq_list, dim=0)


def _legacy_forward(rope: NaMMRotaryEmbedding3d, vid_q, vid_k, vid_shape,
                    txt_q, txt_k, txt_shape):
    """Compute expected forward output via the unchanged
    ``apply_rotary_emb`` wrapper fed with legacy-shape freqs. This is the
    oracle. The wrapper itself is out of scope for the rewrite (Shape B).
    """
    vid_freqs, txt_freqs = _legacy_get_freqs(rope, vid_shape, txt_shape)
    vid_freqs = vid_freqs.to(vid_q.device)
    txt_freqs = txt_freqs.to(txt_q.device)

    from einops import rearrange

    vid_q = rearrange(vid_q, "L h d -> h L d")
    vid_k = rearrange(vid_k, "L h d -> h L d")
    vid_q_out = seedvr_model.apply_rotary_emb(vid_freqs, vid_q.float()).to(vid_q.dtype)
    vid_k_out = seedvr_model.apply_rotary_emb(vid_freqs, vid_k.float()).to(vid_k.dtype)
    vid_q_out = rearrange(vid_q_out, "h L d -> L h d")
    vid_k_out = rearrange(vid_k_out, "h L d -> L h d")

    txt_q = rearrange(txt_q, "L h d -> h L d")
    txt_k = rearrange(txt_k, "L h d -> h L d")
    txt_q_out = seedvr_model.apply_rotary_emb(txt_freqs, txt_q.float()).to(txt_q.dtype)
    txt_k_out = seedvr_model.apply_rotary_emb(txt_freqs, txt_k.float()).to(txt_k.dtype)
    txt_q_out = rearrange(txt_q_out, "h L d -> L h d")
    txt_k_out = rearrange(txt_k_out, "h L d -> L h d")
    return vid_q_out, vid_k_out, txt_q_out, txt_k_out


def test_namm_forward_output_tensor_equal_against_legacy_oracle():
    rope = NaMMRotaryEmbedding3d(dim=_DIM)
    vid_q, vid_k, vid_shape, txt_q, txt_k, txt_shape, cache = _make_inputs()

    expected_vid_q, expected_vid_k, expected_txt_q, expected_txt_k = _legacy_forward(
        rope,
        vid_q.clone(), vid_k.clone(), vid_shape,
        txt_q.clone(), txt_k.clone(), txt_shape,
    )

    actual_vid_q, actual_vid_k, actual_txt_q, actual_txt_k = rope.forward(
        vid_q.clone(), vid_k.clone(), vid_shape,
        txt_q.clone(), txt_k.clone(), txt_shape, cache,
    )

    torch.testing.assert_close(actual_vid_q, expected_vid_q, rtol=0, atol=0,
                                msg="vid_q output diverges from wrapper oracle")
    torch.testing.assert_close(actual_vid_k, expected_vid_k, rtol=0, atol=0,
                                msg="vid_k output diverges from wrapper oracle")
    torch.testing.assert_close(actual_txt_q, expected_txt_q, rtol=0, atol=0,
                                msg="txt_q output diverges from wrapper oracle")
    torch.testing.assert_close(actual_txt_k, expected_txt_k, rtol=0, atol=0,
                                msg="txt_k output diverges from wrapper oracle")


def test_namm_forward_partial_rope_passthrough_matches_wrapper_oracle():
    rope = NaMMRotaryEmbedding3d(dim=128)
    g = torch.Generator(device="cpu").manual_seed(_SEED)
    vid_q = torch.randn(_L_VID, _HEADS, 128, dtype=torch.float32, device="cpu", generator=g)
    vid_k = torch.randn(_L_VID, _HEADS, 128, dtype=torch.float32, device="cpu", generator=g)
    txt_q = torch.randn(_TXT_L, _HEADS, 128, dtype=torch.float32, device="cpu", generator=g)
    txt_k = torch.randn(_TXT_L, _HEADS, 128, dtype=torch.float32, device="cpu", generator=g)
    vid_shape = torch.tensor([[_VID_T, _VID_H, _VID_W]], dtype=torch.long)
    txt_shape = torch.tensor([[_TXT_L]], dtype=torch.long)
    cache = Cache(disable=True)

    expected_vid_q, expected_vid_k, expected_txt_q, expected_txt_k = _legacy_forward(
        rope, vid_q.clone(), vid_k.clone(), vid_shape, txt_q.clone(), txt_k.clone(), txt_shape,
    )
    actual_vid_q, actual_vid_k, actual_txt_q, actual_txt_k = rope.forward(
        vid_q.clone(), vid_k.clone(), vid_shape, txt_q.clone(), txt_k.clone(), txt_shape, cache,
    )

    vid_freqs, _ = rope.get_freqs(vid_shape, txt_shape)
    rot_d = 2 * vid_freqs.shape[-3]
    assert rot_d == 126, f"expected rot_d=126 for dim=128 model; got {rot_d}"
    assert rot_d < 128, "partial-rope path must trigger (rot_d < head_dim)"

    torch.testing.assert_close(actual_vid_q, expected_vid_q, rtol=0, atol=0,
                                msg="vid_q partial-rope output diverges from wrapper oracle")
    torch.testing.assert_close(actual_vid_k, expected_vid_k, rtol=0, atol=0,
                                msg="vid_k partial-rope output diverges from wrapper oracle")
    torch.testing.assert_close(actual_txt_q, expected_txt_q, rtol=0, atol=0,
                                msg="txt_q partial-rope output diverges from wrapper oracle")
    torch.testing.assert_close(actual_txt_k, expected_txt_k, rtol=0, atol=0,
                                msg="txt_k partial-rope output diverges from wrapper oracle")


# ---------------------------------------------------------------------------
# GroupNorm limit tests (test_seedvr_groupnorm_limit.py)
# ---------------------------------------------------------------------------

_NUM_CHANNELS = 8
_NUM_GROUPS = 4
_TENSOR_SHAPE = (1, 8, 2, 4, 4)

_GROUPNORM_SUBCLASSES = [
    pytest.param(comfy_ops.disable_weight_init.GroupNorm, id="disable_weight_init"),
    pytest.param(comfy_ops.manual_cast.GroupNorm, id="manual_cast"),
]


@pytest.mark.parametrize("groupnorm_cls", _GROUPNORM_SUBCLASSES)
def test_seedvr_groupnorm_low_limit_uses_chunked_groupnorm_path(groupnorm_cls):
    real_group_norm = vae_mod.F.group_norm
    set_norm_limit(1e-9)
    try:
        gn = groupnorm_cls(num_channels=_NUM_CHANNELS, num_groups=_NUM_GROUPS)
        gn.eval()

        forward_hook_calls = []

        def _hook(module, inputs, output):
            forward_hook_calls.append(tuple(inputs[0].shape))

        spy_calls = []

        def _group_norm_spy(input_tensor, num_groups_arg, *args, **kwargs):
            spy_calls.append({"num_groups": int(num_groups_arg)})
            return real_group_norm(input_tensor, num_groups_arg, *args, **kwargs)

        handle = gn.register_forward_hook(_hook)
        try:
            with patch.object(vae_mod.F, "group_norm", side_effect=_group_norm_spy):
                out_tensor = causal_norm_wrapper(gn, torch.randn(*_TENSOR_SHAPE))
        finally:
            handle.remove()

        full_calls = len(forward_hook_calls)
        chunked_calls = sum(1 for entry in spy_calls if entry["num_groups"] < _NUM_GROUPS)

        assert tuple(int(s) for s in out_tensor.shape) == _TENSOR_SHAPE
        assert full_calls == 0, (
            f"low-limit GroupNorm gate must NOT take the full-forward path; got full_calls={full_calls}"
        )
        assert chunked_calls > 0, (
            f"low-limit GroupNorm gate must take the chunked path; got chunked_calls={chunked_calls}"
        )
    finally:
        set_norm_limit(None)


# ---------------------------------------------------------------------------
# var_attention backend tests (test_seedvr_var_attention_backends.py)
# ---------------------------------------------------------------------------

def _var_attention_backends_inputs():
    heads = 2
    head_dim = 4
    total = 6
    q = torch.randn(total, heads, head_dim)
    k = torch.randn(total, heads, head_dim)
    v = torch.randn(total, heads, head_dim)
    cu = torch.tensor([0, 3, 6], dtype=torch.int32)
    return q, k, v, heads, cu


def test_var_attention_registry_contains_always_available_entries():
    assert attention.REGISTERED_ATTENTION_FUNCTIONS["var_attention_pytorch"] is attention.var_attention_pytorch
    assert attention.REGISTERED_ATTENTION_FUNCTIONS["var_attention_sub_quad"] is attention.var_attention_sub_quad
    assert attention.REGISTERED_ATTENTION_FUNCTIONS["var_attention_split"] is attention.var_attention_split


def test_var_attention_sage_uses_cu_seqlens_contract(monkeypatch):
    q, k, v, heads, cu = _var_attention_backends_inputs()
    captured = {}

    def fake_sageattn_varlen(q, k, v, cu_q, cu_k, max_q, max_k, is_causal, sm_scale):
        captured.update(cu_q=cu_q, cu_k=cu_k, max_q=max_q, max_k=max_k, is_causal=is_causal)
        return torch.zeros_like(q)

    monkeypatch.setattr(attention, "SAGE_ATTENTION_VARLEN_IS_AVAILABLE", True)
    monkeypatch.setattr(attention, "sageattn_varlen", fake_sageattn_varlen, raising=False)

    out = attention.var_attention_sage(q, k, v, heads, cu, cu, skip_reshape=True, skip_output_reshape=True)

    assert tuple(out.shape) == tuple(q.shape)
    assert torch.equal(captured["cu_q"], cu)
    assert torch.equal(captured["cu_k"], cu)
    assert captured["max_q"] == 3
    assert captured["max_k"] == 3
    assert captured["is_causal"] is False


def test_var_attention_pytorch_split_normalizes_split_indices_to_cpu(monkeypatch):
    q, k, v, heads, cu = _var_attention_backends_inputs()
    captured_devices = []
    real_tensor_split = torch.tensor_split

    def capture_tensor_split(input, indices_or_sections, dim=0):
        if isinstance(indices_or_sections, torch.Tensor):
            captured_devices.append(indices_or_sections.device.type)
        return real_tensor_split(input, indices_or_sections, dim=dim)

    monkeypatch.setattr(torch, "tensor_split", capture_tensor_split)

    out = attention.var_attention_pytorch_split(q, k, v, heads, cu, cu, skip_reshape=True, skip_output_reshape=True)

    assert tuple(out.shape) == tuple(q.shape)
    assert captured_devices == ["cpu", "cpu", "cpu"]


# ---------------------------------------------------------------------------
# var_attention_pytorch SeedVR2 guard tests
# (test_var_attention_pytorch_seedvr2_guard.py)
# ---------------------------------------------------------------------------

def _pytorch_guard_inputs():
    heads, head_dim, total_tokens = 2, 8, 6
    embed_dim = heads * head_dim
    q = torch.randn(total_tokens, embed_dim)
    k = torch.randn(total_tokens, embed_dim)
    v = torch.randn(total_tokens, embed_dim)
    cu = torch.tensor([0, 3, 6], dtype=torch.int32)
    return q, k, v, heads, cu, cu, total_tokens, embed_dim


def _assert_guard_source_pin():
    src = textwrap.dedent(inspect.getsource(var_attention_pytorch))
    tree = ast.parse(src)
    raise_lines = []
    nested_lines = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call):
            func = node.exc.func
            if isinstance(func, ast.Name) and func.id == "RuntimeError":
                raise_lines.append(node.lineno)
        if isinstance(node, ast.Attribute) and node.attr == "nested_tensor_from_jagged":
            nested_lines.append(node.lineno)
    assert raise_lines, (
        "var_attention_pytorch has no `raise RuntimeError(...)` AST node; "
        f"the SeedVR2-named guard is missing.\n--- source ---\n{src}"
    )
    assert nested_lines, (
        "var_attention_pytorch source has no `nested_tensor_from_jagged` "
        f"attribute access; cannot pin guard ordering.\n"
        f"--- source ---\n{src}"
    )
    first_raise = min(raise_lines)
    first_nested = min(nested_lines)
    assert first_raise < first_nested, (
        f"`raise RuntimeError(...)` first appears at line {first_raise}, "
        f"but `torch.nested.nested_tensor_from_jagged` is referenced first "
        f"at line {first_nested}; the guard must precede the lookup.\n"
        f"--- source ---\n{src}"
    )


def test_present_api_returns_expected_shape():
    q, k, v, heads, cu_q, cu_k, total_tokens, embed_dim = _pytorch_guard_inputs()

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
            out = var_attention_pytorch(q, k, v, heads, cu_q, cu_k)
    finally:
        torch_fx_logger.setLevel(old_torch_fx_level)

    assert tuple(out.shape) == (total_tokens, embed_dim), (
        f"expected ({total_tokens}, {embed_dim}); got {tuple(out.shape)}"
    )

    _assert_guard_source_pin()


def test_malformed_offsets_propagates_torch_runtime_error():
    q, k, v, heads, _, _, _, _ = _pytorch_guard_inputs()
    cu_q_bad = torch.tensor([0, 3, 7], dtype=torch.int32)
    cu_k_ok = torch.tensor([0, 3, 6], dtype=torch.int32)

    with pytest.raises(RuntimeError) as exc_info:
        var_attention_pytorch(q, k, v, heads, cu_q_bad, cu_k_ok)

    msg = str(exc_info.value)
    assert "split_with_sizes" in msg, (
        f"expected torch's `split_with_sizes` error to propagate; got: {msg!r}"
    )
    assert "SeedVR2" not in msg, (
        f"SeedVR2-context substring must not be substituted onto torch's "
        f"per-call shape error; got: {msg!r}"
    )

    _assert_guard_source_pin()
