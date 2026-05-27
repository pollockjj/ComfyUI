"""Regression tests for the SeedVR2 native RoPE rewrite that replaces the
``apply_rotary_emb`` wrapper inside ``NaMMRotaryEmbedding3d.forward`` with
direct calls to ``comfy.ldm.flux.math.apply_rope1`` — matching the pattern
used by the other 7 ComfyUI native-DiT models (flux, hidream, kandinsky5,
lumina, qwen_image, wan, sam3).

Pinned invariants:

1. ``NaMMRotaryEmbedding3d.forward`` output is tensor-equal at fp32 against
   an oracle computed from the unchanged ``apply_rotary_emb`` wrapper fed
   with the legacy freqs layout — proving the rewrite is algorithmically
   lossless (dim=192, rot_d == head_dim path).
2. Partial-rope path (dim=128, rot_d=126 < head_dim=128): the rewired
   forward still tensor-equals the wrapper oracle, confirming the
   passthrough of the trailing 2 head-dims matches the legacy
   ``t_right = t[..., end_index:]`` behavior.

Pre-import CPU-only guard mirrors ``test_seedvr_rope_delegation.py`` —
``comfy.ldm.seedvr.model`` transitively imports ``comfy.model_management``
which probes ``torch.cuda.current_device()`` at import time unless
``args.cpu`` is set first.
"""

from __future__ import annotations

import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

import comfy.ldm.seedvr.model as seedvr_model  # noqa: E402
from comfy.ldm.seedvr.model import (  # noqa: E402
    Cache,
    NaMMRotaryEmbedding3d,
)


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
