"""Regression test: comfy.ldm.seedvr.model.apply_rotary_emb must delegate to
comfy.ldm.flux.math.apply_rope1 with byte-exact equality across the wrapper's
slicing, scaling, and concatenation logic. Drift between the wrapper and the
delegate would silently corrupt SeedVR2's RoPE; this test fails loudly on any
future drift.

Imports are taken at module level. Heavy-import stubbing of
``comfy.model_management`` was attempted but is insufficient on the live import
chain (``comfy.ldm.seedvr.model`` pulls
``comfy.ldm.modules.diffusionmodules.model -> comfy.ops ->
comfy.memory_management -> comfy.quant_ops -> comfy_kitchen.tensor ->
torch._dynamo``), so every layer would have to be stubbed in lock-step;
running the test against the real modules instead is the fail-loud-from-real-
state approach this repo's tests follow.

The test uses a local ``torch.Generator`` so global RNG state is not mutated
(Copilot review on PR #21, finding #4) and ``torch.testing.assert_close`` with
``rtol=0, atol=0`` so any future kernel-precision drift is caught (PR #21,
finding #2; live ``max_abs_delta`` on ``issue_101`` HEAD is 0.0 across every
case). Parametrization covers non-default ``start_index`` and ``scale`` so the
wrapper's slicing/concatenation and scale-propagation logic are exercised, not
just the trivial ``rot_feats == t.shape[-1]`` happy path (PR #21, finding #3).
The previous version's session-scoped ``params: [...]`` print fixture was
removed (PR #21, finding #1).
"""

import pytest
import torch
import torch.testing

from comfy.ldm.flux.math import apply_rope1
from comfy.ldm.seedvr.model import apply_rotary_emb


def _direct_reproduction(freqs, t, start_index=0, scale=1.0):
    """Byte-for-byte reproduction of comfy/ldm/seedvr/model.py:471-505
    apply_rotary_emb body, calling apply_rope1 directly on the middle slice.
    """
    rot_feats = freqs.shape[-1]
    end_index = start_index + rot_feats
    t_left = t[..., :start_index]
    t_middle = t[..., start_index:end_index]
    t_right = t[..., end_index:]
    angles = freqs.to(t_middle.device)[..., ::2]
    cos = torch.cos(angles) * scale
    sin = torch.sin(angles) * scale
    col0 = torch.stack([cos, sin], dim=-1)
    col1 = torch.stack([-sin, cos], dim=-1)
    freqs_mat = torch.stack([col0, col1], dim=-1)
    t_middle_out = apply_rope1(t_middle, freqs_mat)
    return torch.cat((t_left, t_middle_out, t_right), dim=-1).type(t.dtype)


# (device, dtype, t_shape, freqs_shape, start_index, scale)
_CASES = [
    pytest.param("cpu", torch.float32, (1, 8, 16), (8, 16), 0, 1.0,
                 id="cpu-float32-base"),
    pytest.param("cpu", torch.float16, (1, 8, 16), (8, 16), 0, 1.0,
                 id="cpu-float16-base"),
    pytest.param("cpu", torch.bfloat16, (1, 8, 16), (8, 16), 0, 1.0,
                 id="cpu-bfloat16-base"),
    pytest.param("cpu", torch.float32, (2, 16, 32), (16, 32), 0, 1.0,
                 id="cpu-float32-larger"),
    pytest.param("cpu", torch.float32, (1, 8, 24), (8, 16), 4, 1.0,
                 id="cpu-float32-non-empty-left-and-right-slices"),
    pytest.param("cpu", torch.float32, (1, 8, 16), (8, 16), 0, 0.5,
                 id="cpu-float32-non-default-scale"),
    pytest.param(
        "cuda", torch.float16, (1, 8, 16), (8, 16), 0, 1.0,
        id="cuda-float16-base",
        marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="no cuda"),
    ),
]


@pytest.mark.parametrize("device,dtype,t_shape,freqs_shape,start_index,scale", _CASES)
def test_apply_rotary_emb_delegates_to_apply_rope1(
    device, dtype, t_shape, freqs_shape, start_index, scale
):
    generator = torch.Generator(device=device).manual_seed(0)
    t = torch.randn(*t_shape, dtype=dtype, device=device, generator=generator)
    freqs = torch.randn(*freqs_shape, dtype=dtype, device=device, generator=generator)

    wrapper_out = apply_rotary_emb(freqs, t, start_index=start_index, scale=scale)
    direct_out = _direct_reproduction(
        freqs, t, start_index=start_index, scale=scale
    )

    torch.testing.assert_close(
        wrapper_out,
        direct_out,
        rtol=0,
        atol=0,
        msg=lambda m: (
            f"apply_rotary_emb does not byte-match direct apply_rope1 reproduction "
            f"(device={device}, dtype={dtype}, t_shape={t_shape}, "
            f"freqs_shape={freqs_shape}, start_index={start_index}, scale={scale}): {m}"
        ),
    )
