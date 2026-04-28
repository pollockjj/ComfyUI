"""Regression test: ``comfy.ldm.seedvr.model.apply_rotary_emb`` must delegate
to ``comfy.ldm.flux.math.apply_rope1`` and produce exact-equality output
across the wrapper's slicing, scaling, and concatenation logic. Drift between
the wrapper and the delegate would silently corrupt SeedVR2's RoPE; this test
fails loudly on any future drift.

Each parametrized case both:

1. Patches ``comfy.ldm.seedvr.model.apply_rope1`` with a ``wraps``-style spy
   and asserts ``spy.call_count >= 1`` so a future change that inlines the
   math and stops calling ``apply_rope1`` fails the test.
2. Compares the wrapper's output against a hand-rolled reproduction using
   ``torch.testing.assert_close(rtol=0, atol=0)`` -- exact tensor equality,
   not bit-equality (``+0.0`` vs ``-0.0`` and NaN payloads can still match);
   the assertion catches any future kernel-precision drift in the
   ``apply_rope1`` dispatch.

The test uses a local ``torch.Generator`` so global RNG state is not mutated.
Parametrization covers non-default ``start_index`` and ``scale`` and a case
where ``freqs.shape[0] > t.shape[seq_dim]`` so the wrapper's
``slice_at_dim(freqs, slice(-seq_len, None), dim=0)`` path is exercised.
Imports are taken at module level; heavy-import stubbing of
``comfy.model_management`` was attempted but is insufficient on the live
import chain (``comfy.ldm.seedvr.model`` pulls
``comfy.ldm.modules.diffusionmodules.model -> comfy.ops ->
comfy.memory_management -> comfy.quant_ops -> comfy_kitchen.tensor ->
torch._dynamo``), so running the test against the real modules is the
fail-loud-from-real-state approach this repo's tests follow.

Test design rationale and per-decision review trail are recorded on the
tracking issue: https://github.com/pollockjj/mydevelopment/issues/120
"""

from unittest.mock import patch

import pytest
import torch

# CPU-only CI fix: ``comfy.ldm.seedvr.model`` transitively imports
# ``comfy.model_management``, whose import-time ``get_torch_device()`` call
# probes ``torch.cuda.current_device()`` unless ``comfy.cli_args.args.cpu`` is
# set. On a CPU-only build that probe can raise during test collection before
# the ``cuda`` case has had a chance to be skipped. Match the pattern used by
# ``tests-unit/comfy_quant/test_mixed_precision.py``: flip ``args.cpu`` before
# importing any ``comfy.ldm.*`` symbol.
from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

import comfy.ldm.seedvr.model as seedvr_model  # noqa: E402
from comfy.ldm.flux.math import apply_rope1  # noqa: E402
from comfy.ldm.seedvr.model import apply_rotary_emb  # noqa: E402


def _direct_reproduction(freqs, t, start_index=0, scale=1.0, seq_dim=-2):
    """Reproduce the body of ``apply_rotary_emb`` for the default case where
    ``freqs.ndim == 2`` and ``t.ndim == 3`` (implicit ``freqs_seq_dim=0``).
    Mirrors the wrapper's ``slice_at_dim(freqs, slice(-seq_len, None), dim=0)``
    step when freqs is longer than ``t`` along ``seq_dim``. Calls the real
    ``apply_rope1`` via the test module's import (the test patches the
    ``seedvr_model.apply_rope1`` attribute; this call uses the unpatched
    ``flux.math`` symbol).
    """
    if freqs.ndim == 2 and t.ndim == 3:
        seq_len = t.shape[seq_dim]
        freqs = freqs[-seq_len:]

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
    pytest.param("cpu", torch.float32, (1, 8, 16), (12, 16), 0, 1.0,
                 id="cpu-float32-freqs-longer-than-seq"),
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

    # Patch the apply_rope1 symbol as imported into seedvr.model with a wraps
    # spy: a future change that inlines the math and stops calling the
    # imported apply_rope1 makes spy.call_count == 0 and fails the test.
    with patch.object(
        seedvr_model, "apply_rope1", wraps=seedvr_model.apply_rope1
    ) as spy:
        wrapper_out = apply_rotary_emb(
            freqs, t, start_index=start_index, scale=scale
        )

    assert spy.call_count >= 1, (
        "apply_rotary_emb did not call comfy.ldm.seedvr.model.apply_rope1; "
        "the delegation invariant is broken"
    )

    direct_out = _direct_reproduction(
        freqs, t, start_index=start_index, scale=scale
    )

    msg = (
        f"apply_rotary_emb output does not match direct apply_rope1 "
        f"reproduction (device={device}, dtype={dtype}, t_shape={t_shape}, "
        f"freqs_shape={freqs_shape}, start_index={start_index}, scale={scale})"
    )
    torch.testing.assert_close(
        wrapper_out,
        direct_out,
        rtol=0,
        atol=0,
        msg=msg,
    )
