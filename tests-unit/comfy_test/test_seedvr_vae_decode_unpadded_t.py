"""Regression test for issue #188 — ``SeedVR2InputProcessing.execute`` must
store the UNPADDED ``original_image_video`` so that
``VideoAutoencoderKLWrapper.decode`` trims its output to the user-visible
``T_in`` rather than the post-``cut_videos`` ``T_padded``.

Pre-fix behavior (``issue_101_pi`` HEAD): ``execute`` writes the PADDED
``images_bcthw`` (post ``cut_videos``) onto ``vae_model.original_image_video``;
``decode`` derives ``t = input.size(0) = B * T_padded`` and trims to
``T_padded`` for ``B == 1``. This silently expands ``T_in`` from
``{2, 3, 4} -> 5`` and ``{6, 7, 8} -> 9``.

Post-fix behavior (``issue_188`` HEAD): ``execute`` writes the UNPADDED
``images_bcthw_unpadded`` (pre ``cut_videos``); ``decode`` therefore trims to
``T_in`` for every ``T_in in {1..8}`` when ``B == 1``.

The test is parametrised over the full ``T_in in {1..8}`` range so that
``T_in in {1, 5}`` (unaffected by ``cut_videos`` padding — see the
``cut_videos`` definition in ``comfy_extras/nodes_seedvr.py``) act as
no-regression cases, and
``T_in in {2, 3, 4, 6, 7, 8}`` exercise the load-bearing fix path. AC-F is
locked in by patching ``lab_color_transfer`` with an identity stub that
records ``(content, style)`` into a closure-captured ``captured`` dict; the
test then asserts ``len(captured["args"][1]) == result.shape[2] * B``.

Test design rationale and per-decision review trail are recorded on the
tracking issue: https://github.com/pollockjj/mydevelopment/issues/188
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

# CPU-only CI fix: ``comfy.ldm.seedvr.vae`` transitively imports
# ``comfy.model_management``, whose import-time ``get_torch_device()`` call
# probes ``torch.cuda.current_device()`` unless ``comfy.cli_args.args.cpu`` is
# set. Match the pattern at
# ``tests-unit/comfy_test/test_seedvr_rope_delegation.py:42-52``: flip
# ``args.cpu`` before importing any ``comfy.ldm.*`` symbol.
from comfy.cli_args import args as _cli_args

if not torch.cuda.is_available():
    _cli_args.cpu = True

import comfy.model_management  # noqa: E402
import comfy.ldm.seedvr.vae as seedvr_vae  # noqa: E402
from comfy.ldm.seedvr.vae import (  # noqa: E402
    VideoAutoencoderKL,
    VideoAutoencoderKLWrapper,
)
from comfy_extras.nodes_seedvr import SeedVR2InputProcessing  # noqa: E402


def _t_padded(t_in: int) -> int:
    """Replicate ``cut_videos`` padding (see ``cut_videos`` in
    ``comfy_extras/nodes_seedvr.py``).

    Table: 1->1, 2->5, 3->5, 4->5, 5->5, 6->9, 7->9, 8->9.

    Hardcoded for test isolation; the ``test_t_padded_matches_cut_videos``
    sentinel below asserts this table stays in sync with the real
    ``cut_videos`` implementation, so any future drift in
    ``comfy_extras.nodes_seedvr.cut_videos`` will fail this module loudly.
    """
    if t_in == 1:
        return 1
    if t_in <= 4:
        return 5
    if (t_in - 1) % 4 == 0:
        return t_in
    return t_in + (4 - ((t_in - 1) % 4))


@pytest.mark.parametrize("t_in", [1, 2, 3, 4, 5, 6, 7, 8])
def test_t_padded_matches_cut_videos(t_in):
    """Drift sentinel — the local ``_t_padded`` table must agree with the
    real ``cut_videos`` output on a dummy ``(B=1, T_in, C=1, H=1, W=1)``
    tensor for every ``T_in in {1..8}``.
    """
    from comfy_extras.nodes_seedvr import cut_videos

    dummy = torch.zeros(1, t_in, 1, 1, 1)
    actual_t_padded = cut_videos(dummy).shape[1]
    assert actual_t_padded == _t_padded(t_in), (
        f"_t_padded({t_in}) returned {_t_padded(t_in)} but real "
        f"cut_videos produced T_padded={actual_t_padded}; the local table "
        f"has drifted from comfy_extras.nodes_seedvr.cut_videos."
    )


def _build_fake_vae(t_in: int, b: int = 1):
    """Construct a minimal vae stand-in for ``SeedVR2InputProcessing.execute``.

    Uses ``VideoAutoencoderKLWrapper.__new__`` to bypass ``nn.Module.__init__``
    so the encode/decode wrapper logic is exercised without weight loading.
    """
    fs = VideoAutoencoderKLWrapper.__new__(VideoAutoencoderKLWrapper)
    fs.original_image_video = None
    fs.tiled_args = {}
    fs.img_dims = None
    fs.enable_tiling = False
    fs.spatial_downsample_factor = 8
    fs.temporal_downsample_factor = 4
    fs.freeze_encoder = True

    def _fake_encode(images_bthwc):
        bb = images_bthwc.shape[0]
        return torch.zeros(bb, 16, _t_padded(t_in), 4, 4)

    return SimpleNamespace(
        patcher=object(),
        first_stage_model=fs,
        encode=_fake_encode,
        encode_tiled=lambda *a, **k: _fake_encode(a[0] if a else k.get("images")),
    )


@pytest.mark.parametrize("t_in", [1, 2, 3, 4, 5, 6, 7, 8])
def test_decode_returns_tin_frames(t_in):
    b, h, w = 1, 32, 32
    images = torch.zeros(b, t_in, h, w, 3)  # BTHWC contract for execute()

    vae = _build_fake_vae(t_in, b=b)

    def _fake_decode_(self, z, return_dict: bool = True):
        # Real ``VideoAutoencoderKL.decode_`` returns a 5D tensor
        # ``(b, 3, T_padded, h, w)``. Wrapper applies ``.squeeze(2)`` after,
        # which only collapses when ``T_padded == 1`` (matches real shape).
        out = torch.zeros(b, 3, _t_padded(t_in), h, w)
        return (out,) if not return_dict else out

    captured = {}

    def _identity_lab_spy(content, style, *_a, **_k):
        captured["args"] = (content, style)
        return content

    with (
        patch.object(
            comfy.model_management,
            "load_models_gpu",
            lambda *a, **k: None,
        ),
        patch(
            "comfy_extras.nodes_seedvr.clear_vae_memory",
            lambda _vae: None,
        ),
        patch.object(VideoAutoencoderKL, "decode_", new=_fake_decode_),
        patch.object(seedvr_vae, "lab_color_transfer", new=_identity_lab_spy),
    ):
        SeedVR2InputProcessing.execute(
            images,
            vae,
            resolution=h,
            spatial_tile_size=h,
            spatial_overlap=8,
            temporal_tile_size=5,
            enable_tiling=False,
        )

        # Decode entry-point shape contract (from vae.py decode):
        #   b, tc, h, w = z.shape
        #   latent = z.view(b, 16, -1, h, w)
        # so z must be 4D with channels = 16 * T_latent.
        synthetic_z = torch.zeros(b, 16 * _t_padded(t_in), 4, 4)
        result = vae.first_stage_model.decode(synthetic_z)

    # AC-A / AC-C / AC-D / AC-E load-bearing assertion.
    assert result.shape[2] == t_in, (
        f"decode returned T_out={result.shape[2]} for T_in={t_in}; "
        f"expected T_out == T_in. Bug locked in by issue #188 — "
        f"original_image_video must carry UNPADDED T_in, not T_padded="
        f"{_t_padded(t_in)}."
    )

    # AC-F: lab_color_transfer received `input` with len == T_out * B.
    assert "args" in captured, "lab_color_transfer was not invoked"
    style_arg = captured["args"][1]
    assert len(style_arg) == result.shape[2] * b, (
        f"lab_color_transfer style arg has len={len(style_arg)}; expected "
        f"{result.shape[2] * b} (T_out * B). AC-F violated."
    )
