"""Regression tests for SeedVR2 ``VideoAutoencoderKLWrapper.decode`` input guards.

Tracks pollockjj/mydevelopment#194 (parent #101). CodeRabbit thread
``r2962219551`` on upstream PR Comfy-Org/ComfyUI#11294 flagged that
``decode()`` accessed ``self.original_image_video`` and ``self.img_dims``
without validation, raising opaque torch errors deep in ``rearrange()`` /
attribute unpacking when a workflow wired the wrapper directly without
``SeedVR2InputProcessing.execute`` populating those attributes. The merged
fix on ``pollockjj/ComfyUI:issue_194`` adds two presence guards plus
shape/arity validation at the top of ``decode()``, raising ``RuntimeError``
with a SeedVR2-specific message identifying the missing or malformed
attribute, and initialises ``self.img_dims = None`` in ``__init__`` so the
missing-state branch is reachable from a default-constructed instance.

Five AC cells exercise every reachable input shape — missing
``original_image_video``, missing ``img_dims`` (parameterised over
explicit-``None`` and attribute-unset), wrong-rank ``original_image_video``,
wrong-arity ``img_dims`` (parameterised over 1-tuple and 3-tuple), and the
valid-both halt-at-post-guard sentinel — mirroring the ``__new__`` +
``nn.Module.__init__`` standin pattern from #189
(``test_seedvr_vae_decode_batch_axes.py``) and the ``_PostGuardReached``
halt-point pattern from #119 (``test_diffusers_metadata_guard.py``).
"""

from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

import comfy.ldm.seedvr.vae as vae_mod  # noqa: E402


class _PostGuardReached(Exception):
    """Sentinel raised by the patched ``VideoAutoencoderKL.decode_`` to
    prove the four guards passed and control reached the first post-guard
    callable in ``VideoAutoencoderKLWrapper.decode``.
    """


def _make_standin(*, original_image_video, img_dims, set_img_dims=True):
    """Build a ``VideoAutoencoderKLWrapper`` instance via ``__new__`` so
    the real ``__init__`` (which would allocate the full VAE weight set)
    does not run. ``nn.Module.__init__`` is invoked manually so that
    ``super()`` resolution inside the borrowed ``decode`` body still
    behaves correctly under ``VideoAutoencoderKLWrapper``'s MRO.

    ``set_img_dims=False`` skips the attribute assignment entirely so the
    guard's ``getattr(self, "img_dims", None)`` branch can be exercised
    against a genuinely unset attribute, not just an explicit ``None``.
    """
    wrapper = vae_mod.VideoAutoencoderKLWrapper.__new__(
        vae_mod.VideoAutoencoderKLWrapper
    )
    nn.Module.__init__(wrapper)
    wrapper.tiled_args = {"enable_tiling": False}
    wrapper.original_image_video = original_image_video
    if set_img_dims:
        wrapper.img_dims = img_dims
    return wrapper


def test_ac1_decode_raises_when_original_image_video_is_none():
    """AC1: ``original_image_video`` is ``None`` and ``img_dims`` is a valid
    ``(H, W)`` 2-tuple. ``decode(z)`` raises ``RuntimeError`` whose message
    mentions ``original_image_video`` and ``SeedVR2``. The unguarded
    ``rearrange(self.original_image_video, ...)`` call further down the
    body is therefore never reached.
    """
    wrapper = _make_standin(original_image_video=None, img_dims=(16, 16))
    z = torch.zeros(1, 16, 2, 2)

    with pytest.raises(RuntimeError) as excinfo:
        wrapper.decode(z)

    msg = str(excinfo.value)
    assert "original_image_video" in msg, (
        f"AC1 message must name `original_image_video`; got: {msg!r}"
    )
    assert "SeedVR2" in msg, (
        f"AC1 message must mention SeedVR2 to disambiguate from generic torch errors; got: {msg!r}"
    )


@pytest.mark.parametrize(
    "set_img_dims",
    [True, False],
    ids=["explicit-None", "attribute-unset"],
)
def test_ac2_decode_raises_when_img_dims_missing(set_img_dims):
    """AC2: ``original_image_video`` is a valid 5-D tensor and ``img_dims``
    is either ``None`` or genuinely unset on the instance. ``decode(z)``
    raises ``RuntimeError`` whose message mentions ``img_dims`` and
    ``SeedVR2``. The ``o_h, o_w = self.img_dims`` unpacking line further
    down the body is therefore never reached.
    """
    oiv = torch.zeros(1, 3, 1, 16, 16)
    wrapper = _make_standin(
        original_image_video=oiv,
        img_dims=None,
        set_img_dims=set_img_dims,
    )
    if not set_img_dims:
        assert not hasattr(wrapper, "img_dims"), (
            "Standin must have no `img_dims` attribute for the unset cell"
        )
    z = torch.zeros(1, 16, 2, 2)

    with pytest.raises(RuntimeError) as excinfo:
        wrapper.decode(z)

    msg = str(excinfo.value)
    assert "img_dims" in msg, (
        f"AC2 message must name `img_dims`; got: {msg!r}"
    )
    assert "SeedVR2" in msg, (
        f"AC2 message must mention SeedVR2; got: {msg!r}"
    )


def test_ac3_decode_raises_when_original_image_video_wrong_rank():
    """AC3: ``original_image_video`` is a non-5-D tensor (3-D here) and
    ``img_dims`` is a valid ``(H, W)`` 2-tuple. ``decode(z)`` raises
    ``RuntimeError`` whose message mentions shape / rank, signalling the
    rank guard fired before any tensor work.
    """
    oiv = torch.zeros(3, 16, 16)
    wrapper = _make_standin(original_image_video=oiv, img_dims=(16, 16))
    z = torch.zeros(1, 16, 2, 2)

    with pytest.raises(RuntimeError) as excinfo:
        wrapper.decode(z)

    msg = str(excinfo.value)
    assert "rank" in msg or "5-D" in msg or "shape" in msg, (
        f"AC3 message must mention rank/shape; got: {msg!r}"
    )
    assert "original_image_video" in msg, (
        f"AC3 message must name the offending attribute; got: {msg!r}"
    )


@pytest.mark.parametrize(
    "img_dims",
    [(16,), (16, 16, 16)],
    ids=["1-tuple", "3-tuple"],
)
def test_ac4_decode_raises_when_img_dims_wrong_arity(img_dims):
    """AC4: ``original_image_video`` is a valid 5-D tensor and ``img_dims``
    is a 1-tuple or 3-tuple (wrong arity). ``decode(z)`` raises
    ``RuntimeError`` whose message mentions ``img_dims`` and arity / dims,
    signalling the arity guard fired before the unpacking site.
    """
    oiv = torch.zeros(1, 3, 1, 16, 16)
    wrapper = _make_standin(original_image_video=oiv, img_dims=img_dims)
    z = torch.zeros(1, 16, 2, 2)

    with pytest.raises(RuntimeError) as excinfo:
        wrapper.decode(z)

    msg = str(excinfo.value)
    assert "img_dims" in msg, (
        f"AC4 message must name `img_dims`; got: {msg!r}"
    )
    assert "arity" in msg or "2-tuple" in msg, (
        f"AC4 message must mention arity / 2-tuple; got: {msg!r}"
    )


def test_ac5_decode_passes_guards_when_both_valid_and_halts_at_post_guard():
    """AC5: ``original_image_video`` is a valid 5-D tensor and ``img_dims``
    is a valid ``(H, W)`` 2-tuple. The four guards do not raise; control
    reaches the first post-guard callable. We patch
    ``VideoAutoencoderKL.decode_`` (the first patchable callable on the
    non-tiled branch the standin selects via ``tiled_args = {"enable_tiling": False}``)
    to raise ``_PostGuardReached`` — a sentinel exception that proves the
    guards passed and control crossed into the post-guard region without
    standing up the heavy decode pipeline (``super().decode_``,
    ``tiled_vae``, ``lab_color_transfer``).
    """
    oiv = torch.zeros(1, 3, 1, 16, 16)
    wrapper = _make_standin(original_image_video=oiv, img_dims=(16, 16))
    z = torch.zeros(1, 16, 2, 2)

    def _sentinel_decode_(self, latent, *args, **kwargs):
        raise _PostGuardReached("guards passed; post-guard region reached")

    with patch.object(vae_mod.VideoAutoencoderKL, "decode_", _sentinel_decode_):
        with pytest.raises(_PostGuardReached):
            wrapper.decode(z)
