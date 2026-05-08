"""Regression: ``comfy.ldm.seedvr.vae.VideoAutoencoderKLWrapper.decode``
must keep batch and time axes distinct for every ``(B, T_orig)`` cell,
including the ``B>1, T_orig==1`` image-mode batched-decode path.

The pre-fix code applied ``super().decode_(latent).squeeze(2)`` followed
by ``if x.ndim == 4: x = x.unsqueeze(0)`` and a ``size(1)==1`` heuristic,
which mis-routed the batch axis into the channel axis and the channel
axis into the time axis when ``B>1`` and ``T_dec==1``.

These tests construct a CPU-only wrapper instance via ``__new__`` +
``nn.Module.__init__`` (bypassing the real init that would otherwise
allocate VAE weights) and patch ``VideoAutoencoderKL.decode_`` with a
fingerprint-tagged stub returning ``[B, 3, T_dec, 16, 16]`` filled with
``float(b + 1)`` per batch index, plus a passthrough
``lab_color_transfer``. The post-fix invariant is
``tuple(out.shape) == (1, 3, B*T_orig, 16, 16)`` for every cell, with
the per-sample fingerprint preserved at the batch position.

The stacked-vs-individual test pins per-sample ordering by feeding the
same per-batch-index fingerprint into a stacked decode and two
single-batch decodes, then comparing the resulting per-sample slabs
under ``torch.equal``.
"""

from unittest.mock import patch

import torch
import torch.nn as nn

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

import comfy.ldm.seedvr.vae as vae_mod  # noqa: E402


def _make_wrapper(B: int, T_orig: int) -> vae_mod.VideoAutoencoderKLWrapper:
    wrapper = vae_mod.VideoAutoencoderKLWrapper.__new__(
        vae_mod.VideoAutoencoderKLWrapper
    )
    nn.Module.__init__(wrapper)
    wrapper.tiled_args = {"enable_tiling": False}
    wrapper.original_image_video = torch.zeros(B, 3, T_orig, 16, 16)
    wrapper.img_dims = (16, 16)
    return wrapper


def _fingerprint_decode_(self, z, return_dict=True):
    B = int(z.shape[0])
    T_dec = int(z.shape[2])
    H_in = int(z.shape[3])
    W_in = int(z.shape[4])
    out = torch.empty(B, 3, T_dec, H_in * 8, W_in * 8)
    for b in range(B):
        out[b].fill_(float(b + 1))
    return out


def _lab_color_passthrough(x, input):
    return x


def _decode_with_patches(wrapper, z):
    with patch.object(vae_mod.VideoAutoencoderKL, "decode_", _fingerprint_decode_), \
         patch.object(vae_mod, "lab_color_transfer", _lab_color_passthrough):
        return wrapper.decode(z)


def test_decode_b1_t1_shape_and_ordering_correct():
    B, T_orig = 1, 1
    wrapper = _make_wrapper(B, T_orig)
    z = torch.zeros(B, 16 * T_orig, 2, 2)
    out = _decode_with_patches(wrapper, z)
    assert tuple(out.shape) == (1, 3, 1, 16, 16)
    assert out[0, 0, 0, 0, 0].item() == 1.0


def test_decode_b1_t5_video_shape_unchanged():
    B, T_orig = 1, 5
    wrapper = _make_wrapper(B, T_orig)
    z = torch.zeros(B, 16 * T_orig, 2, 2)
    out = _decode_with_patches(wrapper, z)
    assert tuple(out.shape) == (1, 3, 5, 16, 16)


def test_decode_b2_t1_fixes_batch_time_axes():
    B, T_orig = 2, 1
    wrapper = _make_wrapper(B, T_orig)
    z = torch.zeros(B, 16 * T_orig, 2, 2)
    out = _decode_with_patches(wrapper, z)
    assert tuple(out.shape) == (1, 3, 2, 16, 16)
    assert out[0, 0, 0, 0, 0].item() == 1.0
    assert out[0, 0, 1, 0, 0].item() == 2.0


def test_decode_b4_t1_fixes_batch_time_axes():
    B, T_orig = 4, 1
    wrapper = _make_wrapper(B, T_orig)
    z = torch.zeros(B, 16 * T_orig, 2, 2)
    out = _decode_with_patches(wrapper, z)
    assert tuple(out.shape) == (1, 3, 4, 16, 16)
    assert [out[0, 0, b, 0, 0].item() for b in range(4)] == [1.0, 2.0, 3.0, 4.0]


def test_decode_b2_t3_multi_frame_batch_unchanged():
    B, T_orig = 2, 3
    wrapper = _make_wrapper(B, T_orig)
    z = torch.zeros(B, 16 * T_orig, 2, 2)
    out = _decode_with_patches(wrapper, z)
    assert tuple(out.shape) == (1, 3, 6, 16, 16)


def _tiled_vae_4d_stub(latent, vae_model, **kwargs):
    """Mimic real ``tiled_vae``'s sf_t==1 + T_lat==1 squeeze branch
    (see ``comfy/ldm/seedvr/vae.py`` line 179-180): return a 4D tensor
    so the wrapper's post-decode pipeline must re-add the temporal
    axis on the tiled path.
    """
    B = int(latent.shape[0])
    H = int(latent.shape[3]) * 8
    W = int(latent.shape[4]) * 8
    out = torch.empty(B, 3, H, W)
    for b in range(B):
        out[b].fill_(float(b + 1))
    return out


def test_decode_tiled_sf_t1_single_frame_4d_output_normalized():
    """Codex P2 / Copilot finding on PR #34: ``tiled_vae`` returns 4D
    when ``temporal_downsample_factor == 1`` AND latent T == 1, so the
    wrapper must re-add the temporal axis on the tiled branch before
    the rearrange ``b c t h w -> (b t) c h w``. Pre-fix this case raised
    an einops ``EinopsError`` because the patch removed the only
    ``x.ndim == 4`` normalization.
    """
    wrapper = vae_mod.VideoAutoencoderKLWrapper.__new__(
        vae_mod.VideoAutoencoderKLWrapper
    )
    nn.Module.__init__(wrapper)
    wrapper.tiled_args = {"enable_tiling": True}
    wrapper.original_image_video = torch.zeros(1, 3, 1, 16, 16)
    wrapper.img_dims = (16, 16)

    z = torch.zeros(1, 16, 2, 2)

    with patch.object(vae_mod, "tiled_vae", _tiled_vae_4d_stub), \
         patch.object(vae_mod, "lab_color_transfer", _lab_color_passthrough):
        out = wrapper.decode(z)

    assert tuple(out.shape) == (1, 3, 1, 16, 16)
    assert out[0, 0, 0, 0, 0].item() == 1.0


def test_decode_tiled_sf_t1_b2_t1_per_sample_ordering():
    """Copilot follow-up 3184602213 on PR #34: the tiled-path 4D->5D
    normalization must preserve distinct batch/time axes for ``B>1``
    too, not only ``B=1``. Mirrors
    ``test_decode_b2_t1_fixes_batch_time_axes`` for the
    ``enable_tiling=True`` path with a wrapper whose
    ``temporal_downsample_factor == 1`` AND latent T == 1, where
    ``tiled_vae`` returns 4D and the wrapper must re-add the temporal
    axis without collapsing batch into channels.
    """
    B, T_orig = 2, 1
    wrapper = vae_mod.VideoAutoencoderKLWrapper.__new__(
        vae_mod.VideoAutoencoderKLWrapper
    )
    nn.Module.__init__(wrapper)
    wrapper.tiled_args = {"enable_tiling": True}
    wrapper.original_image_video = torch.zeros(B, 3, T_orig, 16, 16)
    wrapper.img_dims = (16, 16)

    z = torch.zeros(B, 16 * T_orig, 2, 2)

    with patch.object(vae_mod, "tiled_vae", _tiled_vae_4d_stub), \
         patch.object(vae_mod, "lab_color_transfer", _lab_color_passthrough):
        out = wrapper.decode(z)

    assert tuple(out.shape) == (1, 3, 2, 16, 16)
    assert out[0, 0, 0, 0, 0].item() == 1.0
    assert out[0, 0, 1, 0, 0].item() == 2.0


def test_decode_b2_t1_stacked_equals_individual_per_sample_ordering():
    wrapper = _make_wrapper(2, 1)
    z_stacked = torch.zeros(2, 16, 2, 2)
    out_stacked = _decode_with_patches(wrapper, z_stacked)

    wrapper.original_image_video = torch.zeros(1, 3, 1, 16, 16)

    def _decode_pinned(value):
        def _stub(self, z, return_dict=True):
            B = int(z.shape[0])
            T_dec = int(z.shape[2])
            H_in = int(z.shape[3])
            W_in = int(z.shape[4])
            out = torch.empty(B, 3, T_dec, H_in * 8, W_in * 8)
            out.fill_(value)
            return out
        return _stub

    z_individual = torch.zeros(1, 16, 2, 2)

    with patch.object(vae_mod.VideoAutoencoderKL, "decode_", _decode_pinned(1.0)), \
         patch.object(vae_mod, "lab_color_transfer", _lab_color_passthrough):
        out_individual_0 = wrapper.decode(z_individual)

    with patch.object(vae_mod.VideoAutoencoderKL, "decode_", _decode_pinned(2.0)), \
         patch.object(vae_mod, "lab_color_transfer", _lab_color_passthrough):
        out_individual_1 = wrapper.decode(z_individual)

    assert torch.equal(out_stacked[0, :, 0, :, :], out_individual_0[0, :, 0, :, :])
    assert torch.equal(out_stacked[0, :, 1, :, :], out_individual_1[0, :, 0, :, :])
