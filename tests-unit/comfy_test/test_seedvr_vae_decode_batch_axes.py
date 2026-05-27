from unittest.mock import patch

import torch
import torch.nn as nn

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

import comfy.ldm.seedvr.vae as vae_mod  # noqa: E402


def _make_wrapper() -> vae_mod.VideoAutoencoderKLWrapper:
    wrapper = vae_mod.VideoAutoencoderKLWrapper.__new__(
        vae_mod.VideoAutoencoderKLWrapper
    )
    nn.Module.__init__(wrapper)
    return wrapper


def _fingerprint_decode_(self, z, return_dict=True):
    b = int(z.shape[0])
    t = int(z.shape[2])
    h = int(z.shape[3])
    w = int(z.shape[4])
    out = torch.empty(b, 3, t, h * 8, w * 8)
    for batch_idx in range(b):
        out[batch_idx].fill_(float(batch_idx + 1))
    return out


def _decode_with_patches(wrapper, z):
    with patch.object(vae_mod.VideoAutoencoderKL, "decode_", _fingerprint_decode_):
        return wrapper.decode(z)


def test_decode_b1_t5_video_shape_unchanged():
    wrapper = _make_wrapper()

    out = _decode_with_patches(wrapper, torch.zeros(1, 16 * 5, 2, 2))

    assert tuple(out.shape) == (1, 3, 5, 16, 16)


def test_decode_b2_t3_multi_frame_batch_unchanged():
    wrapper = _make_wrapper()

    out = _decode_with_patches(wrapper, torch.zeros(2, 16 * 3, 2, 2))

    assert tuple(out.shape) == (2, 3, 3, 16, 16)


def test_decode_b2_t1_stacked_equals_individual_per_sample_ordering():
    wrapper = _make_wrapper()
    out_stacked = _decode_with_patches(wrapper, torch.zeros(2, 16, 2, 2))

    def _decode_pinned(value):
        def _stub(self, z, return_dict=True):
            b = int(z.shape[0])
            t = int(z.shape[2])
            h = int(z.shape[3])
            w = int(z.shape[4])
            return torch.full((b, 3, t, h * 8, w * 8), value)
        return _stub

    with patch.object(vae_mod.VideoAutoencoderKL, "decode_", _decode_pinned(1.0)):
        out_individual_0 = wrapper.decode(torch.zeros(1, 16, 2, 2))

    with patch.object(vae_mod.VideoAutoencoderKL, "decode_", _decode_pinned(2.0)):
        out_individual_1 = wrapper.decode(torch.zeros(1, 16, 2, 2))

    assert torch.equal(out_stacked[0, :, 0, :, :], out_individual_0[0, :, 0, :, :])
    assert torch.equal(out_stacked[1, :, 0, :, :], out_individual_1[0, :, 0, :, :])
