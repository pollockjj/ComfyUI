"""Regression: ``comfy.ldm.seedvr.vae.VideoAutoencoderKLWrapper.forward``
must use the tensor returned by ``self.decode(z)`` directly — no
``.sample`` access on the tensor return. ``forward()`` returns the
triple ``(x_out, z, p)`` where ``x_out`` is the decoded tensor, ``z`` is
the latent tensor, and ``p`` is the posterior produced by ``encode()``.

Source: CodeRabbit review on Comfy-Org/ComfyUI#11294 thread
https://github.com/Comfy-Org/ComfyUI/pull/11294#discussion_r2959796348
("Also applies to: 2083-2087" trailer pointing at the wrapper).
Tracker: pollockjj/mydevelopment#192. Sister to #190 which fixed the
parent ``VideoAutoencoderKL.forward`` for the same bug class.

The pre-fix body raised ``AttributeError: 'Tensor' object has no
attribute 'sample'`` on direct wrapper invocation because
``VideoAutoencoderKLWrapper.decode`` returns a plain tensor, not a
diffusers wrapper. The post-fix body uses the tensor return directly.

Tests construct a CPU-only wrapper standin via ``__new__`` plus
``nn.Module.__init__`` (bypassing the real init that would otherwise
allocate the full VAE weight set), register a single dummy parameter
so ``next(self.parameters()).dtype`` resolves inside the wrapper's
``encode``, set ``original_image_video`` / ``img_dims`` /
``tiled_args`` so the wrapper's ``decode`` guards pass, and patch the
parent ``VideoAutoencoderKL.encode`` / ``decode_`` plus the module-
level ``lab_color_transfer`` with fingerprint-tagged stubs so the
encode → decode round trip can be probed without real weights.
"""

import inspect
from unittest.mock import patch

import torch
import torch.nn as nn

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

import comfy.ldm.seedvr.vae as vae_mod  # noqa: E402


_INPUT_SHAPE = (1, 3, 5, 16, 16)
_PARENT_ENCODE_OUT_SHAPE = (1, 16, 1, 2, 2)
_LATENT_SHAPE = (1, 16, 2, 2)
_PARENT_DECODE_OUT_SHAPE = (1, 3, 5, 16, 16)
_FORWARD_OUT_SHAPE = (1, 3, 5, 16, 16)

_PARENT_ENCODE_FINGERPRINT = 7.0
_PARENT_DECODE_FINGERPRINT = 13.0


def _make_wrapper() -> vae_mod.VideoAutoencoderKLWrapper:
    wrapper = vae_mod.VideoAutoencoderKLWrapper.__new__(
        vae_mod.VideoAutoencoderKLWrapper
    )
    nn.Module.__init__(wrapper)
    wrapper.register_parameter(
        "_stub_dtype_param", nn.Parameter(torch.zeros(1))
    )
    wrapper.spatial_downsample_factor = 8
    wrapper.temporal_downsample_factor = 4
    wrapper.freeze_encoder = True
    wrapper.enable_tiling = False
    wrapper.tiled_args = {"enable_tiling": False}
    wrapper.original_image_video = torch.zeros(*_INPUT_SHAPE)
    wrapper.img_dims = (16, 16)
    return wrapper


def _stub_parent_encode(self, x, return_dict=True):
    return torch.full(
        _PARENT_ENCODE_OUT_SHAPE, _PARENT_ENCODE_FINGERPRINT
    )


def _stub_parent_decode_(self, z, return_dict=True):
    B = int(z.shape[0])
    return torch.full(
        (B, 3, 5, 16, 16), _PARENT_DECODE_FINGERPRINT
    )


def _lab_color_passthrough(x, input):
    return x


def _patches():
    return (
        patch.object(
            vae_mod.VideoAutoencoderKL, "encode", _stub_parent_encode
        ),
        patch.object(
            vae_mod.VideoAutoencoderKL, "decode_", _stub_parent_decode_
        ),
        patch.object(
            vae_mod, "lab_color_transfer", _lab_color_passthrough
        ),
    )


def test_forward_returns_three_tensor_triple_no_attribute_error():
    wrapper = _make_wrapper()
    x = torch.zeros(*_INPUT_SHAPE)
    p_encode, p_decode, p_lab = _patches()
    with p_encode, p_decode, p_lab:
        result = wrapper.forward(x)
    assert isinstance(result, tuple)
    assert len(result) == 3
    x_out, z, p = result
    assert type(x_out) is torch.Tensor
    assert type(z) is torch.Tensor
    assert type(p) is torch.Tensor


def test_forward_x_out_shape_dtype_and_fingerprint_binary_equal():
    wrapper = _make_wrapper()
    x = torch.zeros(*_INPUT_SHAPE)
    p_encode, p_decode, p_lab = _patches()
    with p_encode, p_decode, p_lab:
        x_out, z, p = wrapper.forward(x)
    assert x_out.shape == torch.Size(_FORWARD_OUT_SHAPE)
    assert x_out.dtype == torch.float32
    expected_x_out = torch.full(
        _FORWARD_OUT_SHAPE, _PARENT_DECODE_FINGERPRINT
    )
    assert torch.equal(x_out, expected_x_out)


def test_forward_z_matches_encode_side_latent_squeeze():
    wrapper = _make_wrapper()
    x = torch.zeros(*_INPUT_SHAPE)
    p_encode, p_decode, p_lab = _patches()
    with p_encode, p_decode, p_lab:
        x_out, z, p = wrapper.forward(x)
    expected_p = torch.full(
        _PARENT_ENCODE_OUT_SHAPE, _PARENT_ENCODE_FINGERPRINT
    )
    assert p.shape == torch.Size(_PARENT_ENCODE_OUT_SHAPE)
    assert torch.equal(p, expected_p)
    assert z.shape == torch.Size(_LATENT_SHAPE)
    assert torch.equal(z, expected_p.squeeze(2))


def test_forward_source_has_no_sample_access():
    src = inspect.getsource(vae_mod.VideoAutoencoderKLWrapper.forward)
    assert ".sample" not in src
