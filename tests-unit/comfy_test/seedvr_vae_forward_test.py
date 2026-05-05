"""Regression: ``comfy.ldm.seedvr.vae.VideoAutoencoderKL.forward`` must
honor the actual tensor/tuple return contract of ``encode()`` and
``decode_()`` and must NOT dereference diffusers-style ``.latent_dist``
or ``.sample`` attributes on those returns.

Source: CodeRabbit review on Comfy-Org/ComfyUI#11294 thread
https://github.com/Comfy-Org/ComfyUI/pull/11294#discussion_r2959796348 .
Tracker: pollockjj/mydevelopment#190.

The pre-fix body raised ``AttributeError: 'Tensor' object has no
attribute 'latent_dist'`` for ``mode in {"encode", "all"}`` and
``AttributeError: 'VideoAutoencoderKL' object has no attribute 'decode'``
for ``mode == "decode"`` (the class only defines ``decode_`` with a
trailing underscore). The post-fix body unwraps the optional one-element
tuple shape that ``return_dict=False`` produces and returns the tensor
directly, and it forwards ``**kwargs`` to ``encode``/``decode_`` so the
tuple path is reachable via the public ``forward`` API.

Tests construct a stub subclass of ``VideoAutoencoderKL`` that bypasses
the heavy ``__init__`` via ``torch.nn.Module.__init__(self)`` and
overrides ``encode``/``decode_`` with sentinel-valued tensors so the
contract can be probed without loading any real VAE weights. The stubs
honor ``return_dict`` (returning a one-element tuple when
``return_dict=False``) so tests cover both branches of the
tuple-unwrap path. ``decode_`` records its input so ``mode="all"``
can pin the encode->decode composition.
"""

import inspect
import re

import torch
import torch.nn as nn

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

from comfy.ldm.seedvr.vae import VideoAutoencoderKL  # noqa: E402


_LATENT_SHAPE = (1, 16, 2, 2, 2)
_DECODED_SHAPE = (1, 3, 5, 16, 16)
_INPUT_ENCODE_SHAPE = (1, 3, 5, 16, 16)
_INPUT_DECODE_SHAPE = (1, 16, 2, 2, 2)

# Sentinel scalar values let tests verify that ``decode_`` received the
# encode output and not the original ``x`` (or any other unintended tensor).
_ENCODE_SENTINEL = 0.4242
_DECODE_SENTINEL = 0.7373


class _StubVAE(VideoAutoencoderKL):
    def __init__(self):
        nn.Module.__init__(self)
        self._encode_out = torch.full(_LATENT_SHAPE, _ENCODE_SENTINEL)
        self._decode_out = torch.full(_DECODED_SHAPE, _DECODE_SENTINEL)
        self.encode_calls = []
        self.decode_calls = []

    def encode(self, x, return_dict=True):
        self.encode_calls.append({"x": x, "return_dict": return_dict})
        if not return_dict:
            return (self._encode_out,)
        return self._encode_out

    def decode_(self, z, return_dict=True):
        self.decode_calls.append({"z": z, "return_dict": return_dict})
        if not return_dict:
            return (self._decode_out,)
        return self._decode_out


def test_forward_encode_returns_tensor():
    vae = _StubVAE()
    x = torch.zeros(*_INPUT_ENCODE_SHAPE)
    result = vae.forward(x, mode="encode")
    assert type(result) is torch.Tensor
    assert result.shape == torch.Size(_LATENT_SHAPE)
    assert torch.equal(result, vae._encode_out)
    assert len(vae.encode_calls) == 1
    assert vae.encode_calls[0]["return_dict"] is True
    assert len(vae.decode_calls) == 0


def test_forward_decode_returns_tensor():
    vae = _StubVAE()
    z = torch.zeros(*_INPUT_DECODE_SHAPE)
    result = vae.forward(z, mode="decode")
    assert type(result) is torch.Tensor
    assert result.shape == torch.Size(_DECODED_SHAPE)
    assert torch.equal(result, vae._decode_out)
    assert len(vae.encode_calls) == 0
    assert len(vae.decode_calls) == 1
    assert vae.decode_calls[0]["return_dict"] is True


def test_forward_all_returns_tensor():
    vae = _StubVAE()
    x = torch.zeros(*_INPUT_ENCODE_SHAPE)
    result = vae.forward(x, mode="all")
    assert type(result) is torch.Tensor
    assert result.shape == torch.Size(_DECODED_SHAPE)
    assert torch.equal(result, vae._decode_out)


def test_forward_all_pins_encode_then_decode_composition():
    """``mode='all'`` must call ``encode(x)`` and then feed that output
    into ``decode_``. A regression that decoded the original ``x`` (or any
    other tensor) instead of the encode output must fail this test.
    """
    vae = _StubVAE()
    x = torch.zeros(*_INPUT_ENCODE_SHAPE)
    vae.forward(x, mode="all")
    assert len(vae.encode_calls) == 1
    assert len(vae.decode_calls) == 1
    assert torch.equal(vae.encode_calls[0]["x"], x)
    decode_input = vae.decode_calls[0]["z"]
    assert decode_input.shape == torch.Size(_LATENT_SHAPE)
    assert torch.equal(decode_input, vae._encode_out)


def test_forward_encode_with_return_dict_false_unwraps_tuple():
    """When ``return_dict=False`` is forwarded, ``encode`` returns a
    one-element tuple. ``forward`` must unwrap to the underlying tensor.
    A regression that drops ``_unwrap`` (or fails to forward kwargs)
    must fail this test.
    """
    vae = _StubVAE()
    x = torch.zeros(*_INPUT_ENCODE_SHAPE)
    result = vae.forward(x, mode="encode", return_dict=False)
    assert type(result) is torch.Tensor
    assert result.shape == torch.Size(_LATENT_SHAPE)
    assert torch.equal(result, vae._encode_out)
    assert len(vae.encode_calls) == 1
    assert vae.encode_calls[0]["return_dict"] is False


def test_forward_decode_with_return_dict_false_unwraps_tuple():
    vae = _StubVAE()
    z = torch.zeros(*_INPUT_DECODE_SHAPE)
    result = vae.forward(z, mode="decode", return_dict=False)
    assert type(result) is torch.Tensor
    assert result.shape == torch.Size(_DECODED_SHAPE)
    assert torch.equal(result, vae._decode_out)
    assert len(vae.decode_calls) == 1
    assert vae.decode_calls[0]["return_dict"] is False


def test_forward_all_with_return_dict_false_unwraps_tuple():
    vae = _StubVAE()
    x = torch.zeros(*_INPUT_ENCODE_SHAPE)
    result = vae.forward(x, mode="all", return_dict=False)
    assert type(result) is torch.Tensor
    assert result.shape == torch.Size(_DECODED_SHAPE)
    assert torch.equal(result, vae._decode_out)
    assert len(vae.encode_calls) == 1
    assert len(vae.decode_calls) == 1
    assert vae.encode_calls[0]["return_dict"] is False
    assert vae.decode_calls[0]["return_dict"] is False
    assert torch.equal(vae.decode_calls[0]["z"], vae._encode_out)


def test_forward_source_has_no_diffusers_attr_access():
    src = inspect.getsource(VideoAutoencoderKL.forward)
    assert ".latent_dist" not in src
    assert ".sample" not in src
    assert re.search(r"self\.decode\(", src) is None
