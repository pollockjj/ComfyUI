"""Regression tests for the VAE diffusers-format guard at ``comfy/sd.py:443-446``.

Tracks pollockjj/mydevelopment#119 (parent #101). The guard previously
indexed ``metadata["keep_diffusers_format"]`` directly, raising
``KeyError`` when ``metadata`` was non-``None`` but lacked that key —
the common case for any safetensors VAE with ordinary metadata.
CodeRabbit flagged this as Critical on Comfy-Org/ComfyUI#11294
(thread r2959796358). The merged fix on ``pollockjj/ComfyUI:issue_101``
uses ``metadata.get("keep_diffusers_format") != "true"``: a missing
key flows through to invoke ``convert_vae_state_dict`` (the rewrite),
while the explicit ``"true"`` value bypasses it.

The two test cells below are the binary AC contract that the qa-slice
gate evaluates verbatim. Each follows the contract shape exactly:

  * ``unittest.mock.patch.object(comfy.sd.diffusers_convert,
    "convert_vae_state_dict", autospec=True)`` — patches the symbol
    that ``comfy/sd.py:446`` resolves through (``from . import
    diffusers_convert``); ``autospec=True`` pins the mock's signature
    to the production function so a signature drift fails loudly.
  * ``_make_standin()`` — borrows ``comfy.sd.VAE.__init__`` onto a
    bare class, mirroring the precedent set in
    ``tests-unit/comfy_test/seedvr_model_test.py::_make_standin``
    (#109). The guard sits inline in the constructor at lines 443-446,
    above any subsystem instantiation, so binding ``__init__`` is
    sufficient to drive it.
  * ``contextlib.suppress(Exception)`` — the post-guard model-detection
    chain in ``VAE.__init__`` continues past line 446 and may raise on
    a synthetic single-key state dict. The guard's behaviour (whether
    ``convert_vae_state_dict`` was called) is captured before that
    point, so suppressing downstream failures keeps the cell binary.
  * ``mock_convert.call_count`` assertion — AC1 expects 1 (entered
    the conversion branch); AC2 expects 0 (skipped via
    ``keep_diffusers_format == "true"``).

The trigger key ``decoder.up_blocks.0.resnets.0.norm1.weight`` is
hard-coded per the issue's stop-condition: an upstream rename should
become a loud-fail signal, not a silent test miss.
"""

from comfy.cli_args import args
import torch

if not torch.cuda.is_available():
    args.cpu = True

import contextlib  # noqa: E402
import unittest.mock  # noqa: E402

import comfy.sd  # noqa: E402


_DIFFUSERS_TRIGGER_KEY = "decoder.up_blocks.0.resnets.0.norm1.weight"


def _make_standin():
    """Return a stand-in class whose ``__init__`` is bound to
    ``comfy.sd.VAE.__init__``. Instantiating the returned class
    invokes the production ``__init__`` with the stand-in as ``self``,
    exercising the diffusers-format guard at lines 443-446 without
    standing up a full ``VAE`` (which would require a complete
    safetensors state dict). Mirrors the method-borrowing pattern from
    ``seedvr_model_test.py::_make_standin`` (#109), adapted to a
    constructor rather than discrete instance methods.
    """
    class _StandIn:
        __init__ = comfy.sd.VAE.__init__

    return _StandIn


def test_diffusers_guard_invokes_convert_when_metadata_missing_key():
    """AC1: state dict carries the diffusers-format trigger key and
    ``metadata`` is non-``None`` but does not contain
    ``keep_diffusers_format``. The fixed guard at ``comfy/sd.py:445``
    must enter the conversion branch — ``metadata is None`` is
    ``False``; ``metadata.get("keep_diffusers_format")`` returns
    ``None`` which ``!= "true"`` — and invoke
    ``convert_vae_state_dict`` exactly once. The pre-fix
    ``metadata["keep_diffusers_format"]`` form would raise
    ``KeyError`` on this metadata before the call could happen, so a
    ``call_count`` of 1 is positive evidence that the safe ``.get``
    form is in effect.
    """
    StandIn = _make_standin()
    sd = {"decoder.up_blocks.0.resnets.0.norm1.weight": torch.zeros(1)}
    metadata = {"unrelated_key": "value"}

    with unittest.mock.patch.object(
        comfy.sd.diffusers_convert,
        "convert_vae_state_dict",
        autospec=True,
    ) as mock_convert:
        with contextlib.suppress(Exception):
            StandIn(sd=sd, metadata=metadata)

    assert mock_convert.call_count == 1


def test_diffusers_guard_skips_convert_when_metadata_pins_keep_true():
    """AC2: state dict carries the diffusers-format trigger key and
    ``metadata["keep_diffusers_format"] == "true"``. The guard must
    bypass the conversion — ``convert_vae_state_dict`` must not be
    invoked. This is the explicit opt-out path that lets a caller
    declare an already-Diffusers-formatted VAE should not be rewritten
    by ``convert_vae_state_dict``.
    """
    StandIn = _make_standin()
    sd = {"decoder.up_blocks.0.resnets.0.norm1.weight": torch.zeros(1)}
    metadata = {"keep_diffusers_format": "true"}

    with unittest.mock.patch.object(
        comfy.sd.diffusers_convert,
        "convert_vae_state_dict",
        autospec=True,
    ) as mock_convert:
        with contextlib.suppress(Exception):
            StandIn(sd=sd, metadata=metadata)

    assert mock_convert.call_count == 0
