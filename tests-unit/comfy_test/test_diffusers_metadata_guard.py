"""Regression tests for the diffusers-format guard inside
``comfy.sd.VAE.__init__``.

Tracks pollockjj/mydevelopment#119 (parent #101). The guard previously
indexed ``metadata["keep_diffusers_format"]`` directly, raising
``KeyError`` when ``metadata`` was non-``None`` but lacked that key —
the common case for any safetensors VAE with ordinary metadata.
CodeRabbit flagged this as Critical on Comfy-Org/ComfyUI#11294
(thread r2959796358). The merged fix on ``pollockjj/ComfyUI:issue_101``
uses ``metadata.get("keep_diffusers_format") != "true"``: a missing
key flows through to invoke ``convert_vae_state_dict`` (the rewrite),
while the explicit ``"true"`` value bypasses it.

Semantically, the guard sits inline in the constructor and gates the
``diffusers_convert.convert_vae_state_dict(sd)`` rewrite for diffusers-
format VAE state dicts. The first call after the guard is
``model_management.is_amd()`` — the test cells halt the constructor at
that point so the regression contract does not depend on any code
beyond the guarded branch. (Source line numbers are deliberately not
referenced here so unrelated edits to ``comfy/sd.py`` do not silently
drift the documentation.)

The two test cells below are the binary AC contract that the qa-slice
gate evaluates verbatim. Both share an ``_exercise_guard`` helper so the
patched-constructor harness stays single-sourced; each cell supplies its
own ``metadata`` literal and asserts on the call counts. The helper
contract:

  * ``unittest.mock.patch.object(comfy.sd.diffusers_convert,
    "convert_vae_state_dict", autospec=True, side_effect=...)`` —
    patches the symbol the guarded conversion call resolves through
    (``from . import diffusers_convert``); ``autospec=True`` pins the
    mock's signature to the production function so a signature drift
    fails loudly. The ``side_effect`` is a passthrough lambda that
    returns its input ``sd`` unchanged, so the production-line
    ``sd = diffusers_convert.convert_vae_state_dict(sd)`` rebinds
    ``sd`` to the same real dict the cell constructed — not to the
    default ``MagicMock`` return value, which would make any
    post-guard ``in sd`` checks meaningless and make the cell brittle
    to unrelated constructor changes.
  * ``unittest.mock.patch.object(comfy.sd.model_management, "is_amd",
    autospec=True, side_effect=_PostGuardReached(...))`` — patches the
    first call after the guard; raising a dedicated sentinel exception
    from this mock halts ``__init__`` immediately past the guard, so the
    cell never depends on whatever comes after that call site in the
    constructor (model-detection chain, ``first_stage_model``
    instantiation, dtype routing, etc.). Future post-guard refactors can
    change behaviour freely without forcing this regression to grow
    false positives or false negatives.
  * ``_make_standin()`` — borrows ``comfy.sd.VAE.__init__`` onto a bare
    class, mirroring the precedent set in
    ``tests-unit/comfy_test/seedvr_model_test.py::_make_standin``
    (#109). The guard sits inline in the constructor above any
    subsystem instantiation, so binding ``__init__`` is sufficient to
    drive it.
  * ``contextlib.suppress(_PostGuardReached)`` — catches ONLY the
    deliberate halt-point exception raised by the patched ``is_amd``.
    Any exception of a different class (a ``KeyError`` from a
    regression of the original metadata-indexing bug, an
    ``AttributeError`` from an unexpected ``sd`` shape, anything else)
    propagates uncaught and fails the cell loudly. This replaces an
    earlier ``suppress(Exception)`` that masked any post-guard failure
    along with the deliberate halt; per Copilot review feedback on PR
    pollockjj/ComfyUI#36 (comments r3184935719, r3184935740,
    r3184935750), narrowing the suppression class restores the
    regression signal that a broad catch had weakened.
  * Paired assertions per cell — ``mock_convert.call_count`` is the
    branch witness (AC1 expects 1, AC2 expects 0); ``mock_is_amd.called``
    is the post-guard reach witness (both ACs expect ``True``). Both
    must hold for the cell to pass.

The trigger key ``decoder.up_blocks.0.resnets.0.norm1.weight`` is
factored into ``_DIFFUSERS_TRIGGER_KEY`` and referenced from the helper.
It is still hard-coded (not derived from ``comfy/sd.py`` at runtime),
so an upstream rename remains a loud-fail signal per the issue's
stop-condition; centralizing the literal collapses three update sites
to one.
"""

from comfy.cli_args import args
import torch

if not torch.cuda.is_available():
    args.cpu = True

import contextlib  # noqa: E402
import unittest.mock  # noqa: E402

import comfy.sd  # noqa: E402


_DIFFUSERS_TRIGGER_KEY = "decoder.up_blocks.0.resnets.0.norm1.weight"


class _PostGuardReached(Exception):
    """Sentinel raised by the patched ``model_management.is_amd`` to halt
    ``VAE.__init__`` at the first statement past the diffusers-format
    guard. Catching this dedicated class — and only this class — gives
    positive proof that control crossed the guard while leaving any
    other exception (a ``KeyError`` regression of the original indexing
    bug, or any unrelated raise) free to propagate and fail the cell.
    """


def _make_standin():
    """Return a stand-in class whose ``__init__`` is bound to
    ``comfy.sd.VAE.__init__``. Instantiating the returned class invokes
    the production ``__init__`` with the stand-in as ``self``,
    exercising the diffusers-format guard without standing up a full
    ``VAE`` (which would require a complete safetensors state dict).
    Mirrors the method-borrowing pattern from
    ``seedvr_model_test.py::_make_standin`` (#109), adapted to a
    constructor rather than discrete instance methods.
    """
    class _StandIn:
        __init__ = comfy.sd.VAE.__init__

    return _StandIn


def _exercise_guard(metadata):
    """Drive ``comfy.sd.VAE.__init__`` against a stand-in with the
    diffusers-format trigger state dict and the supplied ``metadata``,
    halting at the first call after the guard. Returns the
    ``(mock_convert, mock_is_amd)`` pair so each cell asserts its own
    branch witness (``mock_convert.call_count``) and post-guard reach
    witness (``mock_is_amd.called``).

    Single-sourcing this harness keeps AC1 and AC2 synchronized. The
    halt point and mock wiring are tightly coupled to ``VAE.__init__``;
    duplicating them per cell invited drift between the two ACs (Copilot
    review feedback on PR pollockjj/ComfyUI#36, comment r3184983231).
    The metadata dict is the only per-cell variant — that is where the
    guarded branches diverge.
    """
    StandIn = _make_standin()
    sd = {_DIFFUSERS_TRIGGER_KEY: torch.zeros(1)}

    with unittest.mock.patch.object(
        comfy.sd.diffusers_convert,
        "convert_vae_state_dict",
        autospec=True,
        side_effect=lambda state_dict: state_dict,
    ) as mock_convert, unittest.mock.patch.object(
        comfy.sd.model_management,
        "is_amd",
        autospec=True,
        side_effect=_PostGuardReached(
            "VAE.__init__ reached the first statement past the diffusers-format guard"
        ),
    ) as mock_is_amd:
        with contextlib.suppress(_PostGuardReached):
            StandIn(sd=sd, metadata=metadata)

    return mock_convert, mock_is_amd


def test_diffusers_guard_invokes_convert_when_metadata_missing_key():
    """AC1: state dict carries the diffusers-format trigger key and
    ``metadata`` is non-``None`` but does not contain
    ``keep_diffusers_format``. The fixed guard must enter the conversion
    branch — ``metadata is None`` is ``False``;
    ``metadata.get("keep_diffusers_format")`` returns ``None`` which
    ``!= "true"`` — and invoke ``convert_vae_state_dict`` exactly once.
    The pre-fix ``metadata["keep_diffusers_format"]`` form would raise
    ``KeyError`` on this metadata before the call could happen, so a
    ``call_count`` of 1 is positive evidence that the safe ``.get`` form
    is in effect. The patched converter passes ``sd`` through unchanged
    so the post-guard logic runs against the same real state dict the
    cell constructed; the patched ``is_amd`` then raises
    ``_PostGuardReached`` to halt the constructor at exactly that point.
    ``mock_is_amd.called`` is the paired witness that control reached
    the first post-guard statement, and the narrow
    ``suppress(_PostGuardReached)`` ensures any other regression (e.g.,
    a ``KeyError`` reintroduced into the guard) escapes the cell instead
    of being silently masked.
    """
    mock_convert, mock_is_amd = _exercise_guard({"unrelated_key": "value"})

    assert mock_convert.call_count == 1
    assert mock_is_amd.called


def test_diffusers_guard_skips_convert_when_metadata_pins_keep_true():
    """AC2: state dict carries the diffusers-format trigger key and
    ``metadata["keep_diffusers_format"] == "true"``. The guard must
    bypass the conversion — ``convert_vae_state_dict`` must not be
    invoked. This is the explicit opt-out path that lets a caller
    declare an already-Diffusers-formatted VAE should not be rewritten
    by ``convert_vae_state_dict``. The patched ``is_amd`` raises
    ``_PostGuardReached`` to halt the constructor at the first
    post-guard statement so the cell does not depend on any code past
    that point; ``mock_is_amd.called`` is the paired witness that
    control reached that statement on the opt-out path, and the narrow
    ``suppress(_PostGuardReached)`` ensures any regression that raised
    before reaching ``is_amd`` (a ``KeyError`` resurrection, for
    instance) escapes the cell rather than being silently masked.
    """
    mock_convert, mock_is_amd = _exercise_guard({"keep_diffusers_format": "true"})

    assert mock_convert.call_count == 0
    assert mock_is_amd.called
