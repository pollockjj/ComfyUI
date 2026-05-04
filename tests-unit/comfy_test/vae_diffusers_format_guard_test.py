"""Regression tests for the VAE diffusers-format guard at comfy/sd.py:443-446.

Tracks pollockjj/mydevelopment#119 (parent #101). The guard previously
indexed ``metadata["keep_diffusers_format"]`` directly, which raised
``KeyError`` when ``metadata`` was non-``None`` but lacked that key —
the common case for any safetensors VAE with ordinary metadata. CodeRabbit
flagged this as Critical on Comfy-Org/ComfyUI#11294 (thread r2959796358).
The merged fix on ``pollockjj/ComfyUI:issue_101`` uses
``metadata.get("keep_diffusers_format") != "true"`` so a missing key
flows through to invoke ``convert_vae_state_dict``, while the
explicit ``"true"`` value skips the conversion.

The guard is inline in ``VAE.__init__`` rather than a separate method,
so the ``_make_standin`` precedent from #109's
``seedvr_model_test.py`` (which borrows ``NaDiT`` methods onto a small
``torch.nn.Module`` subclass) does not apply directly. Instead we drive
the guard by calling ``VAE.__init__`` with a synthetic state dict that
contains only the diffusers-format trigger key
(``decoder.up_blocks.0.resnets.0.norm1.weight``). The trigger key
satisfies the outer ``if`` of the guard but matches no other elif
branch in the post-guard model-detection chain, so ``__init__`` falls
through to the ``else: logging.warning(...); self.first_stage_model =
None; return`` path at ``comfy/sd.py:860-863``. ``convert_vae_state_dict``
is monkey-patched to a tracker rather than allowed to execute on the
synthetic dict, since the production routine assumes a real diffusers
VAE state dict.
"""

from comfy.cli_args import args
import torch

if not torch.cuda.is_available():
    args.cpu = True

import comfy.diffusers_convert  # noqa: E402
from comfy.sd import VAE  # noqa: E402


_DIFFUSERS_TRIGGER_KEY = "decoder.up_blocks.0.resnets.0.norm1.weight"


def _make_diffusers_trigger_sd():
    """Synthetic state dict that satisfies the guard's outer ``if`` (the
    trigger key is present and the SeedVR2 sentinel
    ``decoder.up_blocks.2.upsamplers.0.upscale_conv.weight`` is not) and
    matches no other elif branch in ``VAE.__init__``'s model-detection
    chain, so ``__init__`` exits via the ``else: ...; return`` path.
    The tensor value is unused: ``convert_vae_state_dict`` is
    monkey-patched and the real conversion routine never runs.
    """
    return {_DIFFUSERS_TRIGGER_KEY: torch.zeros(1)}


def test_metadata_missing_keep_diffusers_format_invokes_convert(monkeypatch):
    """AC1: state dict carries the diffusers-format trigger key and
    ``metadata`` is non-``None`` but does not contain
    ``keep_diffusers_format``. The fixed guard must enter the conversion
    branch (``metadata is None`` is False; ``metadata.get(
    "keep_diffusers_format")`` returns ``None`` which ``!= "true"``) and
    invoke ``convert_vae_state_dict`` exactly once. The pre-fix
    ``metadata["keep_diffusers_format"]`` form would raise ``KeyError``
    on this metadata; ``__init__`` running to its early-return
    ``self.first_stage_model = None`` is positive evidence that the
    safe ``.get`` form is in effect.
    """
    calls = []

    def _tracker(sd_arg):
        calls.append(sd_arg)
        return {}

    monkeypatch.setattr(
        comfy.diffusers_convert, "convert_vae_state_dict", _tracker
    )

    sd = _make_diffusers_trigger_sd()
    metadata = {"some_other_key": "irrelevant"}
    assert "keep_diffusers_format" not in metadata, (
        "test precondition: metadata must NOT carry 'keep_diffusers_format'"
    )

    VAE(sd=sd, metadata=metadata)

    assert len(calls) == 1, (
        "convert_vae_state_dict must be invoked exactly once when "
        "metadata is non-None but lacks 'keep_diffusers_format'; "
        f"observed {len(calls)} call(s)"
    )
    assert _DIFFUSERS_TRIGGER_KEY in calls[0], (
        "convert_vae_state_dict must receive the original (pre-conversion) "
        "state dict carrying the diffusers-format trigger key"
    )


def test_metadata_keep_diffusers_format_true_skips_convert(monkeypatch):
    """AC2: state dict carries the diffusers-format trigger key and
    ``metadata["keep_diffusers_format"] == "true"``. The guard must skip
    the conversion call — ``convert_vae_state_dict`` must not be
    invoked. This is the explicit opt-out path that lets a caller
    declare an already-Diffusers-formatted VAE should not be rewritten
    by ``convert_vae_state_dict``.
    """
    calls = []

    def _tracker(sd_arg):
        calls.append(sd_arg)
        return {}

    monkeypatch.setattr(
        comfy.diffusers_convert, "convert_vae_state_dict", _tracker
    )

    sd = _make_diffusers_trigger_sd()
    metadata = {"keep_diffusers_format": "true"}

    VAE(sd=sd, metadata=metadata)

    assert calls == [], (
        "convert_vae_state_dict must NOT be invoked when "
        "metadata['keep_diffusers_format'] == 'true'; "
        f"observed {len(calls)} call(s)"
    )
