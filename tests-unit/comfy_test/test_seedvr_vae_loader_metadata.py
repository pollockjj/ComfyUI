"""Regression test for issue #191 — ``comfy/sd.py``'s ``VAE.__init__`` loader
must apply SeedVR2-specific metadata when the SeedVR2 magic key
``decoder.up_blocks.2.upsamplers.0.upscale_conv.weight`` is present in the
state dict.

Pre-fix behaviour (before commit ``3cbb5dd89b`` on ``pollockjj/ComfyUI``):
the SeedVR2 elif branch at ``comfy/sd.py:518`` either did not exist or did
not set ``self.latent_channels = 16`` / ``self.latent_dim = 3`` /
``self.disable_offload = True`` /
``self.downscale_index_formula = (4, 8, 8)`` /
``self.upscale_index_formula = (4, 8, 8)``, leaving the loader at the
defaults of ``latent_channels=4`` / ``latent_dim=2`` written at lines
457-458. Down-stream consumers therefore mis-shape the latent buffer and
crash with a channel-count mismatch (Comfy-Org/ComfyUI#11294 issuecomment
``3668638161``).

Post-fix behaviour (``pollockjj/ComfyUI:issue_101`` HEAD): the elif at
``comfy/sd.py:518-531`` sets ``latent_channels=16``, ``latent_dim=3``,
``disable_offload=True``, ``downscale_index_formula=(4, 8, 8)``,
``upscale_index_formula=(4, 8, 8)``, and also installs the SeedVR2
``memory_used_decode`` / ``memory_used_encode`` lambdas, the
``downscale_ratio`` / ``upscale_ratio`` tuples, and the no-op
``process_input`` / ``crop_input=False`` overrides.

This module exercises the real ``VAE.__init__`` detection-and-load path
with a stubbed state dict containing only the SeedVR2 magic key, and
patches ``comfy.ldm.seedvr.vae.VideoAutoencoderKLWrapper`` with a tiny
``nn.Module`` subclass so the test stays CPU-only and weight-load-free
while still satisfying ``isinstance(...)`` against the real wrapper class
(see ``_StubVideoAutoencoderKLWrapper`` below).

Test design rationale and per-decision review trail are recorded on the
tracking issue: https://github.com/pollockjj/mydevelopment/issues/191
"""

from unittest.mock import patch

import pytest
import torch

# CPU-only CI fix: ``comfy.sd`` transitively imports
# ``comfy.model_management``, whose import-time
# ``cpu_state = CPUState.CPU if args.cpu`` initialiser reads
# ``comfy.cli_args.args.cpu``. Match the pattern at
# ``tests-unit/comfy_test/test_seedvr_vae_decode_unpadded_t.py:33-44``: flip
# ``args.cpu`` BEFORE importing any ``comfy.sd`` / ``comfy.ldm.*`` symbol
# when CUDA is unavailable. Issue-191 AC-3 additionally requires the
# ``_cli_args.cpu = True`` assignment line number to precede every line
# matching ``^import comfy`` or ``^from comfy`` in the committed file, so
# the cli_args module is loaded via ``importlib`` here rather than via
# ``from comfy.cli_args import args``.
import importlib

_cli_args = importlib.import_module("comfy.cli_args").args

if not torch.cuda.is_available():
    _cli_args.cpu = True

import torch.nn as nn  # noqa: E402

import comfy.ldm.seedvr.vae as seedvr_vae  # noqa: E402
import comfy.sd  # noqa: E402


_SEEDVR2_MAGIC_KEY = "decoder.up_blocks.2.upsamplers.0.upscale_conv.weight"


class _StubVideoAutoencoderKLWrapper(seedvr_vae.VideoAutoencoderKLWrapper):
    """Subclass that bypasses the real wrapper's heavy weight construction.

    The downstream ``comfy.sd.VAE.__init__`` lifecycle after line 519 only
    relies on ``nn.Module`` machinery — ``.eval()``, ``.to(dtype)``,
    ``state_dict()`` for ``module_size``, and
    ``load_state_dict(strict=False)``. A bare ``nn.Module.__init__`` provides
    all of that. Subclassing ``VideoAutoencoderKLWrapper`` keeps
    ``isinstance(stub_instance, VideoAutoencoderKLWrapper)`` ``True`` after
    the patch context exits, so the AC-A isinstance assertion holds against
    the real wrapper class.
    """

    def __init__(self):
        nn.Module.__init__(self)


def _build_seedvr2_stub_sd():
    """Minimum state dict that triggers the SeedVR2 elif branch in
    ``comfy/sd.py``. The detection is a pure ``in sd`` containment check
    against the magic key at line 518; no other key is required to reach
    that branch (the diffusers-convert early-out at lines 444-446 is
    short-circuited by the ``is_seedvr2_vae`` flag set at line 443).

    The ``load_state_dict`` call at line 884 uses ``strict=False`` so the
    single magic key is accepted as ``unexpected`` against the empty stub
    module without raising.
    """
    return {_SEEDVR2_MAGIC_KEY: torch.zeros(1)}


@pytest.fixture(scope="module")
def seedvr2_vae():
    """Build a real ``comfy.sd.VAE`` instance through the detection-and-load
    path with the SeedVR2 wrapper class stubbed for CPU-only execution.
    """
    sd = _build_seedvr2_stub_sd()
    with patch.object(
        seedvr_vae,
        "VideoAutoencoderKLWrapper",
        _StubVideoAutoencoderKLWrapper,
    ):
        vae = comfy.sd.VAE(sd=sd)
    return vae


def test_seedvr2_loader_first_stage_model_is_video_autoencoder_kl_wrapper(
    seedvr2_vae,
):
    assert isinstance(
        seedvr2_vae.first_stage_model, seedvr_vae.VideoAutoencoderKLWrapper
    ) is True, (
        "Expected first_stage_model to be a VideoAutoencoderKLWrapper "
        f"instance; got {type(seedvr2_vae.first_stage_model).__name__}. The "
        "SeedVR2 elif branch at comfy/sd.py:518 may not have been taken."
    )


def test_seedvr2_loader_sets_latent_channels_16(seedvr2_vae):
    assert seedvr2_vae.latent_channels == 16, (
        "Expected latent_channels=16 (set at comfy/sd.py:520 inside the "
        f"SeedVR2 elif branch); got {seedvr2_vae.latent_channels}. SeedVR2's "
        "VideoAutoencoderKL uses 16-channel latents per Wang et al., ICLR "
        "2026 (arXiv 2506.05301) §3; the loader default of 4 (comfy/sd.py:457)"
        " is wrong for the SeedVR2 path."
    )


def test_seedvr2_loader_sets_latent_dim_3(seedvr2_vae):
    assert seedvr2_vae.latent_dim == 3, (
        "Expected latent_dim=3 (set at comfy/sd.py:521 inside the SeedVR2 "
        f"elif branch); got {seedvr2_vae.latent_dim}. SeedVR2 latents are 3D "
        "(T, H, W) per the upstream ByteDance-Seed/SeedVR "
        "VideoAutoencoderKL contract; the loader default of 2 "
        "(comfy/sd.py:458) is wrong for the SeedVR2 path."
    )


def test_seedvr2_loader_sets_downscale_index_formula(seedvr2_vae):
    assert seedvr2_vae.downscale_index_formula == (4, 8, 8), (
        "Expected downscale_index_formula=(4, 8, 8) (set at "
        f"comfy/sd.py:527); got {seedvr2_vae.downscale_index_formula}. "
        "SeedVR2's spatial-temporal downscale ratio is 4× temporal × 8× "
        "spatial × 8× spatial."
    )


def test_seedvr2_loader_sets_upscale_index_formula(seedvr2_vae):
    assert seedvr2_vae.upscale_index_formula == (4, 8, 8), (
        "Expected upscale_index_formula=(4, 8, 8) (set at "
        f"comfy/sd.py:529); got {seedvr2_vae.upscale_index_formula}. "
        "SeedVR2's spatial-temporal upscale ratio is the inverse of its "
        "downscale ratio: 4× temporal × 8× spatial × 8× spatial."
    )


def test_seedvr2_loader_sets_disable_offload(seedvr2_vae):
    assert seedvr2_vae.disable_offload is True, (
        "Expected disable_offload=True (set at comfy/sd.py:522); got "
        f"{seedvr2_vae.disable_offload}. SeedVR2 cannot tolerate CPU "
        "offload during decode (the wrapper retains memory-state references "
        "across slice boundaries — see VideoAutoencoderKL.slicing_decode)."
    )


def test_seedvr2_loader_sets_working_dtypes(seedvr2_vae):
    assert seedvr2_vae.working_dtypes == [torch.bfloat16, torch.float32], (
        "Expected working_dtypes=[torch.bfloat16, torch.float32] (set at "
        f"comfy/sd.py:525); got {seedvr2_vae.working_dtypes}. SeedVR2's "
        "weight cast contract excludes float16 — leaking float16 here would "
        "drop the working-dtype probe and route the SeedVR2 path through a "
        "fallback dtype the published checkpoints were not trained for."
    )


def test_seedvr2_loader_sets_downscale_ratio(seedvr2_vae):
    """Lock the SeedVR2 downscale_ratio shape AND the temporal-axis lambda
    against published 4× temporal stride (Wang et al., ICLR 2026 §3). A
    regression that flips the lambda to a non-SeedVR2 form (e.g. SD3's
    ``a // 8`` or VAE-default identity) would break the tiled 3D path's
    latent-frame-count derivation in ``comfy/sd.py``'s ``encode_tiled``.
    """
    ratio = seedvr2_vae.downscale_ratio
    assert isinstance(ratio, tuple) and len(ratio) == 3, (
        "Expected downscale_ratio to be a 3-tuple (set at comfy/sd.py:526); "
        f"got {type(ratio).__name__} of length "
        f"{len(ratio) if hasattr(ratio, '__len__') else 'N/A'}."
    )
    assert ratio[1] == 8 and ratio[2] == 8, (
        "Expected downscale_ratio spatial axes (idx 1, 2) to both equal 8 "
        f"(SeedVR2 spatial 8× downsample); got ({ratio[1]}, {ratio[2]})."
    )
    fn = ratio[0]
    assert callable(fn), (
        f"Expected downscale_ratio[0] to be callable; got {type(fn).__name__}."
    )
    # 4× temporal stride: floor((a + 3) / 4), clamped to >= 0.
    # Anchor against the standard SeedVR2 latent shapes.
    cases = {1: 1, 2: 1, 3: 1, 4: 1, 5: 2, 8: 2, 9: 3, 13: 4, 33: 9}
    for a_in, t_expected in cases.items():
        actual = fn(a_in)
        assert actual == t_expected, (
            f"downscale_ratio temporal lambda: fn({a_in}) returned "
            f"{actual}; expected {t_expected} per "
            f"max(0, floor(({a_in} + 3) / 4)) (comfy/sd.py:526)."
        )


def test_seedvr2_loader_sets_upscale_ratio(seedvr2_vae):
    """Lock the SeedVR2 upscale_ratio shape AND the temporal-axis lambda —
    inverse of the downscale ratio (``a * 4 - 3``, clamped >= 0). Regression
    on this lambda would yield wrong output frame counts in
    ``VideoAutoencoderKLWrapper.decode``'s ``T_out`` reconstruction.
    """
    ratio = seedvr2_vae.upscale_ratio
    assert isinstance(ratio, tuple) and len(ratio) == 3, (
        "Expected upscale_ratio to be a 3-tuple (set at comfy/sd.py:528); "
        f"got {type(ratio).__name__} of length "
        f"{len(ratio) if hasattr(ratio, '__len__') else 'N/A'}."
    )
    assert ratio[1] == 8 and ratio[2] == 8, (
        "Expected upscale_ratio spatial axes (idx 1, 2) to both equal 8 "
        f"(SeedVR2 spatial 8× upscale); got ({ratio[1]}, {ratio[2]})."
    )
    fn = ratio[0]
    assert callable(fn), (
        f"Expected upscale_ratio[0] to be callable; got {type(fn).__name__}."
    )
    cases = {1: 1, 2: 5, 3: 9, 5: 17, 9: 33}
    for a_in, t_expected in cases.items():
        actual = fn(a_in)
        assert actual == t_expected, (
            f"upscale_ratio temporal lambda: fn({a_in}) returned "
            f"{actual}; expected {t_expected} per "
            f"max(0, {a_in} * 4 - 3) (comfy/sd.py:528)."
        )


def test_seedvr2_loader_sets_memory_used_decode(seedvr2_vae):
    """Lock the SeedVR2 ``memory_used_decode`` formula:
    ``shape[1] * shape[-2] * shape[-1] * (4 * 8 * 8) * dtype_size(dtype)``
    (comfy/sd.py:523). Wrong formula would mis-size decode memory budget
    and trigger spurious tile-fallbacks or OOM on legitimate inputs.
    """
    fn = seedvr2_vae.memory_used_decode
    assert callable(fn), (
        f"Expected memory_used_decode to be callable; got {type(fn).__name__}."
    )
    # shape = (B, C, T, H, W); only shape[1] (C), shape[-2] (H), shape[-1] (W)
    # and dtype-size are consumed. dtype_size(bfloat16) == 2 in
    # comfy.model_management.
    actual = fn((1, 16, 4, 32, 32), torch.bfloat16)
    expected = 16 * 32 * 32 * (4 * 8 * 8) * 2
    assert actual == expected, (
        f"memory_used_decode((1,16,4,32,32), bfloat16) returned {actual}; "
        f"expected {expected} per shape[1]*shape[-2]*shape[-1]*(4*8*8)*2 "
        f"(comfy/sd.py:523)."
    )
    # Same shape, fp32 (dtype_size=4) doubles the answer.
    actual_fp32 = fn((1, 16, 4, 32, 32), torch.float32)
    expected_fp32 = 16 * 32 * 32 * (4 * 8 * 8) * 4
    assert actual_fp32 == expected_fp32, (
        f"memory_used_decode fp32 path returned {actual_fp32}; expected "
        f"{expected_fp32}."
    )


def test_seedvr2_loader_sets_memory_used_encode(seedvr2_vae):
    """Lock the SeedVR2 ``memory_used_encode`` formula:
    ``max(shape[2], 5) * shape[3] * shape[4] * 64 * dtype_size(dtype)``
    (comfy/sd.py:524). The ``max(shape[2], 5)`` floor matches SeedVR2's
    minimum 5-frame temporal window after ``cut_videos`` padding (see
    ``cut_videos`` in ``comfy_extras/nodes_seedvr.py``); regression to a
    plain ``shape[2]`` would mis-size encode memory budget on short clips.
    """
    fn = seedvr2_vae.memory_used_encode
    assert callable(fn), (
        f"Expected memory_used_encode to be callable; got {type(fn).__name__}."
    )
    # shape (B, C, T, H, W). For T=2 the max-floor must promote to 5.
    actual_short = fn((1, 3, 2, 32, 32), torch.bfloat16)
    expected_short = 5 * 32 * 32 * 64 * 2
    assert actual_short == expected_short, (
        f"memory_used_encode T=2 path returned {actual_short}; expected "
        f"{expected_short} per max(2, 5)*32*32*64*2 (comfy/sd.py:524). The "
        f"max(_, 5) floor matches SeedVR2's cut_videos minimum window."
    )
    # T=8 stays at T (max(8, 5) == 8).
    actual_long = fn((1, 3, 8, 32, 32), torch.bfloat16)
    expected_long = 8 * 32 * 32 * 64 * 2
    assert actual_long == expected_long, (
        f"memory_used_encode T=8 path returned {actual_long}; expected "
        f"{expected_long} per max(8, 5)*32*32*64*2."
    )


def test_seedvr2_loader_sets_process_input_identity(seedvr2_vae):
    """SeedVR2's ``process_input`` (comfy/sd.py:530) is the identity lambda;
    other VAE branches normalise to [-1, 1] or apply diffusers preprocessing.
    A regression that re-enables the default normaliser would double-shift
    the SeedVR2 input distribution — inputs are already pre-normalised by
    ``SeedVR2InputProcessing.execute`` in comfy_extras/nodes_seedvr.py.
    """
    fn = seedvr2_vae.process_input
    assert callable(fn), (
        f"Expected process_input to be callable; got {type(fn).__name__}."
    )
    img = torch.randn(2, 3, 4, 4)
    out = fn(img)
    assert out is img, (
        "process_input(img) returned a NEW object; expected the same tensor "
        "(identity lambda). A regression here would reintroduce normalisation "
        "on already-normalised SeedVR2 inputs, double-shifting the distribution."
    )


def test_seedvr2_loader_sets_crop_input_false(seedvr2_vae):
    assert seedvr2_vae.crop_input is False, (
        "Expected crop_input=False (set at comfy/sd.py:531); got "
        f"{seedvr2_vae.crop_input}. SeedVR2 inputs are already aspect-ratio "
        "managed in SeedVR2InputProcessing.execute (side_resize + div_pad); "
        "re-enabling crop_input would double-crop and shrink the visible area."
    )
