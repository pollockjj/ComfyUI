"""Unit tests for ``comfy_extras.nodes_seedvr.SeedVR2ProgressiveSampler``.

Covers:

- Single-chunk degeneracy (``frames_per_chunk >= T_pixel``) takes the
  short-circuit path and calls ``comfy.sample.sample`` exactly once with
  the full unsliced latent.
- Auto chunking walks the 2-, 3-, 4-chunk ladder on OOM.
- ``frames_per_chunk`` that violates the 4n+1 pixel-frame constraint
  is rejected with a typed ``ValueError`` before any model invocation.
- Determinism: given a fixed seed, slicing into N chunks runs each
  chunk against the same global noise tensor (sliced per chunk), so
  the same seed always produces the same final latent.
- Latent-space Hann overlap blend: ``temporal_overlap=0`` produces
  output byte-identical to the no-overlap path.

The tests mock ``comfy.sample.sample``, ``comfy.sample.prepare_noise``,
and ``comfy.sample.fix_empty_latent_channels`` so the slicing /
concatenation / cond-handling logic can be exercised in isolation
without GPU, model weights, or ComfyUI's full sampling stack.
"""

from unittest.mock import patch

import pytest
import torch

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

import comfy.sample  # noqa: E402
import comfy_extras.nodes_seedvr as nodes_seedvr_mod  # noqa: E402
from comfy_extras.nodes_seedvr import SeedVR2ProgressiveSampler  # noqa: E402

_LAT_C = 16
_COND_C = 17


def _make_inputs(B: int = 1, T: int = 5, H: int = 8, W: int = 8):
    """Build minimal SeedVR2-shaped sampling inputs."""
    samples_5d = torch.arange(
        B * _LAT_C * T * H * W, dtype=torch.float32
    ).reshape(B, _LAT_C, T, H, W)
    samples = samples_5d.reshape(B, _LAT_C * T, H, W).contiguous()

    cond_5d = torch.arange(
        B * _COND_C * T * H * W, dtype=torch.float32
    ).reshape(B, _COND_C, T, H, W) + 10000.0
    cond = cond_5d.reshape(B, _COND_C * T, H, W).contiguous()

    text_pos = torch.zeros(1, 4, 32)
    text_neg = torch.zeros(1, 4, 32)
    positive = [[text_pos, {"condition": cond.clone()}]]
    negative = [[text_neg, {"condition": cond.clone()}]]
    latent_image = {"samples": samples}
    return latent_image, positive, negative, samples_5d, cond_5d


def _identity_fix_empty(model, latent_image, downscale_ratio_spacial=None):
    return latent_image


def _fingerprinted_prepare_noise(latent_image, seed, batch_inds=None):
    """Return a tensor whose values encode ``(seed, position)``."""
    base = torch.arange(
        latent_image.numel(), dtype=torch.float32
    ).reshape(latent_image.shape)
    return base + float(seed) * 1e6


def _passthrough_sample_returning_latent(
    model, noise, steps, cfg, sampler_name, scheduler,
    positive, negative, latent_image, denoise=1.0,
    noise_mask=None, seed=None,
):
    """Mock for ``comfy.sample.sample``: returns ``latent_image`` unchanged."""
    return latent_image.clone()


def test_progressive_sampler_schema_exposes_manual_default_auto_chunking():
    schema = SeedVR2ProgressiveSampler.define_schema()
    inputs = {item.id: item for item in schema.inputs}

    assert inputs["chunking_mode"].options == ["manual", "auto"]
    assert inputs["chunking_mode"].default == "manual"


def test_t1_single_chunk_degeneracy_calls_sampler_once_with_full_latent():
    """When ``frames_per_chunk >= T_pixel``, the short-circuit
    standard path runs and calls ``comfy.sample.sample`` exactly once
    with the full unsliced ``(B, 16*T_total, H, W)`` latent.
    """
    latent, pos, neg, _, _ = _make_inputs(T=5)  # T_pixel = 4*4+1 = 17
    full_shape = tuple(latent["samples"].shape)
    calls = []

    def _record(model, noise, steps, cfg, sampler_name, scheduler,
                positive, negative, latent_image, denoise=1.0,
                noise_mask=None, seed=None):
        calls.append(tuple(latent_image.shape))
        return latent_image.clone()

    with patch.object(comfy.sample, "sample", side_effect=_record), \
         patch.object(comfy.sample, "fix_empty_latent_channels",
                      side_effect=_identity_fix_empty), \
         patch.object(comfy.sample, "prepare_noise",
                      side_effect=_fingerprinted_prepare_noise):
        out = SeedVR2ProgressiveSampler.execute(
            model=None, seed=0, steps=2, cfg=1.0,
            sampler_name="euler", scheduler="simple",
            positive=pos, negative=neg, latent_image=latent,
            denoise=1.0, frames_per_chunk=21, temporal_overlap=0,
        )

    assert len(calls) == 1
    assert calls[0] == full_shape
    out_latent = out.result[0]
    assert tuple(out_latent["samples"].shape) == full_shape


def test_auto_chunking_walks_two_three_four_chunk_ladder():
    """Auto mode must walk 2-, 3-, then 4-chunk geometries on OOM."""
    latent, pos, neg, _, _ = _make_inputs(T=17)
    calls = []

    def _oom_until_four_chunks(model, noise, steps, cfg, sampler_name,
                               scheduler, positive, negative,
                               latent_image, denoise=1.0,
                               noise_mask=None, seed=None):
        calls.append(tuple(latent_image.shape))
        if latent_image.shape[1] > _LAT_C * 5:
            raise torch.cuda.OutOfMemoryError("chunk too large")
        return latent_image.clone()

    with patch.object(comfy.sample, "sample",
                      side_effect=_oom_until_four_chunks), \
         patch.object(comfy.sample, "fix_empty_latent_channels",
                      side_effect=_identity_fix_empty), \
         patch.object(comfy.sample, "prepare_noise",
                      side_effect=_fingerprinted_prepare_noise), \
         patch.object(nodes_seedvr_mod.comfy.model_management,
                      "soft_empty_cache") as soft_empty:
        out = SeedVR2ProgressiveSampler.execute(
            model=None, seed=0, steps=2, cfg=1.0,
            sampler_name="euler", scheduler="simple",
            positive=pos, negative=neg, latent_image=latent,
            denoise=1.0, frames_per_chunk=65, temporal_overlap=0,
            chunking_mode="auto",
        )

    assert calls[:4] == [
        (1, _LAT_C * 17, 8, 8),
        (1, _LAT_C * 9, 8, 8),
        (1, _LAT_C * 6, 8, 8),
        (1, _LAT_C * 5, 8, 8),
    ]
    assert torch.equal(out.result[0]["samples"], latent["samples"])
    assert soft_empty.call_count == 3


@pytest.mark.parametrize("bad_chunk", [0, -1, 2, 3, 4, 6, 7, 8, 10, 12])
def test_t3_invalid_frames_per_chunk_raises_value_error(bad_chunk):
    """``frames_per_chunk`` violating 4n+1 (for n >= 0) must raise
    ``ValueError`` with a message naming the offending value, before any
    model invocation. ``frames_per_chunk < 1`` is also rejected.
    """
    latent, pos, neg, _, _ = _make_inputs(T=5)

    sampler_called = {"n": 0}

    def _should_not_be_called(*args, **kwargs):
        sampler_called["n"] += 1
        return torch.zeros(1)

    with patch.object(comfy.sample, "sample",
                      side_effect=_should_not_be_called), \
         patch.object(comfy.sample, "fix_empty_latent_channels",
                      side_effect=_identity_fix_empty), \
         patch.object(comfy.sample, "prepare_noise",
                      side_effect=_fingerprinted_prepare_noise):
        with pytest.raises(ValueError) as excinfo:
            SeedVR2ProgressiveSampler.execute(
                model=None, seed=0, steps=2, cfg=1.0,
                sampler_name="euler", scheduler="simple",
                positive=pos, negative=neg, latent_image=latent,
                denoise=1.0, frames_per_chunk=bad_chunk, temporal_overlap=0,
            )
    assert str(bad_chunk) in str(excinfo.value)
    assert sampler_called["n"] == 0


def test_t4_determinism_same_seed_same_output():
    """Two runs with identical (seed, inputs,
    frames_per_chunk) must produce byte-identical output, given the
    inner sampler is deterministic (here: passthrough).
    """
    latent_a, pos_a, neg_a, _, _ = _make_inputs(T=11)
    latent_b, pos_b, neg_b, _, _ = _make_inputs(T=11)

    with patch.object(comfy.sample, "sample",
                      side_effect=_passthrough_sample_returning_latent), \
         patch.object(comfy.sample, "fix_empty_latent_channels",
                      side_effect=_identity_fix_empty), \
         patch.object(comfy.sample, "prepare_noise",
                      side_effect=_fingerprinted_prepare_noise):
        out_a = SeedVR2ProgressiveSampler.execute(
            model=None, seed=42, steps=2, cfg=1.0,
            sampler_name="euler", scheduler="simple",
            positive=pos_a, negative=neg_a, latent_image=latent_a,
            denoise=1.0, frames_per_chunk=21, temporal_overlap=0,
        )
        out_b = SeedVR2ProgressiveSampler.execute(
            model=None, seed=42, steps=2, cfg=1.0,
            sampler_name="euler", scheduler="simple",
            positive=pos_b, negative=neg_b, latent_image=latent_b,
            denoise=1.0, frames_per_chunk=21, temporal_overlap=0,
        )

    assert torch.equal(out_a.result[0]["samples"],
                       out_b.result[0]["samples"])


def test_t5_overlap_zero_byte_identical_to_slice1_path():
    """``temporal_overlap=0`` must produce output byte-identical
    to the no-overlap chunked path under a deterministic inner sampler.
    Verifies the overlap=0 fast path is wired correctly through
    ``_concat_chunks_with_overlap_blend``.
    """
    latent, pos, neg, _, _ = _make_inputs(T=11)
    src = latent["samples"].clone()

    with patch.object(comfy.sample, "sample",
                      side_effect=_passthrough_sample_returning_latent), \
         patch.object(comfy.sample, "fix_empty_latent_channels",
                      side_effect=_identity_fix_empty), \
         patch.object(comfy.sample, "prepare_noise",
                      side_effect=_fingerprinted_prepare_noise):
        out = SeedVR2ProgressiveSampler.execute(
            model=None, seed=0, steps=2, cfg=1.0,
            sampler_name="euler", scheduler="simple",
            positive=pos, negative=neg, latent_image=latent,
            denoise=1.0, frames_per_chunk=21, temporal_overlap=0,
        )

    out_latent = out.result[0]
    assert torch.equal(out_latent["samples"], src)
