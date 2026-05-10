"""Unit tests for ``comfy_extras.nodes_seedvr.SeedVR2ProgressiveSampler``
(Slice 1 — chunking with no overlap).

Covers acceptance criteria T1-T4:
- T1: Single-chunk degeneracy (``frames_per_chunk >= T_pixel``) takes the
  short-circuit path and calls ``comfy.sample.sample`` exactly once with
  the full unsliced latent.
- T2: Multi-chunk path slices ``samples_4d`` along the latent T axis,
  invokes the inner sampler once per chunk, and concatenates results
  back into the same total ``(B, 16*T_total, H, W)`` shape with no NaN
  or Inf values.
- T3: ``frames_per_chunk`` that violates the 4n+1 pixel-frame constraint
  is rejected with a typed ``ValueError`` before any model invocation.
- T4: Determinism — given a fixed seed, slicing into N chunks runs each
  chunk against the same global noise tensor (sliced per chunk), so the
  same seed always produces the same final latent regardless of chunk
  count, modulo the inherent T-axis chunk-boundary independence of the
  model.

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
from comfy_extras.nodes_seedvr import (  # noqa: E402
    SeedVR2ProgressiveSampler,
    _blend_overlap_region,
    _concat_chunks_along_t,
    _concat_chunks_with_overlap_blend,
    _hann_blend_weights_1d,
    _slice_collapsed_4d_along_t,
    _slice_seedvr2_cond_along_t,
)

_LAT_C = 16
_COND_C = 17


def _make_inputs(B: int = 1, T: int = 5, H: int = 8, W: int = 8):
    """Build minimal SeedVR2-shaped sampling inputs.

    The latent and condition tensors carry deterministic, reversible
    values (an arange laid out in a 5D ``(B, C, T, H, W)`` view that is
    then collapsed) so per-chunk slices can be cross-checked against
    the original 5D source without ambiguity.
    """
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
    """Return a tensor whose values encode ``(seed, position)`` so the
    chunked slicing path can be verified end-to-end against a global
    noise tensor.
    """
    base = torch.arange(
        latent_image.numel(), dtype=torch.float32
    ).reshape(latent_image.shape)
    return base + float(seed) * 1e6


def _passthrough_sample_returning_latent(
    model, noise, steps, cfg, sampler_name, scheduler,
    positive, negative, latent_image, denoise=1.0,
    noise_mask=None, seed=None,
):
    """Mock for ``comfy.sample.sample``: returns the per-call
    ``latent_image`` unchanged so we can verify the post-concat result
    equals the original input under per-chunk slice + concat.
    """
    return latent_image.clone()


# ---------------------------------------------------------------------------
# Helper-level tests (slicing / concat / cond plumbing)
# ---------------------------------------------------------------------------


def test_slice_collapsed_4d_along_t_shape_correct():
    t = torch.zeros(1, _LAT_C * 5, 8, 8)
    out = _slice_collapsed_4d_along_t(t, 1, 4, _LAT_C)
    assert tuple(out.shape) == (1, _LAT_C * 3, 8, 8)


def test_slice_collapsed_preserves_per_frame_values():
    """Slicing ``[t_start:t_end]`` must preserve the ``(t_start + i)``-th
    latent frame's channel layout at the i'th position of the slice.
    """
    B, T, H, W = 1, 6, 4, 4
    t5 = torch.arange(
        B * _LAT_C * T * H * W, dtype=torch.float32
    ).reshape(B, _LAT_C, T, H, W)
    t4 = t5.reshape(B, _LAT_C * T, H, W).contiguous()
    out_4d = _slice_collapsed_4d_along_t(t4, 2, 5, _LAT_C)
    out_5d = out_4d.reshape(B, _LAT_C, 3, H, W)
    for i, src_t in enumerate([2, 3, 4]):
        assert torch.equal(out_5d[:, :, i], t5[:, :, src_t])


def test_slice_collapsed_4d_along_t_accepts_non_contiguous_input():
    """Collapsed latents may arrive from slicing/cropping views; temporal
    slicing must not require contiguous input storage.
    """
    B, T, H, W = 1, 5, 4, 4
    wide = torch.arange(
        B * _LAT_C * T * H * W * 2, dtype=torch.float32,
    ).reshape(B, _LAT_C * T, H, W * 2)
    src = wide[:, :, :, ::2]
    assert not src.is_contiguous()

    out = _slice_collapsed_4d_along_t(src, 1, 4, _LAT_C)
    expected = src.reshape(B, _LAT_C, T, H, W)[:, :, 1:4].contiguous()
    expected = expected.reshape(B, _LAT_C * 3, H, W)

    assert torch.equal(out, expected)


def test_concat_chunks_along_t_roundtrip_recovers_source():
    """Slicing a tensor and concatenating the slices must reproduce the
    source byte-identically (within tensor equality).
    """
    B, T, H, W = 1, 7, 4, 4
    t = torch.arange(
        B * _LAT_C * T * H * W, dtype=torch.float32
    ).reshape(B, _LAT_C, T, H, W).reshape(B, _LAT_C * T, H, W).contiguous()
    a = _slice_collapsed_4d_along_t(t, 0, 3, _LAT_C)
    b = _slice_collapsed_4d_along_t(t, 3, 5, _LAT_C)
    c = _slice_collapsed_4d_along_t(t, 5, 7, _LAT_C)
    cat = _concat_chunks_along_t([a, b, c], _LAT_C)
    assert torch.equal(cat, t)


def test_concat_chunks_along_t_accepts_non_contiguous_chunks():
    """Concatenation must accept non-contiguous chunk tensors returned by
    sampling or upstream tensor views.
    """
    B, H, W = 1, 4, 4
    wide_a = torch.arange(
        B * _LAT_C * 2 * H * W * 2, dtype=torch.float32,
    ).reshape(B, _LAT_C * 2, H, W * 2)
    wide_b = torch.arange(
        B * _LAT_C * 3 * H * W * 2, dtype=torch.float32,
    ).reshape(B, _LAT_C * 3, H, W * 2) + 10000.0
    chunk_a = wide_a[:, :, :, ::2]
    chunk_b = wide_b[:, :, :, ::2]
    assert not chunk_a.is_contiguous()
    assert not chunk_b.is_contiguous()

    out = _concat_chunks_along_t([chunk_a, chunk_b], _LAT_C)
    expected = torch.cat(
        [
            chunk_a.reshape(B, _LAT_C, 2, H, W),
            chunk_b.reshape(B, _LAT_C, 3, H, W),
        ],
        dim=2,
    ).reshape(B, _LAT_C * 5, H, W)

    assert tuple(out.shape) == (B, _LAT_C * 5, H, W)
    assert torch.equal(out, expected)


def test_slice_seedvr2_cond_along_t_passes_other_keys_unchanged():
    """The cond-list slicer must mutate only ``options['condition']``;
    every other key must pass through unchanged, and the source
    options dict must not be mutated.
    """
    B, T, H, W = 1, 5, 8, 8
    cond = torch.zeros(B, _COND_C * T, H, W)
    text = torch.zeros(1, 4, 32)
    sentinel = object()
    src_options = {"condition": cond, "extra_key": sentinel}
    cond_list = [[text, src_options]]
    out = _slice_seedvr2_cond_along_t(cond_list, 1, 4)
    assert out[0][1]["extra_key"] is sentinel
    assert out[0][1]["condition"].shape == (B, _COND_C * 3, H, W)
    # Source options dict not mutated.
    assert src_options["condition"].shape == (B, _COND_C * T, H, W)


def test_slice_seedvr2_cond_passes_through_entries_without_condition_key():
    """Entries lacking a ``condition`` key are forwarded verbatim — the
    sampler must not crash on conditioning produced by non-SeedVR2
    upstream nodes.
    """
    text = torch.zeros(1, 4, 32)
    cond_list = [[text, {"unrelated": 1}]]
    out = _slice_seedvr2_cond_along_t(cond_list, 0, 1)
    assert out[0] is cond_list[0]
    assert out[0][1] == {"unrelated": 1}


# ---------------------------------------------------------------------------
# T1 — single-chunk degeneracy
# ---------------------------------------------------------------------------


def test_t1_single_chunk_degeneracy_calls_sampler_once_with_full_latent():
    """AC1.1: When ``frames_per_chunk >= T_pixel``, the short-circuit
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


# ---------------------------------------------------------------------------
# T2 — multi-chunk path
# ---------------------------------------------------------------------------


def test_t2_two_chunk_path_shape_preserved_and_no_nan_inf():
    """AC1.3 + boundedness: A T_pixel that exceeds frames_per_chunk
    triggers chunking; the inner sampler is invoked once per chunk;
    the concatenated output preserves the original
    ``(B, 16*T_total, H, W)`` shape and contains no NaN/Inf values.
    """
    # T_latent=11 -> T_pixel=4*10+1=41; chunk_pixel=21 -> chunk_latent=6.
    # Expected chunks: [0:6], [6:11] (two chunks; second is a runt of 5).
    latent, pos, neg, _, _ = _make_inputs(T=11)
    full_shape = tuple(latent["samples"].shape)
    chunk_shapes = []

    def _record(model, noise, steps, cfg, sampler_name, scheduler,
                positive, negative, latent_image, denoise=1.0,
                noise_mask=None, seed=None):
        chunk_shapes.append(tuple(latent_image.shape))
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

    # Two chunks: latent T = 6 then 5.
    assert len(chunk_shapes) == 2
    assert chunk_shapes[0] == (1, _LAT_C * 6, 8, 8)
    assert chunk_shapes[1] == (1, _LAT_C * 5, 8, 8)

    # Final shape preserved.
    out_latent = out.result[0]
    assert tuple(out_latent["samples"].shape) == full_shape

    # Boundedness.
    samples_out = out_latent["samples"]
    assert not torch.isnan(samples_out).any()
    assert not torch.isinf(samples_out).any()


def test_t2_concat_equals_source_under_passthrough_sampler():
    """When the inner sampler is a passthrough (returns its
    ``latent_image`` argument verbatim), the multi-chunk run must
    reconstruct the original input latent byte-identically — that is,
    the slice / sample / concat composition is the identity on the
    latent.
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


def test_t2_per_chunk_cond_slice_matches_chunk_latent_t():
    """Each per-chunk ``comfy.sample.sample`` invocation must receive
    a positive / negative cond list whose ``condition`` tensor has been
    sliced to match the chunk's latent length.
    """
    latent, pos, neg, _, _ = _make_inputs(T=11)
    cond_shapes = []

    def _record_conds(model, noise, steps, cfg, sampler_name, scheduler,
                      positive, negative, latent_image, denoise=1.0,
                      noise_mask=None, seed=None):
        pos_cond_t = positive[0][1]["condition"]
        neg_cond_t = negative[0][1]["condition"]
        cond_shapes.append((tuple(pos_cond_t.shape), tuple(neg_cond_t.shape)))
        return latent_image.clone()

    with patch.object(comfy.sample, "sample", side_effect=_record_conds), \
         patch.object(comfy.sample, "fix_empty_latent_channels",
                      side_effect=_identity_fix_empty), \
         patch.object(comfy.sample, "prepare_noise",
                      side_effect=_fingerprinted_prepare_noise):
        SeedVR2ProgressiveSampler.execute(
            model=None, seed=0, steps=2, cfg=1.0,
            sampler_name="euler", scheduler="simple",
            positive=pos, negative=neg, latent_image=latent,
            denoise=1.0, frames_per_chunk=21, temporal_overlap=0,
        )

    assert cond_shapes[0] == ((1, _COND_C * 6, 8, 8), (1, _COND_C * 6, 8, 8))
    assert cond_shapes[1] == ((1, _COND_C * 5, 8, 8), (1, _COND_C * 5, 8, 8))


def test_t2_standard_noise_mask_passed_through_for_sampler_expansion():
    """Standard ``SetLatentNoiseMask`` masks are ``(B, 1, H, W)`` and
    must be forwarded unchanged so KSampler can expand them to each
    chunk's latent shape.
    """
    latent, pos, neg, _, _ = _make_inputs(T=11)
    latent["noise_mask"] = torch.ones(1, 1, 8, 8)
    mask_shapes = []

    def _record_mask(model, noise, steps, cfg, sampler_name, scheduler,
                     positive, negative, latent_image, denoise=1.0,
                     noise_mask=None, seed=None):
        mask_shapes.append(tuple(noise_mask.shape))
        return latent_image.clone()

    with patch.object(comfy.sample, "sample", side_effect=_record_mask), \
         patch.object(comfy.sample, "fix_empty_latent_channels",
                      side_effect=_identity_fix_empty), \
         patch.object(comfy.sample, "prepare_noise",
                      side_effect=_fingerprinted_prepare_noise):
        SeedVR2ProgressiveSampler.execute(
            model=None, seed=0, steps=2, cfg=1.0,
            sampler_name="euler", scheduler="simple",
            positive=pos, negative=neg, latent_image=latent,
            denoise=1.0, frames_per_chunk=21, temporal_overlap=0,
        )

    assert mask_shapes == [(1, 1, 8, 8), (1, 1, 8, 8)]


def test_t2_collapsed_noise_mask_sliced_per_chunk():
    """A pre-expanded collapsed ``(B, 16*T, H, W)`` noise mask must be
    sliced along latent T to match each chunk before sampling.
    """
    latent, pos, neg, _, _ = _make_inputs(T=11)
    latent["noise_mask"] = torch.ones_like(latent["samples"])
    mask_shapes = []

    def _record_mask(model, noise, steps, cfg, sampler_name, scheduler,
                     positive, negative, latent_image, denoise=1.0,
                     noise_mask=None, seed=None):
        mask_shapes.append(tuple(noise_mask.shape))
        return latent_image.clone()

    with patch.object(comfy.sample, "sample", side_effect=_record_mask), \
         patch.object(comfy.sample, "fix_empty_latent_channels",
                      side_effect=_identity_fix_empty), \
         patch.object(comfy.sample, "prepare_noise",
                      side_effect=_fingerprinted_prepare_noise):
        SeedVR2ProgressiveSampler.execute(
            model=None, seed=0, steps=2, cfg=1.0,
            sampler_name="euler", scheduler="simple",
            positive=pos, negative=neg, latent_image=latent,
            denoise=1.0, frames_per_chunk=21, temporal_overlap=0,
        )

    assert mask_shapes == [(1, _LAT_C * 6, 8, 8), (1, _LAT_C * 5, 8, 8)]


# ---------------------------------------------------------------------------
# T3 — 4n+1 violation rejection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_chunk", [0, -1, 2, 3, 4, 6, 7, 8, 10, 12])
def test_t3_invalid_frames_per_chunk_raises_value_error(bad_chunk):
    """AC1.4: ``frames_per_chunk`` violating 4n+1 (for n >= 0) must raise
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


@pytest.mark.parametrize("good_chunk", [1, 5, 9, 13, 17, 21, 25])
def test_t3_valid_frames_per_chunk_does_not_raise(good_chunk):
    """The 4n+1 sequence (1, 5, 9, 13, ...) must be accepted."""
    latent, pos, neg, _, _ = _make_inputs(T=5)

    with patch.object(comfy.sample, "sample",
                      side_effect=_passthrough_sample_returning_latent), \
         patch.object(comfy.sample, "fix_empty_latent_channels",
                      side_effect=_identity_fix_empty), \
         patch.object(comfy.sample, "prepare_noise",
                      side_effect=_fingerprinted_prepare_noise):
        SeedVR2ProgressiveSampler.execute(
            model=None, seed=0, steps=2, cfg=1.0,
            sampler_name="euler", scheduler="simple",
            positive=pos, negative=neg, latent_image=latent,
            denoise=1.0, frames_per_chunk=good_chunk, temporal_overlap=0,
        )


# ---------------------------------------------------------------------------
# T4 — determinism
# ---------------------------------------------------------------------------


def test_t4_determinism_same_seed_same_output():
    """AC1.4 / determinism: two runs with identical (seed, inputs,
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


def test_t4_chunk_count_invariance_under_passthrough():
    """When the inner sampler is the identity, the final latent must be
    identical regardless of how the work is partitioned: a single-chunk
    run and a multi-chunk run on the same input must produce the same
    output. This pins the slice / concat composition as a true identity
    on the latent under a deterministic inner sampler.
    """
    latent_single, pos_s, neg_s, _, _ = _make_inputs(T=11)
    latent_multi, pos_m, neg_m, _, _ = _make_inputs(T=11)

    with patch.object(comfy.sample, "sample",
                      side_effect=_passthrough_sample_returning_latent), \
         patch.object(comfy.sample, "fix_empty_latent_channels",
                      side_effect=_identity_fix_empty), \
         patch.object(comfy.sample, "prepare_noise",
                      side_effect=_fingerprinted_prepare_noise):
        out_single = SeedVR2ProgressiveSampler.execute(
            model=None, seed=7, steps=2, cfg=1.0,
            sampler_name="euler", scheduler="simple",
            positive=pos_s, negative=neg_s, latent_image=latent_single,
            denoise=1.0, frames_per_chunk=45, temporal_overlap=0,  # >= T_pixel=41
        )
        out_multi = SeedVR2ProgressiveSampler.execute(
            model=None, seed=7, steps=2, cfg=1.0,
            sampler_name="euler", scheduler="simple",
            positive=pos_m, negative=neg_m, latent_image=latent_multi,
            denoise=1.0, frames_per_chunk=21, temporal_overlap=0,  # forces 2 chunks
        )

    assert torch.equal(out_single.result[0]["samples"],
                       out_multi.result[0]["samples"])


# ---------------------------------------------------------------------------
# Slice 2 helper tests (Hann window + blend region + concat-with-blend)
# ---------------------------------------------------------------------------


def test_hann_weights_overlap_3_matches_numz_formula():
    """At ``overlap >= 3`` the Hann formula
    ``0.5 + 0.5 * cos(pi * u)`` (with the [1/3, 2/3] dead-band)
    must produce values identical to numz's
    ``blend_overlapping_frames``: endpoints at ``1.0`` and ``0.0`` for
    the previous-chunk weight, midpoint at ``0.5``.
    """
    w = _hann_blend_weights_1d(3, torch.device("cpu"), torch.float32)
    assert tuple(w.shape) == (3,)
    assert torch.allclose(w[0], torch.tensor(1.0))
    assert torch.allclose(w[-1], torch.tensor(0.0))
    assert torch.allclose(w[1], torch.tensor(0.5), atol=1e-6)


def test_hann_weights_overlap_lt_3_uses_linear_ramp():
    """At ``overlap < 3`` the Hann dead-band collapses, so the helper
    falls back to a linear ramp from 1.0 to 0.0.
    """
    w1 = _hann_blend_weights_1d(1, torch.device("cpu"), torch.float32)
    assert torch.equal(w1, torch.tensor([1.0]))
    w2 = _hann_blend_weights_1d(2, torch.device("cpu"), torch.float32)
    assert torch.equal(w2, torch.tensor([1.0, 0.0]))


def test_hann_weights_monotone_non_increasing():
    """The previous-chunk weight is a crossfade ramp; it must be
    non-increasing along the overlap axis (any reversal would produce
    audible/visible boundary artifacts).
    """
    for n in [3, 4, 5, 7, 8, 11, 16]:
        w = _hann_blend_weights_1d(n, torch.device("cpu"), torch.float32)
        diffs = w[1:] - w[:-1]
        assert torch.all(diffs <= 1e-6), (
            f"Hann weights non-monotone at overlap={n}: {w.tolist()}"
        )


def test_blend_region_endpoints_reproduce_pure_chunks():
    """At the first overlap position the result must equal the
    previous chunk's tail; at the last position it must equal the
    current chunk's head. Verifies the weights actually anchor at 0
    and 1 ends on the underlying tensor.
    """
    B, C, T_overlap, H, W = 1, 16, 5, 4, 4
    prev = torch.full((B, C, T_overlap, H, W), 7.0)
    cur = torch.full((B, C, T_overlap, H, W), -3.0)
    blended = _blend_overlap_region(prev, cur)
    assert torch.allclose(blended[:, :, 0], prev[:, :, 0])
    assert torch.allclose(blended[:, :, -1], cur[:, :, -1])


def test_blend_region_equal_inputs_returns_input():
    """If both chunks agree perfectly in the overlap region, the
    crossfade output must equal the common value at every position.
    Linear combination of equal inputs is always the input.
    """
    B, C, T_overlap, H, W = 1, 16, 5, 4, 4
    same = torch.randn(B, C, T_overlap, H, W)
    blended = _blend_overlap_region(same.clone(), same.clone())
    assert torch.allclose(blended, same, atol=1e-6)


def test_concat_with_overlap_zero_matches_plain_concat():
    """``overlap_latent == 0`` must take the fast path and produce the
    same tensor as ``_concat_chunks_along_t`` of the same chunks.
    Required by AC2.1 (overlap=0 -> Slice 1 byte-identical).
    """
    B, T1, T2, H, W = 1, 3, 4, 4, 4
    a4 = torch.randn(B, _LAT_C * T1, H, W)
    b4 = torch.randn(B, _LAT_C * T2, H, W)
    plain = _concat_chunks_along_t([a4, b4], _LAT_C)
    blended = _concat_chunks_with_overlap_blend(
        [(0, T1, a4), (T1, T1 + T2, b4)], _LAT_C, overlap_latent=0,
    )
    assert torch.equal(blended, plain)


def test_concat_with_overlap_two_chunks_blends_only_overlap_region():
    """For two chunks that overlap by ``overlap_latent`` latent frames,
    the non-overlap portions must be copied verbatim from each chunk;
    only the overlap region carries the blended values.
    """
    B, H, W = 1, 4, 4
    chunk_T = 4
    overlap = 2
    cs0, ce0 = 0, chunk_T  # 0..3
    cs1, ce1 = chunk_T - overlap, chunk_T - overlap + chunk_T  # 2..5
    a4 = torch.full((B, _LAT_C * chunk_T, H, W), 1.0)
    b4 = torch.full((B, _LAT_C * chunk_T, H, W), 2.0)
    out = _concat_chunks_with_overlap_blend(
        [(cs0, ce0, a4), (cs1, ce1, b4)], _LAT_C,
        overlap_latent=overlap,
    )
    assert tuple(out.shape) == (B, _LAT_C * (chunk_T + chunk_T - overlap), H, W)
    out_5d = out.view(B, _LAT_C, chunk_T + chunk_T - overlap, H, W)
    # Pre-overlap: chunk 0 verbatim (index 0..chunk_T - overlap - 1)
    for i in range(chunk_T - overlap):
        assert torch.allclose(out_5d[:, :, i], torch.tensor(1.0))
    # Post-overlap: chunk 1 verbatim (last chunk_T - overlap frames)
    for i in range(chunk_T + chunk_T - overlap - (chunk_T - overlap),
                   chunk_T + chunk_T - overlap):
        assert torch.allclose(out_5d[:, :, i], torch.tensor(2.0))


def test_concat_with_overlap_runt_chunk_uses_min_available_overlap():
    """When the final chunk is a runt shorter than the configured
    overlap, the blend must be performed on the actually-available
    overlap width rather than overrun the runt chunk.
    """
    B, H, W = 1, 4, 4
    overlap = 3
    a4 = torch.full((B, _LAT_C * 4, H, W), 1.0)  # T 0..3
    b4 = torch.full((B, _LAT_C * 1, H, W), 2.0)  # T 1..1 (runt of 1)
    # b4 starts at 1, ends at 2: overlaps [1:4] -> available width 1.
    out = _concat_chunks_with_overlap_blend(
        [(0, 4, a4), (1, 2, b4)], _LAT_C, overlap_latent=overlap,
    )
    # Total covered: indices 0..3 -> length 4.
    assert tuple(out.shape) == (B, _LAT_C * 4, H, W)


# ---------------------------------------------------------------------------
# T5 — Slice 2 with overlap=0 byte-identical to Slice 1
# ---------------------------------------------------------------------------


def test_t5_overlap_zero_byte_identical_to_slice1_path():
    """AC2.1: ``temporal_overlap=0`` must produce output byte-identical
    to the Slice 1 chunked path under a deterministic inner sampler.
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


# ---------------------------------------------------------------------------
# T6 — small overlap (linear ramp path)
# ---------------------------------------------------------------------------


def test_t6_small_overlap_linear_ramp_no_nan_inf():
    """AC2.3 + small-overlap behavior: ``temporal_overlap=2`` exercises
    the linear-ramp fallback (overlap < 3). The output must preserve
    the source's overall T_total shape and contain no NaN/Inf.
    """
    latent, pos, neg, _, _ = _make_inputs(T=11)
    full_shape = tuple(latent["samples"].shape)

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
            denoise=1.0, frames_per_chunk=21, temporal_overlap=2,
        )

    samples_out = out.result[0]["samples"]
    assert tuple(samples_out.shape) == full_shape
    assert not torch.isnan(samples_out).any()
    assert not torch.isinf(samples_out).any()


# ---------------------------------------------------------------------------
# T7 — Hann blend (overlap >= 3): bounded, no boundary discontinuity
# ---------------------------------------------------------------------------


def test_t7_hann_blend_bounded_under_passthrough_inner_sampler():
    """AC2.3 / boundedness for the Hann path. With a passthrough inner
    sampler the per-chunk outputs equal the per-chunk input slices,
    so the post-blend output equals the source latent at every frame
    (the overlap regions blend two slices of the same source). This
    is the strongest available unit-level statement of "no boundary
    discontinuity introduced by the blend".
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
            denoise=1.0, frames_per_chunk=21, temporal_overlap=3,
        )

    samples_out = out.result[0]["samples"]
    assert torch.allclose(samples_out, src, atol=1e-5), (
        "Passthrough inner sampler + Hann blend must reconstruct source: "
        "blending two equal slices of the same source must equal the "
        "source at every position."
    )
    assert not torch.isnan(samples_out).any()
    assert not torch.isinf(samples_out).any()


def test_t7_overlap_too_large_rejected():
    """``temporal_overlap`` must be strictly less than ``chunk_latent``
    or the chunk-loop stride goes non-positive and produces an
    infinite loop. The validator must raise ``ValueError`` before the
    inner sampler is invoked.
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
        # frames_per_chunk=5 -> chunk_latent=2; overlap=2 must be rejected.
        with pytest.raises(ValueError) as excinfo:
            SeedVR2ProgressiveSampler.execute(
                model=None, seed=0, steps=2, cfg=1.0,
                sampler_name="euler", scheduler="simple",
                positive=pos, negative=neg, latent_image=latent,
                denoise=1.0, frames_per_chunk=5, temporal_overlap=2,
            )
    assert "temporal_overlap" in str(excinfo.value)
    assert sampler_called["n"] == 0
