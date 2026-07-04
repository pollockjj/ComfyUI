"""SeedVR2 temporal chunk/merge node regression tests."""

import pytest
import torch

from comfy.cli_args import args as cli_args
from comfy.ldm.seedvr.constants import SEEDVR2_LATENT_CHANNELS

if not torch.cuda.is_available():
    cli_args.cpu = True

import comfy.model_management  # noqa: E402
from comfy_extras.nodes_seedvr import (  # noqa: E402
    SeedVR2TemporalChunk,
    SeedVR2TemporalMerge,
    _seedvr2_chunk_crossfade_weights,
)


def _latent(t_latent, b=1, h=8, w=8):
    g = torch.Generator().manual_seed(7)
    return {"samples": torch.randn(b, SEEDVR2_LATENT_CHANNELS, t_latent, h, w, generator=g)}


def _split(latent, frames_per_chunk, temporal_overlap, chunking_mode="manual"):
    return SeedVR2TemporalChunk.execute(
        latent, frames_per_chunk, temporal_overlap, chunking_mode).args


def _merge(chunks, temporal_overlap):
    return SeedVR2TemporalMerge.execute(chunks, [temporal_overlap]).args[0]


def test_chunk_rejects_non_4n1_frames_per_chunk():
    with pytest.raises(ValueError, match="4n\\+1"):
        _split(_latent(9), 20, 0)


def test_chunk_rejects_non_5d_latent():
    with pytest.raises(ValueError, match="5-D"):
        _split({"samples": torch.zeros(1, SEEDVR2_LATENT_CHANNELS * 9, 8, 8)}, 21, 0)


def test_chunk_windows_match_source_slices():
    latent = _latent(13)
    chunks, overlap = _split(latent, 21, 2)  # chunk_latent=6, step=4 -> [0:6], [4:10], [8:13]
    src = latent["samples"]
    assert overlap == 2
    assert [c["samples"].shape[2] for c in chunks] == [6, 6, 5]
    assert torch.equal(chunks[0]["samples"], src[:, :, 0:6])
    assert torch.equal(chunks[1]["samples"], src[:, :, 4:10])
    assert torch.equal(chunks[2]["samples"], src[:, :, 8:13])


def test_chunk_short_sequence_passes_through():
    latent = _latent(5)  # t_pixel=17 <= 21
    chunks, overlap = _split(latent, 21, 3)
    assert len(chunks) == 1
    assert overlap == 0
    assert torch.equal(chunks[0]["samples"], latent["samples"])


def test_chunk_overlap_clamps_to_chunk_length():
    chunks, overlap = _split(_latent(13), 21, 999)  # clamps to chunk_latent-1 -> step=1
    assert overlap == 5
    assert len(chunks) == 8
    assert all(c["samples"].shape[2] == 6 for c in chunks)


def test_chunk_slices_temporal_noise_mask_only():
    latent = _latent(13)
    latent["noise_mask"] = torch.rand(1, 1, 13, 8, 8)
    chunks, _ = _split(latent, 21, 0)
    assert [c["noise_mask"].shape[2] for c in chunks] == [6, 6, 1]
    latent["noise_mask"] = torch.rand(1, 1, 8, 8)
    chunks, _ = _split(latent, 21, 0)
    assert all(c["noise_mask"].shape == (1, 1, 8, 8) for c in chunks)


def test_chunk_auto_mode_applies_vram_law(monkeypatch):
    monkeypatch.setattr(comfy.model_management, "get_free_memory", lambda dev=None: 10.8 * (1024 ** 3))
    # budget = 10.8 - 8.5 - 4*0.55 = 0.1 GiB; 32x32 latent = 0.0655 Mpx -> 0.0197 GiB
    # per latent frame -> chunk_latent = 5
    chunks, _ = _split(_latent(13, h=32, w=32), 1, 0, "auto")
    assert [c["samples"].shape[2] for c in chunks] == [5, 5, 3]


def test_crossfade_weights_descend_from_one_to_zero():
    w = _seedvr2_chunk_crossfade_weights(7, torch.device("cpu"), torch.float32)
    assert w[0] == 1.0 and w[-1] == 0.0
    assert torch.all(w[:-1] >= w[1:])
    assert torch.equal(
        _seedvr2_chunk_crossfade_weights(2, torch.device("cpu"), torch.float32),
        torch.tensor([1.0, 0.0]),
    )


def test_merge_zero_overlap_is_exact_concat():
    latent = _latent(13)
    chunks, overlap = _split(latent, 21, 0)
    merged = _merge(chunks, overlap)
    assert torch.equal(merged["samples"], latent["samples"])


def test_merge_round_trips_overlapping_split():
    latent = _latent(13)
    chunks, overlap = _split(latent, 21, 3)
    merged = _merge(chunks, overlap)
    assert merged["samples"].shape == latent["samples"].shape
    assert torch.allclose(merged["samples"], latent["samples"], atol=1e-6)


def test_merge_single_chunk_passes_through():
    latent = _latent(5)
    chunks, overlap = _split(latent, 21, 0)
    merged = _merge(chunks, overlap)
    assert torch.equal(merged["samples"], latent["samples"])


def test_merge_rejects_short_mid_chunk():
    chunks, overlap = _split(_latent(13), 21, 2)
    chunks[0]["samples"] = chunks[0]["samples"][:, :, :4]
    with pytest.raises(ValueError, match="only the final chunk may be shorter"):
        _merge(chunks, overlap)


def test_merge_drops_sliced_mask_keeps_batch_index():
    latent = _latent(13)
    latent["noise_mask"] = torch.rand(1, 1, 13, 8, 8)
    latent["batch_index"] = [0]
    chunks, overlap = _split(latent, 21, 0)
    merged = _merge(chunks, overlap)
    assert "noise_mask" not in merged
    assert merged["batch_index"] == [0]
