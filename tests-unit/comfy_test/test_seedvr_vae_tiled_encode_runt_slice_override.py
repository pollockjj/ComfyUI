import re
from pathlib import Path

import torch


_VAE_PATH = Path(__file__).resolve().parents[2] / "comfy" / "ldm" / "seedvr" / "vae.py"


def test_encode_branch_sets_slicing_sample_min_size_to_chunk_length():
    """tiled_vae's encode branch must disable inner slicing per outer chunk by setting
    slicing_sample_min_size = t_chunk.shape[2]. The previous clamp
    `min(t_chunk.shape[2], max(old, sf_t * 2))` only happened to satisfy
    slicing_encode's `(T-1) > min_size` predicate for temporal_tile_size=16; for
    schema-allowed values like 12 it produced a runt T=3 ACTIVE slice that
    underflowed the second temporal downsampler. See PR #51 review threads
    discussion_r3205644091 / discussion_r3205651100."""
    src = _VAE_PATH.read_text(encoding="utf-8")
    encode_block = re.search(
        r"if encode:\s*\n"
        r"(?:\s*#[^\n]*\n)*"
        r"\s*old_slicing_sample_min_size = getattr\(vae_model, \"slicing_sample_min_size\", None\)\s*\n"
        r"\s*if old_slicing_sample_min_size is not None:\s*\n"
        r"\s*vae_model\.slicing_sample_min_size = ([^\n]+)\n",
        src,
    )
    assert encode_block is not None, (
        "tiled_vae encode branch must override vae_model.slicing_sample_min_size; "
        f"pattern not found in {_VAE_PATH}."
    )
    rhs = encode_block.group(1).strip()
    assert rhs == "t_chunk.shape[2]", (
        f"tiled_vae encode branch must set slicing_sample_min_size = t_chunk.shape[2] "
        f"to disable inner slicing per outer chunk; found `{rhs}`. "
        f"The bare-minimum clamp `min(...)` is insufficient: see PR #51 discussion."
    )


def test_encode_branch_restores_slicing_sample_min_size_via_try_finally():
    """The override must be restored after the encode call so the wrapper's state
    is not mutated across calls."""
    src = _VAE_PATH.read_text(encoding="utf-8")
    pattern = re.compile(
        r"vae_model\.slicing_sample_min_size = t_chunk\.shape\[2\]\s*\n"
        r"\s*try:\s*\n"
        r"\s*out = vae_model\.encode\(t_chunk\)\[0\]\s*\n"
        r"\s*finally:\s*\n"
        r"\s*if old_slicing_sample_min_size is not None:\s*\n"
        r"\s*vae_model\.slicing_sample_min_size = old_slicing_sample_min_size",
    )
    assert pattern.search(src) is not None, (
        "tiled_vae encode branch must restore slicing_sample_min_size via try/finally "
        f"after vae_model.encode; pattern not found in {_VAE_PATH}."
    )


def test_runtime_override_disables_slicing_for_chunk_and_restores():
    """Runtime contract test: for any t_chunk_len, the override must make
    slicing_encode's `(T-1) > min_size` predicate FALSE during encode (no inner
    slicing) and the original min_size must be restored after encode returns —
    even when encode raises.

    Uses a stand-in vae_model with no real encoder weights; the goal is to
    observe slicing_sample_min_size mutation/restoration semantics without
    requiring SeedVR2 safetensors."""
    from comfy.ldm.seedvr.vae import tiled_vae

    observed = {"during_encode": [], "raised": False}

    class StubVAEModel(torch.nn.Module):
        def __init__(self, original_min_size):
            super().__init__()
            self.slicing_sample_min_size = original_min_size
            # Minimal attribute surface tiled_vae reads.
            self.spatial_downsample_factor = 8
            self.temporal_downsample_factor = 4
            self.device = torch.device("cpu")
            # tiled_vae calls next(vae_model.parameters()).dtype; provide one.
            self._dummy = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))

        def encode(self, t_chunk):
            # Capture the override value AT the point encode runs.
            observed["during_encode"].append(self.slicing_sample_min_size)
            # Slicing predicate `(T-1) > min_size` must be False for the
            # chunk; equivalently min_size must be >= T-1. Stronger: PR #51
            # sets min_size = T exactly.
            assert self.slicing_sample_min_size >= t_chunk.shape[2] - 1, (
                f"slicing_sample_min_size={self.slicing_sample_min_size} fails "
                f"`(T-1) > min_size` => False predicate for t_chunk T={t_chunk.shape[2]}"
            )
            # Return a 5D BCDHW latent with the right downsampled shape so
            # tiled_vae's combine loop runs.
            b, c, t_in, h, w = t_chunk.shape
            sf_s = self.spatial_downsample_factor
            sf_t = self.temporal_downsample_factor
            target_d = (t_in + sf_t - 1) // sf_t
            target_h = (h + sf_s - 1) // sf_s
            target_w = (w + sf_s - 1) // sf_s
            z = torch.zeros((b, 16, target_d, target_h, target_w), dtype=t_chunk.dtype)
            return (z, z)

    original_min_size = 4
    vae = StubVAEModel(original_min_size)

    # T=12 is the schema-allowed value the original clamp-bug failed on.
    x = torch.zeros((1, 3, 12, 64, 64), dtype=torch.float32)
    _ = tiled_vae(
        x,
        vae,
        tile_size=(64, 64),
        tile_overlap=(0, 0),
        temporal_size=12,
        encode=True,
    )

    # The override must have been applied during encode.
    assert observed["during_encode"], "vae_model.encode was never called"
    for during in observed["during_encode"]:
        assert during >= 12 - 1, (
            f"during encode, slicing_sample_min_size={during} did not satisfy "
            f"`(T-1) > min_size` => False predicate for T=12"
        )
    # And restored after.
    assert vae.slicing_sample_min_size == original_min_size, (
        f"slicing_sample_min_size was not restored after encode: "
        f"got {vae.slicing_sample_min_size}, expected {original_min_size}"
    )


def test_runtime_override_restores_when_encode_raises():
    """try/finally must restore slicing_sample_min_size even when encode raises."""
    from comfy.ldm.seedvr.vae import tiled_vae

    class RaisingVAEModel(torch.nn.Module):
        def __init__(self, original_min_size):
            super().__init__()
            self.slicing_sample_min_size = original_min_size
            self.spatial_downsample_factor = 8
            self.temporal_downsample_factor = 4
            self.device = torch.device("cpu")
            self._dummy = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))

        def encode(self, t_chunk):
            raise RuntimeError("simulated encode failure")

    original_min_size = 4
    vae = RaisingVAEModel(original_min_size)
    x = torch.zeros((1, 3, 12, 64, 64), dtype=torch.float32)
    raised = False
    try:
        _ = tiled_vae(
            x,
            vae,
            tile_size=(64, 64),
            tile_overlap=(0, 0),
            temporal_size=12,
            encode=True,
        )
    except RuntimeError as exc:
        if "simulated encode failure" not in str(exc):
            raise
        raised = True
    assert raised, "Expected encode to raise RuntimeError"
    assert vae.slicing_sample_min_size == original_min_size, (
        f"slicing_sample_min_size was not restored after raised encode: "
        f"got {vae.slicing_sample_min_size}, expected {original_min_size}"
    )
