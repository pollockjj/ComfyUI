"""Regression tests for the decode-side ``slicing_latent_min_size`` override
in ``tiled_vae``. Mirrors the encode-side override tests
(``test_seedvr_vae_tiled_encode_runt_slice_override.py``) — the decode branch
temporarily mutates ``vae_model.slicing_latent_min_size`` to bound inner
slicing per call and must restore the prior value via try/finally, even
when ``decode_`` raises.
"""

import re
from pathlib import Path

import torch


_VAE_PATH = Path(__file__).resolve().parents[2] / "comfy" / "ldm" / "seedvr" / "vae.py"


def test_decode_branch_sets_slicing_latent_min_size_to_chunk_length():
    """tiled_vae's decode branch must disable inner slicing per outer chunk by
    setting ``slicing_latent_min_size = t_chunk.shape[2]``. Without this, the
    same runt-slice failure mode that affects encode would manifest on decode
    when ``(t_chunk_len - 1) % slicing_latent_min_size != 0``."""
    src = _VAE_PATH.read_text(encoding="utf-8")
    decode_block = re.search(
        r"else:\s*\n"
        r"\s*old_slicing_latent_min_size = getattr\(vae_model, \"slicing_latent_min_size\", None\)\s*\n"
        r"\s*if old_slicing_latent_min_size is not None:\s*\n"
        r"\s*vae_model\.slicing_latent_min_size = ([^\n]+)\n",
        src,
    )
    assert decode_block is not None, (
        "tiled_vae decode branch must override vae_model.slicing_latent_min_size; "
        f"pattern not found in {_VAE_PATH}."
    )
    rhs = decode_block.group(1).strip()
    assert rhs == "t_chunk.shape[2]", (
        "tiled_vae decode branch must set slicing_latent_min_size = "
        f"t_chunk.shape[2] to disable inner slicing per outer chunk; found `{rhs}`."
    )


def test_decode_branch_restores_slicing_latent_min_size_via_try_finally():
    """The override must be restored after ``decode_`` returns so the wrapper's
    state is not mutated across calls."""
    src = _VAE_PATH.read_text(encoding="utf-8")
    pattern = re.compile(
        r"vae_model\.slicing_latent_min_size = t_chunk\.shape\[2\]\s*\n"
        r"\s*try:\s*\n"
        r"\s*out = vae_model\.decode_\(t_chunk\)\s*\n"
        r"\s*finally:\s*\n"
        r"\s*if old_slicing_latent_min_size is not None:\s*\n"
        r"\s*vae_model\.slicing_latent_min_size = old_slicing_latent_min_size",
    )
    assert pattern.search(src) is not None, (
        "tiled_vae decode branch must restore slicing_latent_min_size via "
        f"try/finally after vae_model.decode_; pattern not found in {_VAE_PATH}."
    )


def test_runtime_decode_override_disables_slicing_for_chunk_and_restores():
    """Runtime contract: at the moment ``decode_`` runs, the override must
    have set ``slicing_latent_min_size = t_chunk.shape[2]`` so
    ``slicing_decode``'s ``(T-1) > slicing_latent_min_size`` predicate is
    FALSE for the local t_chunk. The original value must be restored after.

    Note: ``run_temporal_chunks`` sub-chunks the input latent T by
    ``input_chunk = max(1, temporal_size // sf_t)``; we use parameters that
    yield exactly one iteration so the override target equals the input
    latent T."""
    from comfy.ldm.seedvr.vae import tiled_vae

    observed = {"during_decode_min_size": [], "during_decode_t": []}

    class StubVAEModel(torch.nn.Module):
        def __init__(self, original_min_size):
            super().__init__()
            self.slicing_latent_min_size = original_min_size
            self.spatial_downsample_factor = 8
            self.temporal_downsample_factor = 4
            self.device = torch.device("cpu")
            self._dummy = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))

        def decode_(self, t_chunk):
            observed["during_decode_min_size"].append(self.slicing_latent_min_size)
            observed["during_decode_t"].append(t_chunk.shape[2])
            # Pin the contract: override must equal the local t_chunk T,
            # which is what makes slicing_decode's `(T-1) > min_size`
            # predicate false.
            assert self.slicing_latent_min_size == t_chunk.shape[2], (
                f"override must set slicing_latent_min_size to t_chunk.shape[2]; "
                f"got {self.slicing_latent_min_size} vs {t_chunk.shape[2]}"
            )
            assert (t_chunk.shape[2] - 1) <= self.slicing_latent_min_size, (
                f"`(T-1) > min_size` => False predicate must hold; "
                f"T-1={t_chunk.shape[2]-1}, min_size={self.slicing_latent_min_size}"
            )
            b, c, d, h, w = t_chunk.shape
            sf_s = self.spatial_downsample_factor
            sf_t = self.temporal_downsample_factor
            target_d = max(1, d * sf_t - (sf_t - 1))
            target_h = h * sf_s
            target_w = w * sf_s
            return torch.zeros((b, 3, target_d, target_h, target_w), dtype=t_chunk.dtype)

    # Pick a very large original min_size so the override is unmistakably
    # different and restoration is observable.
    original_min_size = 999
    vae = StubVAEModel(original_min_size)

    # Latent z T=4; with temporal_size=16 and sf_t=4 we get
    # input_chunk = max(1, 16 // 4) = 4 → exactly one decode_ call with
    # t_chunk T=4.
    z = torch.zeros((1, 16, 4, 8, 8), dtype=torch.float32)
    _ = tiled_vae(
        z,
        vae,
        tile_size=(64, 64),
        tile_overlap=(0, 0),
        temporal_size=16,
        encode=False,
    )

    assert observed["during_decode_min_size"], "vae_model.decode_ was never called"
    # Each call must observe min_size == t_chunk.shape[2] (override active),
    # not the original 999.
    for min_size, t in zip(
        observed["during_decode_min_size"], observed["during_decode_t"], strict=True,
    ):
        assert min_size != original_min_size, (
            f"override did not fire during decode_; saw min_size={min_size} "
            f"== original={original_min_size}"
        )
        assert min_size == t, (
            f"override must equal t_chunk.shape[2]; got {min_size} vs {t}"
        )
    # And restored to original after the outer call.
    assert vae.slicing_latent_min_size == original_min_size, (
        "slicing_latent_min_size was not restored after decode_: "
        f"got {vae.slicing_latent_min_size}, expected {original_min_size}"
    )


def test_runtime_decode_override_restores_when_decode_raises():
    """try/finally must restore slicing_latent_min_size even when decode_ raises."""
    from comfy.ldm.seedvr.vae import tiled_vae

    class RaisingVAEModel(torch.nn.Module):
        def __init__(self, original_min_size):
            super().__init__()
            self.slicing_latent_min_size = original_min_size
            self.spatial_downsample_factor = 8
            self.temporal_downsample_factor = 4
            self.device = torch.device("cpu")
            self._dummy = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))

        def decode_(self, t_chunk):
            raise RuntimeError("simulated decode failure")

    original_min_size = 1
    vae = RaisingVAEModel(original_min_size)
    z = torch.zeros((1, 16, 4, 8, 8), dtype=torch.float32)
    raised = False
    try:
        _ = tiled_vae(
            z,
            vae,
            tile_size=(64, 64),
            tile_overlap=(0, 0),
            temporal_size=4,
            encode=False,
        )
    except RuntimeError as exc:
        if "simulated decode failure" not in str(exc):
            raise
        raised = True
    assert raised, "Expected decode_ to raise RuntimeError"
    assert vae.slicing_latent_min_size == original_min_size, (
        "slicing_latent_min_size was not restored after raised decode_: "
        f"got {vae.slicing_latent_min_size}, expected {original_min_size}"
    )
