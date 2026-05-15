import re
from pathlib import Path

import torch


_VAE_PATH = Path(__file__).resolve().parents[2] / "comfy" / "ldm" / "seedvr" / "vae.py"


def test_decode_branch_sets_slicing_latent_min_size_from_temporal_size():
    src = _VAE_PATH.read_text(encoding="utf-8")
    decode_block = re.search(
        r"else:\s*\n"
        r"\s*old_slicing_latent_min_size = getattr\(vae_model, \"slicing_latent_min_size\", None\)\s*\n"
        r"\s*if old_slicing_latent_min_size is not None:\s*\n"
        r"\s*vae_model\.slicing_latent_min_size = ([^\n]+)\n",
        src,
    )
    assert decode_block is not None
    assert decode_block.group(1).strip() == "max(1, temporal_size // sf_t)"


def test_decode_branch_restores_slicing_latent_min_size_via_try_finally():
    src = _VAE_PATH.read_text(encoding="utf-8")
    pattern = re.compile(
        r"vae_model\.slicing_latent_min_size = max\(1, temporal_size // sf_t\)\s*\n"
        r"\s*try:\s*\n"
        r"\s*out = vae_model\.decode_\(t_chunk\)\s*\n"
        r"\s*finally:\s*\n"
        r"\s*if old_slicing_latent_min_size is not None:\s*\n"
        r"\s*vae_model\.slicing_latent_min_size = old_slicing_latent_min_size",
    )
    assert pattern.search(src) is not None


def test_runtime_decode_override_uses_temporal_size_and_restores():
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
            b, c, d, h, w = t_chunk.shape
            sf_s = self.spatial_downsample_factor
            sf_t = self.temporal_downsample_factor
            target_d = max(1, d * sf_t - (sf_t - 1))
            target_h = h * sf_s
            target_w = w * sf_s
            return torch.zeros((b, 3, target_d, target_h, target_w), dtype=t_chunk.dtype)

    original_min_size = 999
    vae = StubVAEModel(original_min_size)
    z = torch.zeros((1, 16, 4, 8, 8), dtype=torch.float32)

    tiled_vae(
        z,
        vae,
        tile_size=(64, 64),
        tile_overlap=(0, 0),
        temporal_size=16,
        encode=False,
    )

    assert observed["during_decode_min_size"] == [4]
    assert observed["during_decode_t"] == [4]
    assert vae.slicing_latent_min_size == original_min_size


def test_runtime_decode_override_restores_when_decode_raises():
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

    original_min_size = 999
    vae = RaisingVAEModel(original_min_size)
    z = torch.zeros((1, 16, 4, 8, 8), dtype=torch.float32)

    raised = False
    try:
        tiled_vae(
            z,
            vae,
            tile_size=(64, 64),
            tile_overlap=(0, 0),
            temporal_size=16,
            encode=False,
        )
    except RuntimeError as exc:
        if "simulated decode failure" not in str(exc):
            raise
        raised = True

    assert raised
    assert vae.slicing_latent_min_size == original_min_size
