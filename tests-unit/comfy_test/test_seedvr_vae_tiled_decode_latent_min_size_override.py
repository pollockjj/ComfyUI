import re
from pathlib import Path

import torch


_VAE_PATH = Path(__file__).resolve().parents[2] / "comfy" / "ldm" / "seedvr" / "vae.py"


def test_decode_branch_delegates_to_wrapper_slicing_decode():
    src = _VAE_PATH.read_text(encoding="utf-8")
    pattern = re.compile(
        r"if encode:\s*\n"
        r"\s*out = vae_model\.encode\(t_chunk\)\[0\]\s*\n"
        r"\s*else:\s*\n"
        r"\s*out = vae_model\.decode_\(t_chunk\)",
    )
    assert pattern.search(src) is not None


def test_runtime_decode_uses_existing_slicing_latent_min_size():
    from comfy.ldm.seedvr.vae import MemoryState, VideoAutoencoderKL, tiled_vae

    class StubVAEModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.slicing_latent_min_size = 2
            self.spatial_downsample_factor = 8
            self.temporal_downsample_factor = 4
            self.device = torch.device("cpu")
            self.use_slicing = True
            self._dummy = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))
            self.decode_min_sizes = []
            self.memory_states = []

        def decode_(self, t_chunk):
            self.decode_min_sizes.append(self.slicing_latent_min_size)
            return VideoAutoencoderKL.slicing_decode(self, t_chunk)

        def _decode(self, z, memory_state=MemoryState.DISABLED):
            self.memory_states.append(memory_state)
            b, c, d, h, w = z.shape
            return torch.zeros((b, 3, d, h * 8, w * 8), dtype=z.dtype)

    vae = StubVAEModel()
    z = torch.zeros((1, 16, 5, 8, 8), dtype=torch.float32)

    tiled_vae(
        z,
        vae,
        tile_size=(64, 64),
        tile_overlap=(0, 0),
        temporal_size=16,
        encode=False,
    )

    assert vae.decode_min_sizes == [2]
    assert vae.memory_states == [MemoryState.INITIALIZING, MemoryState.ACTIVE]
    assert vae.slicing_latent_min_size == 2


def test_runtime_decode_preserves_min_size_when_decode_raises():
    from comfy.ldm.seedvr.vae import tiled_vae

    class RaisingVAEModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.slicing_latent_min_size = 2
            self.spatial_downsample_factor = 8
            self.temporal_downsample_factor = 4
            self.device = torch.device("cpu")
            self._dummy = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))

        def decode_(self, t_chunk):
            raise RuntimeError("simulated decode failure")

    vae = RaisingVAEModel()
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
    assert vae.slicing_latent_min_size == 2
