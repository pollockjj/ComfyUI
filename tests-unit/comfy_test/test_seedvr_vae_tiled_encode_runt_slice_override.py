import re
from pathlib import Path

import torch


_VAE_PATH = Path(__file__).resolve().parents[2] / "comfy" / "ldm" / "seedvr" / "vae.py"


def test_encode_branch_delegates_to_wrapper_slicing_encode():
    src = _VAE_PATH.read_text(encoding="utf-8")
    pattern = re.compile(
        r"if encode:\s*\n"
        r"\s*out = vae_model\.encode\(t_chunk\)\[0\]\s*\n"
        r"\s*else:\s*\n"
        r"\s*out = vae_model\.decode_\(t_chunk\)",
    )
    assert pattern.search(src) is not None


def test_slicing_encode_merges_runt_active_tail():
    from comfy.ldm.seedvr.vae import MemoryState, VideoAutoencoderKL, tiled_vae

    class StubVAEModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.slicing_sample_min_size = 4
            self.spatial_downsample_factor = 8
            self.temporal_downsample_factor = 4
            self.device = torch.device("cpu")
            self.use_slicing = True
            self._dummy = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))
            self.memory_states = []
            self.encode_t = []

        def encode(self, t_chunk):
            h = VideoAutoencoderKL.slicing_encode(self, t_chunk)
            return (h, h)

        def _encode(self, x, memory_state=MemoryState.DISABLED):
            self.memory_states.append(memory_state)
            self.encode_t.append(x.shape[2])
            b, c, t_in, h, w = x.shape
            target_d = max(1, (t_in + self.temporal_downsample_factor - 1) // self.temporal_downsample_factor)
            target_h = (h + self.spatial_downsample_factor - 1) // self.spatial_downsample_factor
            target_w = (w + self.spatial_downsample_factor - 1) // self.spatial_downsample_factor
            return torch.zeros((b, 16, target_d, target_h, target_w), dtype=x.dtype)

    vae = StubVAEModel()
    x = torch.zeros((1, 3, 12, 64, 64), dtype=torch.float32)

    tiled_vae(
        x,
        vae,
        tile_size=(64, 64),
        tile_overlap=(0, 0),
        temporal_size=12,
        encode=True,
    )

    assert vae.memory_states == [MemoryState.INITIALIZING, MemoryState.ACTIVE]
    assert vae.encode_t == [5, 7]
    assert min(vae.encode_t[1:]) >= vae.temporal_downsample_factor


def test_slicing_encode_preserves_min_size_when_encode_raises():
    from comfy.ldm.seedvr.vae import tiled_vae

    class RaisingVAEModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.slicing_sample_min_size = 4
            self.spatial_downsample_factor = 8
            self.temporal_downsample_factor = 4
            self.device = torch.device("cpu")
            self._dummy = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))

        def encode(self, t_chunk):
            raise RuntimeError("simulated encode failure")

    vae = RaisingVAEModel()
    x = torch.zeros((1, 3, 12, 64, 64), dtype=torch.float32)

    raised = False
    try:
        tiled_vae(
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

    assert raised
    assert vae.slicing_sample_min_size == 4
