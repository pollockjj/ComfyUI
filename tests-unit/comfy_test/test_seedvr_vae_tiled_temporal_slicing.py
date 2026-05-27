from unittest.mock import patch

import torch
import torch.nn as nn

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

import comfy.ldm.seedvr.vae as vae_mod  # noqa: E402
from comfy.ldm.seedvr.vae import MemoryState, tiled_vae  # noqa: E402


class _SlicingDecodeVAE(nn.Module):
    def __init__(self, slicing_latent_min_size):
        super().__init__()
        self.slicing_latent_min_size = slicing_latent_min_size
        self.spatial_downsample_factor = 8
        self.temporal_downsample_factor = 4
        self.device = torch.device("cpu")
        self.use_slicing = True
        self._dummy = nn.Parameter(torch.zeros(1, dtype=torch.float32))
        self.decode_min_sizes = []
        self.memory_states = []

    def decode_(self, z):
        self.decode_min_sizes.append(self.slicing_latent_min_size)
        return vae_mod.VideoAutoencoderKL.slicing_decode(self, z)

    def _decode(self, z, memory_state=MemoryState.DISABLED):
        self.memory_states.append(memory_state)
        x = z[:, :1].repeat(
            1,
            3,
            1,
            self.spatial_downsample_factor,
            self.spatial_downsample_factor,
        )
        return x


class _EncodeVAE(nn.Module):
    def __init__(self, slicing_sample_min_size):
        super().__init__()
        self.slicing_sample_min_size = slicing_sample_min_size
        self.spatial_downsample_factor = 8
        self.temporal_downsample_factor = 4
        self.device = torch.device("cpu")
        self.use_slicing = True
        self._dummy = nn.Parameter(torch.zeros(1, dtype=torch.float32))
        self.memory_states = []
        self.encoded_t = []
        self.encode_min_sizes = []

    def encode(self, t_chunk):
        self.encode_min_sizes.append(self.slicing_sample_min_size)
        h = vae_mod.VideoAutoencoderKL.slicing_encode(self, t_chunk)
        return (h, h)

    def _encode(self, x, memory_state=MemoryState.DISABLED):
        self.memory_states.append(memory_state)
        self.encoded_t.append(x.shape[2])
        b, c, t_in, h, w = x.shape
        target_d = max(1, (t_in + self.temporal_downsample_factor - 1) // self.temporal_downsample_factor)
        target_h = (h + self.spatial_downsample_factor - 1) // self.spatial_downsample_factor
        target_w = (w + self.spatial_downsample_factor - 1) // self.spatial_downsample_factor
        z = torch.zeros((b, 16, target_d, target_h, target_w), dtype=x.dtype)
        return z


def test_decode_tiled_vae_maps_temporal_args_to_latent_slicing_min_size():
    vae = _SlicingDecodeVAE(slicing_latent_min_size=2)
    z = torch.arange(1 * 16 * 5 * 8 * 8, dtype=torch.float32).reshape(1, 16, 5, 8, 8)

    tiled_vae(
        z,
        vae,
        tile_size=(64, 64),
        tile_overlap=(0, 0),
        temporal_size=12,
        temporal_overlap=4,
        encode=False,
    )

    assert vae.decode_min_sizes == [2]
    assert vae.memory_states == [MemoryState.INITIALIZING, MemoryState.ACTIVE]
    assert vae.slicing_latent_min_size == 2

    wrapper = vae_mod.VideoAutoencoderKLWrapper.__new__(
        vae_mod.VideoAutoencoderKLWrapper
    )
    nn.Module.__init__(wrapper)
    seedvr2_tiling = {
        "enable_tiling": True,
        "tile_size": (64, 64),
        "tile_overlap": (0, 0),
        "temporal_size": 8,
        "temporal_overlap": 7,
    }

    captured = {}

    def _fake_tiled_vae(latent, model, **kwargs):
        captured.update(kwargs)
        return torch.zeros(1, 3, 1, 16, 16)

    with (
        patch.object(vae_mod, "tiled_vae", side_effect=_fake_tiled_vae),
        patch.object(vae_mod, "lab_color_transfer", side_effect=lambda content, style: content),
    ):
        wrapper.decode(torch.zeros(1, 16, 2, 2), seedvr2_tiling=seedvr2_tiling)

    assert captured["temporal_overlap"] == 7


def test_encode_tiled_vae_zero_temporal_size_disables_wrapper_slicing():
    vae = _EncodeVAE(slicing_sample_min_size=4)
    x = torch.zeros((1, 3, 12, 64, 64), dtype=torch.float32)

    tiled_vae(
        x,
        vae,
        tile_size=(64, 64),
        tile_overlap=(0, 0),
        temporal_size=0,
        temporal_overlap=0,
        encode=True,
    )

    assert vae.encode_min_sizes == [12]
    assert vae.memory_states == [MemoryState.DISABLED]
    assert vae.encoded_t == [12]
    assert vae.slicing_sample_min_size == 4
