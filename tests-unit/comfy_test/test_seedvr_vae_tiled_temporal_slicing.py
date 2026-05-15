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


def test_decode_tiled_uses_temporal_size_for_wrapper_slicing():
    vae = _SlicingDecodeVAE(slicing_latent_min_size=999)
    z = torch.arange(1 * 16 * 5 * 8 * 8, dtype=torch.float32).reshape(1, 16, 5, 8, 8)

    tiled_vae(
        z,
        vae,
        tile_size=(64, 64),
        tile_overlap=(0, 0),
        temporal_size=8,
        temporal_overlap=4,
        encode=False,
    )

    assert vae.decode_min_sizes == [2]
    assert vae.memory_states == [MemoryState.INITIALIZING, MemoryState.ACTIVE]
    assert vae.slicing_latent_min_size == 999


def test_decode_wrapper_passes_temporal_overlap_to_tiled_vae():
    wrapper = vae_mod.VideoAutoencoderKLWrapper.__new__(
        vae_mod.VideoAutoencoderKLWrapper
    )
    nn.Module.__init__(wrapper)
    wrapper.tiled_args = {
        "enable_tiling": True,
        "tile_size": (64, 64),
        "tile_overlap": (0, 0),
        "temporal_size": 8,
        "temporal_overlap": 7,
    }
    wrapper.original_image_video = torch.zeros(1, 3, 1, 16, 16)
    wrapper.img_dims = (16, 16)

    captured = {}

    def _fake_tiled_vae(latent, model, **kwargs):
        captured.update(kwargs)
        return torch.zeros(1, 3, 1, 16, 16)

    with (
        patch.object(vae_mod, "tiled_vae", side_effect=_fake_tiled_vae),
        patch.object(vae_mod, "lab_color_transfer", side_effect=lambda content, style: content),
    ):
        wrapper.decode(torch.zeros(1, 16, 2, 2))

    assert captured["temporal_overlap"] == 7


def test_decode_tiled_output_matches_causal_slicing_reference():
    z = torch.arange(1 * 16 * 7 * 8 * 8, dtype=torch.float32).reshape(1, 16, 7, 8, 8)

    reference_vae = _SlicingDecodeVAE(slicing_latent_min_size=3)
    expected = reference_vae.decode_(z)

    tiled_vae_model = _SlicingDecodeVAE(slicing_latent_min_size=999)
    actual = tiled_vae(
        z,
        tiled_vae_model,
        tile_size=(64, 64),
        tile_overlap=(0, 0),
        temporal_size=12,
        temporal_overlap=4,
        encode=False,
    )

    assert torch.equal(actual, expected)
    assert tiled_vae_model.memory_states == [
        MemoryState.INITIALIZING,
        MemoryState.ACTIVE,
    ]
    assert tiled_vae_model.slicing_latent_min_size == 999
