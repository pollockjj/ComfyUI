from unittest.mock import patch

import torch
import torch.nn as nn

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

import comfy.ldm.seedvr.vae as vae_mod  # noqa: E402
import comfy.sd as sd_mod  # noqa: E402


def _lab_color_passthrough(content, style):
    return content


def _decode_fingerprint(self, z, return_dict=True):
    b, _, t, h, w = z.shape
    out = torch.empty(b, 3, t, h * 8, w * 8, dtype=z.dtype, device=z.device)
    for batch_idx in range(b):
        out[batch_idx].fill_(float(batch_idx + 1))
    return out


def _make_wrapper(b=2, t=3, enable_tiling=False):
    wrapper = vae_mod.VideoAutoencoderKLWrapper.__new__(
        vae_mod.VideoAutoencoderKLWrapper
    )
    nn.Module.__init__(wrapper)
    wrapper.tiled_args = {"enable_tiling": enable_tiling}
    wrapper.original_image_video = torch.zeros(b, 3, t, 16, 16)
    wrapper.img_dims = (16, 16)
    return wrapper


def test_seedvr2_decode_accepts_5d_bcthw_latents_and_preserves_batch_time_axes():
    wrapper = _make_wrapper(b=2, t=3, enable_tiling=False)
    latent = torch.zeros(2, 16, 3, 2, 2)

    with patch.object(vae_mod.VideoAutoencoderKL, "decode_", _decode_fingerprint), \
         patch.object(vae_mod, "lab_color_transfer", _lab_color_passthrough):
        out = wrapper.decode(latent)

    assert tuple(out.shape) == (2, 3, 3, 16, 16)
    assert out[0, 0, 0, 0, 0].item() == 1.0
    assert out[1, 0, 0, 0, 0].item() == 2.0


class _SeedVR2DecodeStub(vae_mod.VideoAutoencoderKLWrapper):
    def __init__(self):
        nn.Module.__init__(self)
        self.tiled_args = {}
        self.calls = []
        self.original_image_video = torch.zeros(1, 3, 12, 16, 16)
        self.spatial_downsample_factor = 8
        self.temporal_downsample_factor = 4

    def decode(self, z):
        self.calls.append({"tiled_args": dict(self.tiled_args), "shape": tuple(z.shape)})
        return z


def test_seedvr2_decode_tiled_uses_seedvr2_path_not_generic_3d_tiler(monkeypatch):
    vae = sd_mod.VAE.__new__(sd_mod.VAE)
    vae.first_stage_model = _SeedVR2DecodeStub()
    vae.vae_dtype = torch.float32
    vae.device = "cpu"
    vae.output_device = "cpu"
    vae.disable_offload = True
    vae.extra_1d_channel = None
    vae.memory_used_decode = lambda shape, dtype: 1
    vae.process_output = lambda x: x
    vae.patcher = object()

    monkeypatch.setattr(sd_mod.model_management, "load_models_gpu", lambda *a, **k: None)
    monkeypatch.setattr(sd_mod.VAE, "decode_tiled_3d", lambda *a, **k: (_ for _ in ()).throw(AssertionError("generic decode_tiled_3d called")))

    latent = torch.zeros(1, 16, 3, 2, 2)
    out = vae.decode_tiled(latent, tile_x=2, tile_y=2, overlap=1, tile_t=16, overlap_t=4)

    assert tuple(out.shape) == (1, 3, 2, 2, 16)
    assert vae.first_stage_model.calls == [
        {
            "shape": (1, 16, 3, 2, 2),
            "tiled_args": {
                "enable_tiling": True,
                "tile_size": (16, 16),
                "tile_overlap": (8, 8),
                "temporal_size": 16,
                "temporal_overlap": 4,
            },
        }
    ]


def test_seedvr2_decode_tiled_disambiguates_channel_last_temporal_16_latents(monkeypatch):
    vae = sd_mod.VAE.__new__(sd_mod.VAE)
    vae.first_stage_model = _SeedVR2DecodeStub()
    vae.first_stage_model.original_image_video = torch.zeros(1, 3, 64, 64, 64)
    vae.vae_dtype = torch.float32
    vae.device = "cpu"
    vae.output_device = "cpu"
    vae.disable_offload = True
    vae.extra_1d_channel = None
    vae.latent_channels = 16
    vae.memory_used_decode = lambda shape, dtype: 1
    vae.process_output = lambda x: x
    vae.patcher = object()

    monkeypatch.setattr(sd_mod.model_management, "load_models_gpu", lambda *a, **k: None)
    monkeypatch.setattr(sd_mod.VAE, "decode_tiled_3d", lambda *a, **k: (_ for _ in ()).throw(AssertionError("generic decode_tiled_3d called")))

    latent = torch.zeros(1, 16, 8, 8, 16)
    vae.decode_tiled(latent, tile_x=2, tile_y=2, overlap=1, tile_t=16, overlap_t=4)

    assert vae.first_stage_model.calls[0]["shape"] == (1, 16, 16, 8, 8)


def test_seedvr2_decode_tiled_routes_collapsed_latents_to_seedvr2_tiler(monkeypatch):
    vae = sd_mod.VAE.__new__(sd_mod.VAE)
    vae.first_stage_model = _SeedVR2DecodeStub()
    vae.vae_dtype = torch.float32
    vae.device = "cpu"
    vae.output_device = "cpu"
    vae.disable_offload = True
    vae.extra_1d_channel = None
    vae.latent_channels = 16
    vae.memory_used_decode = lambda shape, dtype: 1
    vae.process_output = lambda x: x
    vae.patcher = object()

    monkeypatch.setattr(sd_mod.model_management, "load_models_gpu", lambda *a, **k: None)
    monkeypatch.setattr(sd_mod.VAE, "decode_tiled_", lambda *a, **k: (_ for _ in ()).throw(AssertionError("generic decode_tiled_ called")))

    latent = torch.zeros(1, 48, 2, 2)
    vae.decode_tiled(latent, tile_x=2, tile_y=2, overlap=1, tile_t=16, overlap_t=4)

    assert vae.first_stage_model.calls[0]["shape"] == (1, 48, 2, 2)
    assert vae.first_stage_model.calls[0]["tiled_args"]["temporal_overlap"] == 4


class _TemporalChunkRecorder(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(()))
        self.device = "cpu"
        self.spatial_downsample_factor = 1
        self.temporal_downsample_factor = 4
        self.chunks = []

    def decode_(self, z):
        self.chunks.append([int(v) for v in z[0, 0, :, 0, 0].tolist()])
        pieces = [z[:, :1, :1]]
        if z.shape[2] > 1:
            pieces.append(z[:, :1, 1:].repeat_interleave(4, dim=2))
        return torch.cat(pieces, dim=2)


def test_seedvr2_tiled_vae_decode_uses_temporal_overlap_prefix():
    recorder = _TemporalChunkRecorder()
    latent = torch.arange(6, dtype=torch.float32).view(1, 1, 6, 1, 1)

    out = vae_mod.tiled_vae(
        latent,
        recorder,
        tile_size=(1, 1),
        tile_overlap=(0, 0),
        temporal_size=16,
        temporal_overlap=4,
        encode=False,
    )

    assert recorder.chunks == [[0, 1, 2, 3], [3, 4, 5, 5, 5]]
    assert tuple(out.shape) == (1, 1, 21, 1, 1)
    assert [int(v) for v in out[0, 0, [0, 1, 5, 9, 13, 17], 0, 0].tolist()] == [0, 1, 2, 3, 4, 5]
