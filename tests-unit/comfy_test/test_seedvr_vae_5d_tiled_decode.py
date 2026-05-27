import torch
import torch.nn as nn

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

import comfy.ldm.seedvr.vae as vae_mod  # noqa: E402
import comfy.sd as sd_mod  # noqa: E402
import nodes as nodes_mod  # noqa: E402


class _SeedVR2DecodeStub(vae_mod.VideoAutoencoderKLWrapper):
    def __init__(self):
        nn.Module.__init__(self)
        self.tiled_args = {}
        self.calls = []
        self.original_image_video = torch.zeros(1, 3, 12, 16, 16)
        self.spatial_downsample_factor = 8
        self.temporal_downsample_factor = 4

    def decode(self, z, seedvr2_tiling=None):
        self.calls.append({"seedvr2_tiling": seedvr2_tiling, "shape": tuple(z.shape)})
        return z


def test_vae_decode_tiled_allows_zero_temporal_controls_and_passes_them_through():
    input_types = nodes_mod.VAEDecodeTiled.INPUT_TYPES()["required"]
    assert input_types["temporal_size"][1]["min"] == 0
    assert input_types["temporal_overlap"][1]["min"] == 0
    assert "SeedVR2 allows 0" in input_types["temporal_size"][1]["tooltip"]

    class _DecodeRecorder:
        def __init__(self):
            self.calls = []

        def temporal_compression_decode(self):
            return 4

        def spacial_compression_decode(self):
            return 8

        def decode_tiled(self, samples, **kwargs):
            self.calls.append({"shape": tuple(samples.shape), **kwargs})
            return torch.zeros(1, 8, 8, 3)

    recorder = _DecodeRecorder()
    node = nodes_mod.VAEDecodeTiled()

    node.decode(
        recorder,
        {"samples": torch.zeros(1, 16, 3, 32, 32)},
        tile_size=256,
        overlap=64,
        temporal_size=0,
        temporal_overlap=0,
    )

    assert recorder.calls == [
        {
            "shape": (1, 16, 3, 32, 32),
            "tile_x": 32,
            "tile_y": 32,
            "overlap": 8,
            "tile_t": 0,
            "overlap_t": 0,
        }
    ]


def test_seedvr2_decode_tiled_uses_seedvr2_path_not_generic_3d_tiler(monkeypatch):
    vae = sd_mod.VAE.__new__(sd_mod.VAE)
    vae.first_stage_model = _SeedVR2DecodeStub()
    vae.vae_dtype = torch.float32
    vae.device = torch.device("cpu")
    vae.output_device = torch.device("cpu")
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
            "seedvr2_tiling": {
                "enable_tiling": True,
                "tile_size": (16, 16),
                "tile_overlap": (8, 8),
                "temporal_size": 64,
                "temporal_overlap": 16,
            },
        }
    ]
