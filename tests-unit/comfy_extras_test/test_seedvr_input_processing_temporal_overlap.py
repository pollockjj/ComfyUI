from types import SimpleNamespace
from unittest.mock import patch

import torch

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

import comfy.model_management  # noqa: E402
from comfy_extras import nodes_seedvr  # noqa: E402


class _FakeFirstStage:
    def __init__(self):
        self.img_dims = None
        self.original_image_video = None
        self.tiled_args = {}

    def modules(self):
        return []


def test_seedvr2_input_processing_wires_temporal_overlap_to_schema_and_tiled_decode_args():
    schema_ids = [i.id for i in nodes_seedvr.SeedVR2InputProcessing.define_schema().inputs]
    assert schema_ids == [
        "images",
        "vae",
        "resolution",
        "spatial_tile_size",
        "spatial_overlap",
        "temporal_tile_size",
        "temporal_overlap",
        "enable_tiling",
    ]

    first_stage = _FakeFirstStage()
    captured = {}

    def _tiled_vae(images, vae_model, **kwargs):
        captured["tiled_vae"] = kwargs
        return torch.zeros(1, 16, 8, 4, 4)

    vae = SimpleNamespace(
        patcher=object(),
        first_stage_model=first_stage,
    )
    images = torch.zeros(1, 8, 32, 32, 3)

    with patch.object(comfy.model_management, "load_models_gpu", lambda *a, **k: None), \
         patch.object(nodes_seedvr, "clear_vae_memory", lambda _vae: None), \
         patch.object(nodes_seedvr, "tiled_vae", _tiled_vae):
        nodes_seedvr.SeedVR2InputProcessing.execute(
            images,
            vae,
            resolution=32,
            spatial_tile_size=32,
            spatial_overlap=8,
            temporal_tile_size=16,
            temporal_overlap=4,
            enable_tiling=True,
        )

    assert captured["tiled_vae"]["temporal_size"] == 16
    assert captured["tiled_vae"]["temporal_overlap"] == 4
    assert captured["tiled_vae"]["encode"] is True
    assert first_stage.tiled_args["temporal_size"] == 16
    assert first_stage.tiled_args["temporal_overlap"] == 4
