import torch

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True


def test_zero_temporal_size_preserves_min_size_when_encode_raises():
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
            temporal_size=0,
            temporal_overlap=0,
            encode=True,
        )
    except RuntimeError as exc:
        if "simulated encode failure" not in str(exc):
            raise
        raised = True

    assert raised
    assert vae.slicing_sample_min_size == 4
