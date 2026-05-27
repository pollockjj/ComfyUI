import torch

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

import comfy_extras.nodes_seedvr as nodes_seedvr
import nodes


def test_seedvr2_postprocessing_restores_flat_decoded_batch_time():
    decoded = torch.arange(6 * 4 * 6 * 1, dtype=torch.float32).reshape(6, 4, 6, 1)
    original = torch.ones((2, 3, 4, 6, 1), dtype=torch.float32)

    output = nodes_seedvr.SeedVR2PostProcessing.execute(decoded, original, 4, "none").result[0]

    assert output.shape == (6, 4, 6, 1)
    torch.testing.assert_close(output, decoded)


def test_seedvr2_decode_node_ignores_seedvr2_sideband_metadata():
    class FakeVAE:
        def __init__(self):
            self.decode_call = None

        def decode(self, samples, **kwargs):
            self.decode_call = kwargs
            return torch.zeros((1, 1, 2, 2, 3), dtype=torch.float32)

    vae = FakeVAE()
    samples = {
        "samples": torch.zeros((1, 16, 4, 4, 16), dtype=torch.float32),
        "seedvr2_channel_last": True,
    }

    nodes.VAEDecode().decode(vae, samples)

    assert "seedvr2_channel_last" not in vae.decode_call


def test_seedvr2_encode_node_does_not_mark_model_specific_layout_metadata():
    class FakeVAE:
        def encode(self, pixels):
            return torch.zeros((1, 16, 2, 3, 4), dtype=torch.float32)

    output = nodes.VAEEncode().encode(FakeVAE(), torch.zeros((1, 8, 8, 3)))[0]

    assert set(output) == {"samples"}


def test_seedvr2_tiled_decode_node_preserves_legacy_decode_tiled_signature():
    class FakeVAE:
        def __init__(self):
            self.decode_call = None

        def temporal_compression_decode(self):
            return 4

        def spacial_compression_decode(self):
            return 8

        def decode_tiled(self, samples, tile_x, tile_y, overlap, tile_t, overlap_t):
            self.decode_call = {
                "tile_x": tile_x,
                "tile_y": tile_y,
                "overlap": overlap,
                "tile_t": tile_t,
                "overlap_t": overlap_t,
            }
            return torch.zeros((1, 1, 2, 2, 3), dtype=torch.float32)

    vae = FakeVAE()
    samples = {"samples": torch.zeros((1, 16, 4, 4, 16), dtype=torch.float32)}

    nodes.VAEDecodeTiled().decode(
        vae,
        samples,
        tile_size=64,
        overlap=0,
        temporal_size=64,
        temporal_overlap=8,
    )

    assert vae.decode_call == {
        "tile_x": 8,
        "tile_y": 8,
        "overlap": 0,
        "tile_t": 16,
        "overlap_t": 2,
    }
