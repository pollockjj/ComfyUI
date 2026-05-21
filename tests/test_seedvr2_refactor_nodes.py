from types import SimpleNamespace

import torch

import comfy.sd
import comfy_extras.nodes_seedvr as nodes_seedvr
import nodes


def test_seedvr2_postprocessing_restores_flat_decoded_batch_time():
    decoded = torch.arange(6 * 4 * 6 * 1, dtype=torch.float32).reshape(6, 4, 6, 1)
    reference = torch.ones((2, 3, 4, 6, 1), dtype=torch.float32)

    (output,) = nodes_seedvr.SeedVR2PostProcessing.execute(decoded, reference, "none")

    assert output.shape == (6, 4, 6, 1)
    torch.testing.assert_close(output, decoded)


def test_seedvr2_postprocessing_crops_processed_reference_padding():
    decoded = torch.ones((1, 128, 176, 3), dtype=torch.float32)
    reference = torch.zeros((1, 1, 128, 176, 3), dtype=torch.float32)
    reference[:, :, :120, :169, :] = 0.25
    reference[:, :, 120:, :, :] = -1.0
    reference[:, :, :, 169:, :] = -1.0

    (output,) = nodes_seedvr.SeedVR2PostProcessing.execute(decoded, reference, "none")

    assert output.shape == (1, 120, 168, 3)


def test_seedvr2_postprocessing_lab_uses_unpadded_reference(monkeypatch):
    decoded = torch.ones((1, 128, 176, 3), dtype=torch.float32)
    reference = torch.zeros((1, 1, 128, 176, 3), dtype=torch.float32)
    reference[:, :, :120, :169, :] = 0.25
    reference[:, :, 120:, :, :] = -1.0
    reference[:, :, :, 169:, :] = -1.0
    calls = []

    def fake_lab_color_transfer(decoded_flat, reference_flat):
        calls.append((tuple(decoded_flat.shape), tuple(reference_flat.shape)))
        return decoded_flat

    monkeypatch.setattr(nodes_seedvr, "lab_color_transfer", fake_lab_color_transfer)

    (output,) = nodes_seedvr.SeedVR2PostProcessing.execute(decoded, reference, "lab")

    assert calls == [((1, 3, 120, 169), (1, 3, 120, 169))]
    assert output.shape == (1, 120, 168, 3)


def test_seedvr2_ambiguous_channel_last_decode_requires_explicit_flag():
    sample = torch.arange(1 * 16 * 4 * 5 * 16, dtype=torch.float32).reshape(1, 16, 4, 5, 16)
    vae = SimpleNamespace(latent_channels=16)

    channel_first = comfy.sd.VAE._normalize_seedvr2_decode_samples(vae, sample)
    channel_last = comfy.sd.VAE._normalize_seedvr2_decode_samples(vae, sample, channel_last=True)

    torch.testing.assert_close(channel_first, sample)
    torch.testing.assert_close(channel_last, sample.movedim(-1, 1))


def test_seedvr2_tiled_decode_node_forces_channel_last(monkeypatch):
    class FakeSeedVR2Wrapper:
        pass

    class FakeVAE:
        first_stage_model = FakeSeedVR2Wrapper()

        def __init__(self):
            self.decode_call = None

        def temporal_compression_decode(self):
            return 4

        def spacial_compression_decode(self):
            return 8

        def decode_tiled(self, samples, **kwargs):
            self.decode_call = kwargs
            return torch.zeros((1, 1, 2, 2, 3), dtype=torch.float32)

    monkeypatch.setattr(nodes.comfy.ldm.seedvr.vae, "VideoAutoencoderKLWrapper", FakeSeedVR2Wrapper)
    vae = FakeVAE()
    samples = {"samples": torch.zeros((1, 16, 4, 4, 16), dtype=torch.float32)}

    nodes.VAEDecodeTiled().decode(vae, samples, tile_size=64, overlap=0, temporal_size=64, temporal_overlap=8)

    assert vae.decode_call["seedvr2_channel_last"] is True
