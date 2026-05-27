from unittest.mock import MagicMock, patch

import torch

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

import comfy.ldm.seedvr.vae as seedvr_vae_mod  # noqa: E402
import comfy.sd as sd_mod  # noqa: E402


def _make_minimal_seedvr2_vae():
    vae = sd_mod.VAE.__new__(sd_mod.VAE)
    wrapper = seedvr_vae_mod.VideoAutoencoderKLWrapper.__new__(
        seedvr_vae_mod.VideoAutoencoderKLWrapper
    )
    vae.first_stage_model = wrapper
    vae.patcher = MagicMock()
    vae.patcher.get_free_memory = MagicMock(return_value=8 * 1024 * 1024 * 1024)
    vae.device = torch.device("cpu")
    vae.output_device = torch.device("cpu")
    vae.vae_dtype = torch.float32
    vae.disable_offload = True
    vae.extra_1d_channel = None
    vae.upscale_ratio = 8
    vae.upscale_index_formula = None
    vae.output_channels = 3
    vae.latent_channels = 16
    vae.latent_dim = 3
    vae.downscale_ratio = 8
    vae.downscale_index_formula = None
    vae.vae_output_dtype = lambda: torch.float32
    vae.spacial_compression_decode = lambda: 8
    vae.process_input = lambda x: x
    vae.process_output = lambda x: x
    vae.throw_exception_if_invalid = lambda: None
    vae.memory_used_decode = lambda *a, **k: 1
    return vae


def _force_regular_decode_oom(*args, **kwargs):
    raise torch.cuda.OutOfMemoryError("forced OOM for dispatcher test")


def test_4d_seedvr2_latent_routes_to_decode_tiled_seedvr2():
    vae = _make_minimal_seedvr2_vae()
    samples_4d = torch.zeros(1, 16 * 3, 8, 8)

    seedvr2_call = MagicMock(return_value=torch.zeros(1, 3, 9, 64, 64))
    generic_call = MagicMock(return_value=torch.zeros(1, 3, 64, 64))

    with patch.object(sd_mod.model_management, "raise_non_oom",
                      lambda e: None), \
         patch.object(sd_mod.model_management, "load_models_gpu",
                      lambda *a, **k: None), \
         patch.object(sd_mod.model_management, "soft_empty_cache",
                      lambda: None), \
         patch.object(seedvr_vae_mod.VideoAutoencoderKLWrapper, "decode",
                      side_effect=_force_regular_decode_oom), \
         patch.object(sd_mod.VAE, "decode_tiled_seedvr2", seedvr2_call), \
         patch.object(sd_mod.VAE, "decode_tiled_", generic_call):
        vae.decode(samples_4d)

    assert seedvr2_call.call_count == 1
    assert generic_call.call_count == 0


def test_4d_non_seedvr2_latent_still_routes_to_generic_decode_tiled():
    vae = sd_mod.VAE.__new__(sd_mod.VAE)
    vae.first_stage_model = MagicMock()
    vae.patcher = MagicMock()
    vae.patcher.get_free_memory = MagicMock(return_value=8 * 1024 * 1024 * 1024)
    vae.device = torch.device("cpu")
    vae.output_device = torch.device("cpu")
    vae.vae_dtype = torch.float32
    vae.disable_offload = True
    vae.extra_1d_channel = None
    vae.upscale_ratio = 8
    vae.upscale_index_formula = None
    vae.output_channels = 3
    vae.latent_channels = 4
    vae.latent_dim = 2
    vae.downscale_ratio = 8
    vae.downscale_index_formula = None
    vae.vae_output_dtype = lambda: torch.float32
    vae.spacial_compression_decode = lambda: 8
    vae.process_output = lambda x: x
    vae.throw_exception_if_invalid = lambda: None
    vae.memory_used_decode = lambda *a, **k: 1
    vae.first_stage_model.decode = MagicMock(
        side_effect=_force_regular_decode_oom
    )

    samples_4d = torch.zeros(1, 4, 8, 8)
    generic_call = MagicMock(return_value=torch.zeros(1, 3, 64, 64))
    seedvr2_call = MagicMock(return_value=torch.zeros(1, 3, 9, 64, 64))

    with patch.object(sd_mod.model_management, "raise_non_oom",
                      lambda e: None), \
         patch.object(sd_mod.model_management, "load_models_gpu",
                      lambda *a, **k: None), \
         patch.object(sd_mod.model_management, "soft_empty_cache",
                      lambda: None), \
         patch.object(sd_mod.VAE, "decode_tiled_seedvr2", seedvr2_call), \
         patch.object(sd_mod.VAE, "decode_tiled_", generic_call):
        vae.decode(samples_4d)

    assert generic_call.call_count == 1
    assert seedvr2_call.call_count == 0
