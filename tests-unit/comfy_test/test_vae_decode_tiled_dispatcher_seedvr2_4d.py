from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import torch

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

import comfy.ldm.seedvr.vae as seedvr_vae_mod  # noqa: E402
import comfy.sd as sd_mod  # noqa: E402


def _force_oom(*a, **k):
    raise torch.cuda.OutOfMemoryError("forced OOM for dispatcher test")


def _make_vae(first_stage_model, latent_channels, latent_dim):
    vae = sd_mod.VAE.__new__(sd_mod.VAE)
    vae.first_stage_model = first_stage_model
    vae.patcher = MagicMock()
    vae.patcher.get_free_memory = MagicMock(return_value=8 * 1024 * 1024 * 1024)
    vae.device = vae.output_device = torch.device("cpu")
    vae.vae_dtype = torch.float32
    vae.disable_offload = True
    vae.extra_1d_channel = None
    vae.upscale_ratio = vae.downscale_ratio = 8
    vae.upscale_index_formula = vae.downscale_index_formula = None
    vae.output_channels = 3
    vae.latent_channels = latent_channels
    vae.latent_dim = latent_dim
    vae.vae_output_dtype = lambda: torch.float32
    vae.spacial_compression_decode = lambda: 8
    vae.process_input = lambda x: x
    vae.process_output = lambda x: x
    vae.throw_exception_if_invalid = lambda: None
    vae.memory_used_decode = lambda *a, **k: 1
    return vae


def _dispatch(vae, samples, seedvr2_call, generic_call, patch_wrapper_decode):
    mm = sd_mod.model_management
    with ExitStack() as stack:
        stack.enter_context(patch.object(mm, "raise_non_oom", lambda e: None))
        stack.enter_context(patch.object(mm, "load_models_gpu", lambda *a, **k: None))
        stack.enter_context(patch.object(mm, "soft_empty_cache", lambda: None))
        stack.enter_context(patch.object(sd_mod.VAE, "decode_tiled_seedvr2", seedvr2_call))
        stack.enter_context(patch.object(sd_mod.VAE, "decode_tiled_", generic_call))
        if patch_wrapper_decode:
            stack.enter_context(patch.object(
                seedvr_vae_mod.VideoAutoencoderKLWrapper, "decode",
                side_effect=_force_oom))
        vae.decode(samples)


def test_4d_seedvr2_latent_routes_to_decode_tiled_seedvr2():
    wrapper = seedvr_vae_mod.VideoAutoencoderKLWrapper.__new__(
        seedvr_vae_mod.VideoAutoencoderKLWrapper)
    vae = _make_vae(wrapper, latent_channels=16, latent_dim=3)
    seedvr2_call = MagicMock(return_value=torch.zeros(1, 3, 9, 64, 64))
    generic_call = MagicMock(return_value=torch.zeros(1, 3, 64, 64))
    _dispatch(vae, torch.zeros(1, 16 * 3, 8, 8), seedvr2_call, generic_call, True)
    assert seedvr2_call.call_count == 1
    assert generic_call.call_count == 0


def test_4d_non_seedvr2_latent_still_routes_to_generic_decode_tiled():
    first_stage = MagicMock()
    first_stage.decode = MagicMock(side_effect=_force_oom)
    vae = _make_vae(first_stage, latent_channels=4, latent_dim=2)
    seedvr2_call = MagicMock(return_value=torch.zeros(1, 3, 9, 64, 64))
    generic_call = MagicMock(return_value=torch.zeros(1, 3, 64, 64))
    _dispatch(vae, torch.zeros(1, 4, 8, 8), seedvr2_call, generic_call, False)
    assert generic_call.call_count == 1
    assert seedvr2_call.call_count == 0
