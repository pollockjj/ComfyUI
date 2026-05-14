"""Unit tests for ``VAE.encode_tiled_seedvr2``: existence with the
SeedVR2 tile-shape signature and delegation through
``comfy.ldm.seedvr.vae.tiled_vae(..., encode=True)`` with one call per
spatial tile.

Mirrors the decode-side method-existence + delegation contract for
``VAE.decode_tiled_seedvr2``; CPU-only via mocks and a
``VideoAutoencoderKLWrapper.__new__`` wrapper stub (no weights, no
GPU).
"""

import inspect
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

    vae.device = torch.device("cpu")
    vae.output_device = torch.device("cpu")
    vae.vae_dtype = torch.float32
    vae.latent_channels = 16
    vae.latent_dim = 3
    vae.downscale_ratio = 8

    vae.vae_output_dtype = lambda: torch.float32
    vae.process_input = lambda x: x
    return vae


def test_method_exists_with_seedvr2_signature():
    assert hasattr(sd_mod.VAE, "encode_tiled_seedvr2"), (
        "VAE.encode_tiled_seedvr2 must be defined on the VAE class."
    )
    sig = inspect.signature(sd_mod.VAE.encode_tiled_seedvr2)
    params = list(sig.parameters)
    for required in ("self", "pixel_samples", "tile_x", "tile_y",
                     "overlap", "tile_t", "overlap_t"):
        assert required in params, (
            f"VAE.encode_tiled_seedvr2 missing required parameter "
            f"{required!r}; got parameters {params}."
        )


def test_method_routes_through_tiled_vae_encode_true():
    vae = _make_minimal_seedvr2_vae()
    pixel_samples = torch.zeros((1, 3, 8, 64, 64))

    tiled_vae_mock = MagicMock(return_value=torch.zeros((1, 16, 2, 8, 8)))

    with patch.object(seedvr_vae_mod, "tiled_vae", tiled_vae_mock):
        vae.encode_tiled_seedvr2(pixel_samples)

    assert tiled_vae_mock.call_count >= 1, (
        f"Expected encode_tiled_seedvr2 to delegate to tiled_vae at "
        f"least once; got {tiled_vae_mock.call_count} calls."
    )
    for call in tiled_vae_mock.call_args_list:
        assert call.kwargs.get("encode") is True, (
            f"Every tiled_vae delegation from encode_tiled_seedvr2 must "
            f"pass encode=True; got kwargs={call.kwargs!r}."
        )
