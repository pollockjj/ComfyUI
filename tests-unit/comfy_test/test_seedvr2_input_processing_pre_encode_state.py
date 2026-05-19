"""Unit tests for SeedVR2InputProcessing(enable_tiling=False) pre-encode state-order.

The `enable_tiling=False` branch of ``SeedVR2InputProcessing.execute`` must
populate ``vae_model.img_dims``, ``vae_model.original_image_video``, and
``vae_model.tiled_args`` on the wrapper BEFORE calling ``vae.encode(...)``,
so that a SeedVR2-aware encode tiled fallback can consult the wrapper's
``tiled_args`` (including ``enable_tiling: False``) when triggered.

These tests stub ``vae.encode`` with a ``MagicMock`` whose ``side_effect``
captures the three attributes AT THE MOMENT ``vae.encode`` is invoked.
"""

from unittest.mock import MagicMock, patch

import torch

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

import comfy_extras.nodes_seedvr as nodes_seedvr  # noqa: E402


_MISSING = object()


def _make_capture_fixture(temporal_tile_size=16, temporal_overlap=4):
    """Build the SeedVR2InputProcessing.execute call surface with a vae.encode
    stub whose side_effect captures vae_model state at invocation time.

    Returns (call_execute, captured) where:
      - call_execute() drives SeedVR2InputProcessing.execute(enable_tiling=False, ...)
      - captured is a dict populated by the side_effect at vae.encode call time
    """
    captured = {}
    vae_model = MagicMock()
    vae = MagicMock()
    vae.first_stage_model = vae_model
    vae.patcher = MagicMock()

    def _capture_and_return(*args, **kwargs):
        for attr in ("img_dims", "original_image_video", "tiled_args"):
            captured[attr] = getattr(vae_model, attr, _MISSING)
        # Return a minimally shaped latent so the rest of execute does not crash.
        # Expected post-rearrange shape: (b, c, t, h, w) -> unsqueeze if 4D -> rearrange to (b ... c).
        return torch.zeros(1, 16, 1, 4, 4)

    vae.encode = MagicMock(side_effect=_capture_and_return)

    # Minimal images tensor (B, T, H, W, C) in float for the BTHWC pipeline.
    images = torch.zeros(1, 1, 16, 16, 3)

    def call_execute():
        with patch.object(nodes_seedvr.comfy.model_management,
                          "load_models_gpu", lambda *a, **k: None), \
             patch.object(nodes_seedvr, "clear_vae_memory",
                          lambda *a, **k: None):
            return nodes_seedvr.SeedVR2InputProcessing.execute(
                images=images,
                vae=vae,
                resolution=120,
                spatial_tile_size=512,
                spatial_overlap=64,
                temporal_tile_size=temporal_tile_size,
                temporal_overlap=temporal_overlap,
                enable_tiling=False,
            )

    return call_execute, captured


def test_pre_encode_state_set_before_vae_encode():
    """img_dims, original_image_video, tiled_args must all be populated on
    vae_model at the moment vae.encode(images_bthwc) is invoked."""
    call_execute, captured = _make_capture_fixture()
    call_execute()

    assert captured.get("img_dims", _MISSING) is not _MISSING, (
        "vae_model.img_dims must be set before vae.encode is called; "
        "captured at the moment of vae.encode invocation: _MISSING."
    )
    assert captured.get("original_image_video", _MISSING) is not _MISSING, (
        "vae_model.original_image_video must be set before vae.encode is called; "
        "captured at the moment of vae.encode invocation: _MISSING."
    )
    assert captured.get("tiled_args", _MISSING) is not _MISSING, (
        "vae_model.tiled_args must be set before vae.encode is called so a "
        "SeedVR2-aware encode tiled fallback can consult wrapper state; "
        "captured at the moment of vae.encode invocation: _MISSING."
    )


def test_pre_encode_tiled_args_contains_enable_tiling_false():
    """Captured tiled_args at vae.encode invocation time must be a dict with
    enable_tiling explicitly set to False."""
    call_execute, captured = _make_capture_fixture()
    call_execute()

    tiled_args = captured.get("tiled_args", _MISSING)
    assert isinstance(tiled_args, dict), (
        f"vae_model.tiled_args at vae.encode invocation must be a dict; got "
        f"{type(tiled_args).__name__}: {tiled_args!r}."
    )
    assert tiled_args.get("enable_tiling") is False, (
        f"vae_model.tiled_args['enable_tiling'] at vae.encode invocation must "
        f"be False (the enable_tiling=False branch); got "
        f"{tiled_args.get('enable_tiling')!r}."
    )


def test_pre_encode_tiled_args_preserve_zero_temporal_bkm():
    """A configured 0/0 temporal BKM must reach the wrapper state unchanged."""
    call_execute, captured = _make_capture_fixture(
        temporal_tile_size=0,
        temporal_overlap=0,
    )
    call_execute()

    tiled_args = captured.get("tiled_args", _MISSING)
    assert tiled_args["temporal_size"] == 0
    assert tiled_args["temporal_overlap"] == 0
