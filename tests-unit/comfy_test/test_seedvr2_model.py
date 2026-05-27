"""Consolidated SeedVR2 model/graph/forward regression tests.

Merged from:
- seedvr_model_test.py
- test_seedvr_7b_final_block_text_path.py
- test_seedvr_forward_no_device_cast.py
- test_seedvr_latent_format.py
- test_seedvr2_vae_graph_boundaries.py
"""

from __future__ import annotations

import ast
import inspect
from unittest.mock import MagicMock

import torch
from torch import nn

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

import comfy  # noqa: E402
import comfy.latent_formats  # noqa: E402
import comfy.ldm.seedvr.model  # noqa: E402
import comfy.ldm.seedvr.model as seedvr_model  # noqa: E402
import comfy.ldm.seedvr.vae as seedvr_vae_mod  # noqa: E402
import comfy.model_management  # noqa: E402
import comfy.sample  # noqa: E402
import comfy.sd as sd_mod  # noqa: E402
import nodes as nodes_mod  # noqa: E402
from comfy.ldm.seedvr.model import MMModule, NaDiT  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers from seedvr_model_test.py
# ---------------------------------------------------------------------------


def _make_standin(positive_conditioning):
    class _StandIn(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer(
                "positive_conditioning", positive_conditioning
            )

        _resolve_text_conditioning = NaDiT._resolve_text_conditioning
        _swap_pos_neg_halves = NaDiT._swap_pos_neg_halves

    return _StandIn()


# ---------------------------------------------------------------------------
# Helpers from test_seedvr_7b_final_block_text_path.py
# ---------------------------------------------------------------------------


class _StubModule(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()


def _capture_last_layer_flags(monkeypatch, vid_dim: int, txt_in_dim: int) -> list[bool]:
    flags = []

    class _Block(_StubModule):
        def __init__(self, *args, **kwargs):
            flags.append(kwargs["is_last_layer"])
            super().__init__()

    monkeypatch.setattr(seedvr_model, "NaPatchIn", _StubModule)
    monkeypatch.setattr(seedvr_model, "NaPatchOut", _StubModule)
    monkeypatch.setattr(seedvr_model, "TimeEmbedding", _StubModule)
    monkeypatch.setattr(seedvr_model, "NaMMSRTransformerBlock", _Block)

    seedvr_model.NaDiT(
        norm_eps=1e-5,
        qk_rope=None,
        num_layers=4,
        mlp_type="normal",
        vid_dim=vid_dim,
        txt_in_dim=txt_in_dim,
        heads=24,
        mm_layers=3,
    )

    return flags


# ---------------------------------------------------------------------------
# Helpers from test_seedvr_latent_format.py
# ---------------------------------------------------------------------------


class _Model:
    def __init__(self, latent_format):
        self._latent_format = latent_format

    def get_model_object(self, name):
        assert name == "latent_format"
        return self._latent_format


# ---------------------------------------------------------------------------
# Helpers from test_seedvr2_vae_graph_boundaries.py
# ---------------------------------------------------------------------------


class _Patcher:
    def get_free_memory(self, device):
        return 1024 * 1024 * 1024


class _EncodeWrapper(seedvr_vae_mod.VideoAutoencoderKLWrapper):
    def __init__(self, encoded):
        nn.Module.__init__(self)
        self.encoded = encoded
        self.spatial_downsample_factor = 8
        self.temporal_downsample_factor = 4
        self.seen = []

    def encode(self, x):
        self.seen.append(tuple(x.shape))
        return self.encoded.to(device=x.device, dtype=x.dtype)


class _DecodeWrapper(seedvr_vae_mod.VideoAutoencoderKLWrapper):
    def __init__(self):
        nn.Module.__init__(self)
        self.spatial_downsample_factor = 8
        self.temporal_downsample_factor = 4
        self.calls = []

    def decode(self, z, seedvr2_tiling=None):
        self.calls.append({"shape": tuple(z.shape), "seedvr2_tiling": seedvr2_tiling})
        if z.ndim == 4:
            b, tc, h, w = z.shape
            t = tc // 16
        else:
            b, _, t, h, w = z.shape
        return torch.zeros(b, 3, t, h * 8, w * 8, dtype=z.dtype, device=z.device)


def _make_vae(wrapper):
    vae = sd_mod.VAE.__new__(sd_mod.VAE)
    vae.first_stage_model = wrapper
    vae.device = torch.device("cpu")
    vae.output_device = torch.device("cpu")
    vae.vae_dtype = torch.float32
    vae.latent_channels = 16
    vae.latent_dim = 3
    vae.downscale_ratio = (lambda a: max(0, (a + 3) // 4), 8, 8)
    vae.upscale_ratio = (lambda a: max(0, a * 4 - 3), 8, 8)
    vae.output_channels = 3
    vae.disable_offload = True
    vae.extra_1d_channel = None
    vae.crop_input = False
    vae.not_video = False
    vae.patcher = _Patcher()
    vae.process_input = lambda image: image
    vae.process_output = lambda image: image.add(1.0).div(2.0).clamp(0.0, 1.0)
    vae.vae_output_dtype = lambda: torch.float32
    vae.memory_used_encode = lambda shape, dtype: 1
    vae.memory_used_decode = lambda shape, dtype: 1
    vae.throw_exception_if_invalid = lambda: None
    vae.vae_encode_crop_pixels = lambda pixels: pixels
    vae.spacial_compression_decode = lambda: 8
    vae.temporal_compression_decode = lambda: 4
    return vae


# ---------------------------------------------------------------------------
# Tests from seedvr_model_test.py
# ---------------------------------------------------------------------------


def test_missing_context_falls_back_to_positive_buffer():
    """AC: ``context is None`` falls back to the registered
    ``positive_conditioning`` buffer and runs to completion — no
    silent zero substitution, no raised exception.
    """
    pos_buffer = torch.full((58, 5120), 7.0)
    standin = _make_standin(pos_buffer)
    txt, txt_shape = standin._resolve_text_conditioning(None)
    assert txt.shape == (58, 5120)
    assert (txt == 7.0).all(), (
        "fallback path must use the positive_conditioning buffer "
        "verbatim, not a zero tensor"
    )
    assert txt_shape.shape == (1, 1)
    assert txt_shape[0, 0].item() == 58


def test_output_side_swaps_pos_neg_halves():
    """AC complement: ``_swap_pos_neg_halves`` reorders the post-network
    output so the first half (positive) and second half (negative) trade
    places. For a 2-batch tensor with distinguishable halves, the
    returned tensor must be the swap — first half becomes negative,
    second half becomes positive — matching the original
    ``torch.cat([neg, pos])`` semantics from the pre-fix forward path.
    """
    pos_buffer = torch.zeros((58, 5120))
    standin = _make_standin(pos_buffer)
    pos_half = torch.full((1, 4, 8, 8), 1.0)
    neg_half = torch.full((1, 4, 8, 8), -1.0)
    out = torch.cat([pos_half, neg_half], dim=0)
    swapped = standin._swap_pos_neg_halves(out)
    assert swapped.shape == out.shape
    assert (swapped[0] == -1.0).all(), "first half of swapped output must be the original negative half"
    assert (swapped[1] == 1.0).all(), "second half of swapped output must be the original positive half"


# ---------------------------------------------------------------------------
# Tests from test_seedvr_7b_final_block_text_path.py
# ---------------------------------------------------------------------------


def test_seedvr2_7b_keeps_final_block_text_path(monkeypatch):
    assert _capture_last_layer_flags(monkeypatch, vid_dim=3072, txt_in_dim=3072) == [
        False,
        False,
        False,
        False,
    ]


def test_seedvr2_7b_rope3d_matches_wrapper_oracle():
    rope = seedvr_model.get_na_rope("rope3d", dim=64)
    generator = torch.Generator(device="cpu").manual_seed(0)
    q = torch.randn(4, 2, 128, generator=generator)
    k = torch.randn(4, 2, 128, generator=generator)
    shape = torch.tensor([[1, 2, 2]], dtype=torch.long)
    freqs = rope.get_axial_freqs(1, 2, 2).reshape(4, -1)

    expected_q = seedvr_model.apply_rotary_emb(
        freqs,
        q.permute(1, 0, 2).float(),
    ).to(q.dtype).permute(1, 0, 2)
    expected_k = seedvr_model.apply_rotary_emb(
        freqs,
        k.permute(1, 0, 2).float(),
    ).to(k.dtype).permute(1, 0, 2)

    actual_q, actual_k = rope(q.clone(), k.clone(), shape, seedvr_model.Cache(disable=True))

    torch.testing.assert_close(actual_q, expected_q, rtol=0, atol=0)
    torch.testing.assert_close(actual_k, expected_k, rtol=0, atol=0)


def test_adasingle_init_preserves_supported_dtype():
    ada = seedvr_model.AdaSingle(
        dim=4,
        emb_dim=24,
        layers=["test"],
        modes=["in", "out"],
        device="cpu",
        dtype=torch.bfloat16,
    )

    assert ada.test_shift.dtype is torch.bfloat16
    assert ada.test_scale.dtype is torch.bfloat16
    assert ada.test_gate.dtype is torch.bfloat16


# ---------------------------------------------------------------------------
# Tests from test_seedvr_forward_no_device_cast.py
# ---------------------------------------------------------------------------


def test_no_get_torch_device_in_forward_methods():
    tree = ast.parse(inspect.getsource(comfy.ldm.seedvr.model))
    assert [
        (n.lineno, i.lineno)
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "forward"
        for i in ast.walk(n)
        if isinstance(i, ast.Call)
        and isinstance(i.func, ast.Attribute)
        and i.func.attr == "get_torch_device"
    ] == []


def test_mmmodule_forward_succeeds_without_get_torch_device_lookup(monkeypatch):
    call_count = [0]

    def boom():
        call_count[0] += 1
        raise RuntimeError("MMModule.forward called get_torch_device()")

    monkeypatch.setattr(comfy.model_management, "get_torch_device", boom)

    class _IdentityCallable(nn.Module):
        def forward(self, x, *args, **kwargs):
            return x

    mm = MMModule(_IdentityCallable, shared_weights=False, vid_only=False)

    vid_in = torch.zeros(2, 4)
    txt_in = torch.ones(2, 4)
    vid_out, txt_out = mm.forward(vid_in, txt_in)

    assert call_count[0] == 0
    assert torch.equal(vid_out, vid_in)
    assert torch.equal(txt_out, txt_in)
    assert vid_out.device == vid_in.device
    assert txt_out.device == txt_in.device


# ---------------------------------------------------------------------------
# Tests from test_seedvr_latent_format.py
# ---------------------------------------------------------------------------


def test_seedvr2_latent_format_uses_16_channels_without_3d_empty_latent_expansion():
    latent_format = comfy.latent_formats.SeedVR2()
    latent_image = torch.zeros(1, 1, 4, 5)

    fixed = comfy.sample.fix_empty_latent_channels(_Model(latent_format), latent_image)

    assert latent_format.latent_channels == 16
    assert latent_format.latent_dimensions == 2
    assert fixed.shape == (1, 16, 4, 5)


def test_seedvr2_empty_collapsed_latent_preserves_temporal_channel_multiples():
    latent_format = comfy.latent_formats.SeedVR2()
    latent_image = torch.zeros(1, 48, 4, 5)

    fixed = comfy.sample.fix_empty_latent_channels(_Model(latent_format), latent_image)

    assert latent_format.preserve_empty_channel_multiples is True
    assert fixed.shape == latent_image.shape
    assert fixed.data_ptr() == latent_image.data_ptr()


# ---------------------------------------------------------------------------
# Tests from test_seedvr2_vae_graph_boundaries.py
# ---------------------------------------------------------------------------


def test_seedvr2_encode_and_encode_tiled_preserve_native_latent_contract(monkeypatch):
    monkeypatch.setattr(sd_mod.model_management, "load_models_gpu", lambda *a, **k: None)

    encoded = torch.full((1, 16, 2, 4, 5), 2.0)
    vae = _make_vae(_EncodeWrapper(encoded))
    pixels = torch.zeros(1, 5, 32, 40, 3)

    node_output = nodes_mod.VAEEncode().encode(vae, pixels)[0]
    node_latent = node_output["samples"]
    assert set(node_output) == {"samples"}
    assert tuple(node_latent.shape) == (1, 16, 2, 4, 5)
    assert node_latent.dtype == torch.float32
    assert node_latent.stride()[-1] == 1
    assert torch.equal(node_latent, torch.full_like(node_latent, 2.0 * 0.9152))

    tiled = torch.full((1, 16, 2, 4, 5), 3.0)
    monkeypatch.setattr(seedvr_vae_mod, "tiled_vae", MagicMock(return_value=tiled))
    tiled_output = nodes_mod.VAEEncodeTiled().encode(
        vae,
        pixels,
        tile_size=512,
        overlap=64,
        temporal_size=16,
        temporal_overlap=4,
    )[0]
    tiled_latent = tiled_output["samples"]
    assert set(tiled_output) == {"samples"}
    assert tuple(tiled_latent.shape) == (1, 16, 2, 4, 5)
    assert tiled_latent.dtype == torch.float32
    assert torch.equal(tiled_latent, torch.full_like(tiled_latent, 3.0 * 0.9152))


def test_vaedecode_tiled_visible_inputs_are_seedvr2_decode_tiling_authority(monkeypatch):
    monkeypatch.setattr(sd_mod.model_management, "load_models_gpu", lambda *a, **k: None)
    vae = _make_vae(_DecodeWrapper())

    nodes_mod.VAEDecodeTiled().decode(
        vae,
        {"samples": torch.zeros(1, 16, 2, 4, 5)},
        tile_size=512,
        overlap=64,
        temporal_size=16,
        temporal_overlap=4,
    )

    assert vae.first_stage_model.calls == [
        {
            "shape": (1, 16, 2, 4, 5),
            "seedvr2_tiling": {
                "enable_tiling": True,
                "tile_size": (512, 512),
                "tile_overlap": (64, 64),
                "temporal_size": 16,
                "temporal_overlap": 4,
            },
        }
    ]
