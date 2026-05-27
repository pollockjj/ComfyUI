import inspect
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

import comfy.ldm.seedvr.vae as vae_mod  # noqa: E402
import comfy.ldm.seedvr.vae as seedvr_vae_mod  # noqa: E402
import comfy.sd as sd_mod  # noqa: E402
import nodes as nodes_mod  # noqa: E402
from comfy.ldm.seedvr.vae import MemoryState, VideoAutoencoderKL, tiled_vae  # noqa: E402


# ---------------------------------------------------------------------------
# From test_seedvr_vae_tiled_decode_latent_min_size_override.py
# ---------------------------------------------------------------------------


def test_runtime_decode_zero_temporal_size_disables_slicing_for_call():
    from comfy.ldm.seedvr.vae import MemoryState, VideoAutoencoderKL, tiled_vae

    class StubVAEModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.slicing_latent_min_size = 2
            self.spatial_downsample_factor = 8
            self.temporal_downsample_factor = 4
            self.device = torch.device("cpu")
            self.use_slicing = True
            self._dummy = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))
            self.decode_min_sizes = []
            self.memory_states = []

        def decode_(self, t_chunk):
            self.decode_min_sizes.append(self.slicing_latent_min_size)
            return VideoAutoencoderKL.slicing_decode(self, t_chunk)

        def _decode(self, z, memory_state=MemoryState.DISABLED):
            self.memory_states.append(memory_state)
            b, c, d, h, w = z.shape
            return torch.zeros((b, 3, d, h * 8, w * 8), dtype=z.dtype)

    vae = StubVAEModel()
    z = torch.zeros((1, 16, 5, 8, 8), dtype=torch.float32)

    tiled_vae(
        z,
        vae,
        tile_size=(64, 64),
        tile_overlap=(0, 0),
        temporal_size=0,
        temporal_overlap=0,
        encode=False,
    )

    assert vae.decode_min_sizes == [5]
    assert vae.memory_states == [MemoryState.DISABLED]
    assert vae.slicing_latent_min_size == 2


# ---------------------------------------------------------------------------
# From test_seedvr_vae_tiled_encode_runt_slice_override.py
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# From test_seedvr_vae_tiled_temporal_slicing.py
# ---------------------------------------------------------------------------


class _SlicingDecodeVAE(nn.Module):
    def __init__(self, slicing_latent_min_size):
        super().__init__()
        self.slicing_latent_min_size = slicing_latent_min_size
        self.spatial_downsample_factor = 8
        self.temporal_downsample_factor = 4
        self.device = torch.device("cpu")
        self.use_slicing = True
        self._dummy = nn.Parameter(torch.zeros(1, dtype=torch.float32))
        self.decode_min_sizes = []
        self.memory_states = []

    def decode_(self, z):
        self.decode_min_sizes.append(self.slicing_latent_min_size)
        return vae_mod.VideoAutoencoderKL.slicing_decode(self, z)

    def _decode(self, z, memory_state=MemoryState.DISABLED):
        self.memory_states.append(memory_state)
        x = z[:, :1].repeat(
            1,
            3,
            1,
            self.spatial_downsample_factor,
            self.spatial_downsample_factor,
        )
        return x


class _EncodeVAE(nn.Module):
    def __init__(self, slicing_sample_min_size):
        super().__init__()
        self.slicing_sample_min_size = slicing_sample_min_size
        self.spatial_downsample_factor = 8
        self.temporal_downsample_factor = 4
        self.device = torch.device("cpu")
        self.use_slicing = True
        self._dummy = nn.Parameter(torch.zeros(1, dtype=torch.float32))
        self.memory_states = []
        self.encoded_t = []
        self.encode_min_sizes = []

    def encode(self, t_chunk):
        self.encode_min_sizes.append(self.slicing_sample_min_size)
        h = vae_mod.VideoAutoencoderKL.slicing_encode(self, t_chunk)
        return (h, h)

    def _encode(self, x, memory_state=MemoryState.DISABLED):
        self.memory_states.append(memory_state)
        self.encoded_t.append(x.shape[2])
        b, c, t_in, h, w = x.shape
        target_d = max(1, (t_in + self.temporal_downsample_factor - 1) // self.temporal_downsample_factor)
        target_h = (h + self.spatial_downsample_factor - 1) // self.spatial_downsample_factor
        target_w = (w + self.spatial_downsample_factor - 1) // self.spatial_downsample_factor
        z = torch.zeros((b, 16, target_d, target_h, target_w), dtype=x.dtype)
        return z


def test_decode_tiled_vae_maps_temporal_args_to_latent_slicing_min_size():
    vae = _SlicingDecodeVAE(slicing_latent_min_size=2)
    z = torch.arange(1 * 16 * 5 * 8 * 8, dtype=torch.float32).reshape(1, 16, 5, 8, 8)

    tiled_vae(
        z,
        vae,
        tile_size=(64, 64),
        tile_overlap=(0, 0),
        temporal_size=12,
        temporal_overlap=4,
        encode=False,
    )

    assert vae.decode_min_sizes == [2]
    assert vae.memory_states == [MemoryState.INITIALIZING, MemoryState.ACTIVE]
    assert vae.slicing_latent_min_size == 2

    wrapper = vae_mod.VideoAutoencoderKLWrapper.__new__(
        vae_mod.VideoAutoencoderKLWrapper
    )
    nn.Module.__init__(wrapper)
    seedvr2_tiling = {
        "enable_tiling": True,
        "tile_size": (64, 64),
        "tile_overlap": (0, 0),
        "temporal_size": 8,
        "temporal_overlap": 7,
    }

    captured = {}

    def _fake_tiled_vae(latent, model, **kwargs):
        captured.update(kwargs)
        return torch.zeros(1, 3, 1, 16, 16)

    with (
        patch.object(vae_mod, "tiled_vae", side_effect=_fake_tiled_vae),
        patch.object(vae_mod, "lab_color_transfer", side_effect=lambda content, style: content),
    ):
        wrapper.decode(torch.zeros(1, 16, 2, 2), seedvr2_tiling=seedvr2_tiling)

    assert captured["temporal_overlap"] == 7


def test_encode_tiled_vae_zero_temporal_size_disables_wrapper_slicing():
    vae = _EncodeVAE(slicing_sample_min_size=4)
    x = torch.zeros((1, 3, 12, 64, 64), dtype=torch.float32)

    tiled_vae(
        x,
        vae,
        tile_size=(64, 64),
        tile_overlap=(0, 0),
        temporal_size=0,
        temporal_overlap=0,
        encode=True,
    )

    assert vae.encode_min_sizes == [12]
    assert vae.memory_states == [MemoryState.DISABLED]
    assert vae.encoded_t == [12]
    assert vae.slicing_sample_min_size == 4


# ---------------------------------------------------------------------------
# From test_seedvr_vae_5d_tiled_decode.py
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# From test_vae_decode_tiled_dispatcher_seedvr2_4d.py
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# From test_vae_encode_tiled_explicit_dispatcher_seedvr2.py
# ---------------------------------------------------------------------------


def _populate_common_vae_attrs_explicit(vae):
    vae.patcher = MagicMock()
    vae.patcher.get_free_memory = MagicMock(return_value=8 * 1024 * 1024 * 1024)
    vae.device = torch.device("cpu")
    vae.output_device = torch.device("cpu")
    vae.vae_dtype = torch.float32
    vae.disable_offload = True
    vae.extra_1d_channel = None
    vae.upscale_ratio = [lambda x: x]
    vae.upscale_index_formula = None
    vae.output_channels = 3
    vae.latent_channels = 16
    vae.latent_dim = 3
    vae.downscale_ratio = [lambda x: x]
    vae.downscale_index_formula = None
    vae.not_video = False
    vae.crop_input = False
    vae.pad_channel_value = None
    vae.vae_output_dtype = lambda: torch.float32
    vae.spacial_compression_encode = lambda: 8
    vae.vae_encode_crop_pixels = lambda x: x
    vae.throw_exception_if_invalid = lambda: None
    vae.memory_used_encode = lambda *a, **k: 1


def _make_seedvr2_vae_explicit():
    vae = sd_mod.VAE.__new__(sd_mod.VAE)
    wrapper = seedvr_vae_mod.VideoAutoencoderKLWrapper.__new__(
        seedvr_vae_mod.VideoAutoencoderKLWrapper
    )
    vae.first_stage_model = wrapper
    _populate_common_vae_attrs_explicit(vae)
    return vae


def _make_non_seedvr2_vae_explicit():
    vae = sd_mod.VAE.__new__(sd_mod.VAE)
    vae.first_stage_model = MagicMock()
    _populate_common_vae_attrs_explicit(vae)
    return vae


def test_explicit_encode_tiled_seedvr2_3d_routes_to_seedvr2_tiler():
    vae = _make_seedvr2_vae_explicit()
    pixel_samples = torch.zeros((1, 64, 64, 3))
    seedvr2_call = MagicMock(return_value=torch.zeros(1, 16, 2, 8, 8))
    generic_call = MagicMock(return_value=torch.zeros(1, 16, 2, 8, 8))

    with patch.object(sd_mod.model_management, "load_models_gpu",
                      lambda *a, **k: None), \
         patch.object(sd_mod.VAE, "encode_tiled_seedvr2", seedvr2_call,
                      create=True), \
         patch.object(sd_mod.VAE, "encode_tiled_3d", generic_call):
        vae.encode_tiled(pixel_samples)

    assert seedvr2_call.call_count == 1
    assert generic_call.call_count == 0


def test_explicit_encode_tiled_dispatcher_breakdown():
    seedvr2_call = MagicMock(return_value=torch.zeros(1, 16, 2, 8, 8))
    generic_call = MagicMock(return_value=torch.zeros(1, 16, 2, 8, 8))
    seedvr2_vae = _make_seedvr2_vae_explicit()
    non_seedvr2_vae = _make_non_seedvr2_vae_explicit()
    pixel_samples = torch.zeros((1, 64, 64, 3))

    with patch.object(sd_mod.model_management, "load_models_gpu",
                      lambda *a, **k: None), \
         patch.object(sd_mod.VAE, "encode_tiled_seedvr2", seedvr2_call,
                      create=True), \
         patch.object(sd_mod.VAE, "encode_tiled_3d", generic_call):
        seedvr2_vae.encode_tiled(pixel_samples)
        non_seedvr2_vae.encode_tiled(pixel_samples)

    assert seedvr2_call.call_count == 1
    assert generic_call.call_count == 1


# ---------------------------------------------------------------------------
# From test_vae_encode_tiled_fallback_dispatcher_seedvr2.py
# ---------------------------------------------------------------------------


def _populate_common_vae_attrs_fallback(vae):
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
    vae.not_video = False
    vae.crop_input = False
    vae.pad_channel_value = None

    vae.vae_output_dtype = lambda: torch.float32
    vae.spacial_compression_encode = lambda: 8
    vae.process_input = lambda x: x
    vae.process_output = lambda x: x
    vae.throw_exception_if_invalid = lambda: None
    vae.memory_used_encode = lambda *a, **k: 1


def _make_seedvr2_vae_fallback():
    vae = sd_mod.VAE.__new__(sd_mod.VAE)
    wrapper = seedvr_vae_mod.VideoAutoencoderKLWrapper.__new__(
        seedvr_vae_mod.VideoAutoencoderKLWrapper
    )
    vae.first_stage_model = wrapper
    _populate_common_vae_attrs_fallback(vae)
    return vae


def _make_non_seedvr2_vae_fallback():
    vae = sd_mod.VAE.__new__(sd_mod.VAE)
    vae.first_stage_model = MagicMock()
    _populate_common_vae_attrs_fallback(vae)
    return vae


def _force_regular_encode_oom(*args, **kwargs):
    raise torch.cuda.OutOfMemoryError("forced OOM for dispatcher test")


def test_seedvr2_3d_routes_to_encode_tiled_seedvr2_on_oom():
    vae = _make_seedvr2_vae_fallback()
    pixel_samples = torch.zeros((1, 8, 64, 64, 3))

    seedvr2_call = MagicMock(return_value=torch.zeros(1, 16, 2, 8, 8))
    generic_call = MagicMock(return_value=torch.zeros(1, 16, 2, 8, 8))

    with patch.object(sd_mod.model_management, "raise_non_oom",
                      lambda e: None), \
         patch.object(sd_mod.model_management, "load_models_gpu",
                      lambda *a, **k: None), \
         patch.object(sd_mod.model_management, "soft_empty_cache",
                      lambda: None), \
         patch.object(seedvr_vae_mod.VideoAutoencoderKLWrapper, "encode",
                      side_effect=_force_regular_encode_oom), \
         patch.object(sd_mod.VAE, "encode_tiled_seedvr2", seedvr2_call,
                      create=True), \
         patch.object(sd_mod.VAE, "encode_tiled_3d", generic_call):
        vae.encode(pixel_samples)

    assert seedvr2_call.call_count == 1, (
        f"Expected encode_tiled_seedvr2 to be called once for a SeedVR2 3D "
        f"input under OOM fallback; got {seedvr2_call.call_count} calls."
    )
    assert generic_call.call_count == 0, (
        f"encode_tiled_3d must NOT be called for a SeedVR2 input; got "
        f"{generic_call.call_count} calls."
    )


def test_non_seedvr2_encode_tiled_3d_default_overlap_is_concrete():
    vae = _make_non_seedvr2_vae_fallback()
    vae.downscale_ratio = (lambda a: max(1, a // 4), 8, 8)
    vae.upscale_ratio = (lambda a: a * 4, 8, 8)
    generic_call = MagicMock(return_value=torch.zeros(1, 16, 2, 8, 8))
    pixel_samples = torch.zeros((1, 8, 64, 64, 3))

    with patch.object(sd_mod.model_management, "load_models_gpu",
                      lambda *a, **k: None), \
         patch.object(sd_mod.VAE, "encode_tiled_3d", generic_call):
        vae.encode_tiled(pixel_samples)

    assert generic_call.call_args.kwargs["overlap"] == (1, 64, 64)


# ---------------------------------------------------------------------------
# From test_vae_encode_tiled_seedvr2_method.py
# ---------------------------------------------------------------------------


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


def test_vae_encode_tiled_allows_zero_temporal_controls_and_passes_zero_through():
    input_types = nodes_mod.VAEEncodeTiled.INPUT_TYPES()["required"]
    assert input_types["temporal_size"][1]["min"] == 0
    assert input_types["temporal_overlap"][1]["min"] == 0
    assert "SeedVR2 allows 0" in input_types["temporal_size"][1]["tooltip"]

    class _EncodeRecorder:
        def __init__(self):
            self.calls = []

        def encode_tiled(self, pixels, **kwargs):
            self.calls.append({"shape": tuple(pixels.shape), **kwargs})
            return torch.zeros(1, 16, 1, 8, 8)

    recorder = _EncodeRecorder()
    node = nodes_mod.VAEEncodeTiled()

    output = node.encode(
        recorder,
        torch.zeros(1, 64, 64, 3),
        tile_size=256,
        overlap=64,
        temporal_size=0,
        temporal_overlap=8,
    )

    assert recorder.calls == [
        {
            "shape": (1, 64, 64, 3),
            "tile_x": 256,
            "tile_y": 256,
            "overlap": 64,
            "tile_t": 0,
            "overlap_t": 0,
        }
    ]
    assert torch.equal(output[0]["samples"], torch.zeros(1, 16, 1, 8, 8))


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
