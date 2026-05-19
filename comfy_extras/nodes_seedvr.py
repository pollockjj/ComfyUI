from typing_extensions import override
from comfy_api.latest import ComfyExtension, io
import torch
import math
from einops import rearrange

import gc
import logging
import time
import comfy.model_management
import comfy.sample
import comfy.samplers
from comfy.ldm.seedvr.vae import tiled_vae

import torch.nn.functional as F
from torchvision.transforms import functional as TVF
from torchvision.transforms import Lambda, Normalize
from torchvision.transforms.functional import InterpolationMode


_SEEDVR2_INVALID_MODEL_MSG_PREFIX = (
    "SeedVR2Conditioning: model object does not match expected SeedVR2 structure"
)

# Private sentinel for getattr default: distinguishes "attribute missing"
# from "attribute present but None" so the failure message is accurate.
_ATTR_MISSING = object()


def _resolve_seedvr2_diffusion_model(model):
    """Resolve the inner SeedVR2 diffusion-model module from a ComfyUI model
    patcher object. Fails loud with a ``RuntimeError`` whose message begins
    with ``_SEEDVR2_INVALID_MODEL_MSG_PREFIX`` when the expected wrapper
    shape (``model.model.diffusion_model``) is absent.

    Distinguishes four failure modes via the ``_ATTR_MISSING`` sentinel:
    ``model.model`` missing, ``model.model is None``,
    ``model.model.diffusion_model`` missing, ``model.model.diffusion_model
    is None``. Each mode produces an accurate error message rather than
    conflating "attribute missing" with "attribute is None".
    """
    inner = getattr(model, "model", _ATTR_MISSING)
    if inner is _ATTR_MISSING:
        raise RuntimeError(
            f"{_SEEDVR2_INVALID_MODEL_MSG_PREFIX}: input has no 'model' attribute "
            f"(got type {type(model).__name__})."
        )
    if inner is None:
        raise RuntimeError(
            f"{_SEEDVR2_INVALID_MODEL_MSG_PREFIX}: input.model is None "
            f"(input type {type(model).__name__})."
        )
    diffusion_model = getattr(inner, "diffusion_model", _ATTR_MISSING)
    if diffusion_model is _ATTR_MISSING:
        raise RuntimeError(
            f"{_SEEDVR2_INVALID_MODEL_MSG_PREFIX}: 'model.model' has no "
            f"'diffusion_model' attribute (got type {type(inner).__name__})."
        )
    if diffusion_model is None:
        raise RuntimeError(
            f"{_SEEDVR2_INVALID_MODEL_MSG_PREFIX}: 'model.model.diffusion_model' "
            f"is None (model.model type {type(inner).__name__})."
        )
    return diffusion_model


def _describe_seedvr2_model_source(model_patcher) -> str:
    """Best-effort extraction of the source ``.safetensors`` path for a
    SeedVR2 model patcher. ``comfy.sd.load_diffusion_model`` stores
    ``cached_patcher_init = (function, (path, ...))`` on the returned
    patcher; surface that path in the fail-loud message when the
    conditioning buffers are unpopulated. Returns an empty string when
    the path is unavailable so the caller can choose a fallback message.
    """
    cached = getattr(model_patcher, "cached_patcher_init", None)
    if cached is None:
        return ""
    try:
        args = cached[1]
        for arg in args:
            if isinstance(arg, str) and arg.endswith(".safetensors"):
                return arg
    except (TypeError, IndexError):
        return ""
    return ""


def _apply_rope_freqs_float32_cast(diffusion_model):
    """Cast every nested module's ``rope.freqs`` parameter data to ``float32``
    when it is not already in float32. Idempotency is per-tensor by dtype
    check, NOT a per-instance sentinel attribute — a sentinel would survive
    Comfy's dynamic model unload/reload cycle while ``rope.freqs`` itself
    is restored from the archived dtype, leaving RoPE running in fp16/bf16
    on subsequent calls. The dtype check makes the cast self-correcting
    against weight-restore lifecycle events. Iteration cost is one walk of
    the diffusion-model module tree per ``execute()`` call (microseconds).
    """
    for module in diffusion_model.modules():
        if hasattr(module, 'rope') and hasattr(module.rope, 'freqs'):
            if module.rope.freqs.data.dtype != torch.float32:
                module.rope.freqs.data = module.rope.freqs.data.to(torch.float32)


def clear_vae_memory(vae_model):
    for module in vae_model.modules():
        if hasattr(module, "memory"):
            module.memory = None
    gc.collect()
    comfy.model_management.soft_empty_cache()

def expand_dims(tensor, ndim):
    shape = tensor.shape + (1,) * (ndim - tensor.ndim)
    return tensor.reshape(shape)

def get_conditions(latent, latent_blur):
    t, h, w, c = latent.shape
    cond = torch.ones([t, h, w, c + 1], device=latent.device, dtype=latent.dtype)
    cond[:, ..., :-1] = latent_blur[:]
    cond[:, ..., -1:] = 1.0
    return cond

def timestep_transform(timesteps, latents_shapes):
    vt = 4
    vs = 8
    frames = (latents_shapes[:, 0] - 1) * vt + 1
    heights = latents_shapes[:, 1] * vs
    widths = latents_shapes[:, 2] * vs

    # Compute shift factor.
    def get_lin_function(x1, y1, x2, y2):
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        return lambda x: m * x + b

    img_shift_fn = get_lin_function(x1=256 * 256, y1=1.0, x2=1024 * 1024, y2=3.2)
    vid_shift_fn = get_lin_function(x1=256 * 256 * 37, y1=1.0, x2=1280 * 720 * 145, y2=5.0)
    shift = torch.where(
        frames > 1,
        vid_shift_fn(heights * widths * frames),
        img_shift_fn(heights * widths),
    ).to(timesteps.device)

    # Shift timesteps.
    T = 1000.0
    timesteps = timesteps / T
    timesteps = shift * timesteps / (1 + (shift - 1) * timesteps)
    timesteps = timesteps * T
    return timesteps

def inter(x_0, x_T, t):
    t = expand_dims(t, x_0.ndim)
    T = 1000.0
    B = lambda t: t / T
    A = lambda t: 1 - (t / T)
    return A(t) * x_0 + B(t) * x_T
def area_resize(image, max_area):

    height, width = image.shape[-2:]
    scale = math.sqrt(max_area / (height * width))

    resized_height, resized_width = round(height * scale), round(width * scale)

    return TVF.resize(
        image,
        size=(resized_height, resized_width),
        interpolation=InterpolationMode.BICUBIC,
    )

def div_pad(image, factor):

    height_factor, width_factor = factor
    height, width = image.shape[-2:]

    pad_height = (height_factor - (height % height_factor)) % height_factor
    pad_width = (width_factor - (width % width_factor)) % width_factor

    if pad_height == 0 and pad_width == 0:
        return image

    if isinstance(image, torch.Tensor):
        padding = (0, pad_width, 0, pad_height)
        image = torch.nn.functional.pad(image, padding, mode='constant', value=0.0)

    return image

def cut_videos(videos):
    t = videos.size(1)
    if t == 1:
        return videos
    if t <= 4 :
        padding = [videos[:, -1].unsqueeze(1)] * (4 - t + 1)
        padding = torch.cat(padding, dim=1)
        videos = torch.cat([videos, padding], dim=1)
        return videos
    if (t - 1) % (4) == 0:
        return videos
    else:
        padding = [videos[:, -1].unsqueeze(1)] * (
            4 - ((t - 1) % (4))
        )
        padding = torch.cat(padding, dim=1)
        videos = torch.cat([videos, padding], dim=1)
        assert (videos.size(1) - 1) % (4) == 0
        return videos

def side_resize(image, size):
    antialias = not (isinstance(image, torch.Tensor) and image.device.type == 'mps')
    resized = TVF.resize(image, size, InterpolationMode.BICUBIC, antialias=antialias)
    return resized

class SeedVR2InputProcessing(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id = "SeedVR2InputProcessing",
            category="image/video",
            inputs = [
                io.Image.Input("images"),
                io.Vae.Input("vae"),
                io.Int.Input("resolution", default = 1280, min = 120), # just non-zero value
                io.Int.Input("spatial_tile_size", default = 512, min = 1),
                io.Int.Input("spatial_overlap", default = 64, min = 1),
                io.Int.Input("temporal_tile_size", default=16, min=0, max=16384, step=4),
                io.Int.Input("temporal_overlap", default=4, min=0, max=16384, step=4),
                io.Boolean.Input("enable_tiling", default=False),
            ],
            outputs = [
                io.Latent.Output("vae_conditioning")
            ]
        )

    @classmethod
    def execute(cls, images, vae, resolution, spatial_tile_size, spatial_overlap, temporal_tile_size, temporal_overlap, enable_tiling):

        comfy.model_management.load_models_gpu([vae.patcher])
        vae_model = vae.first_stage_model
        scale = 0.9152
        shift = 0
        if images.dim() != 5: # add the t dim
            images = images.unsqueeze(0)
        images = images.permute(0, 1, 4, 2, 3)

        b, t, c, h, w = images.shape
        images = images.reshape(b * t, c, h, w)

        clip = Lambda(lambda x: torch.clamp(x, 0.0, 1.0))
        normalize = Normalize(0.5, 0.5)
        images = side_resize(images, resolution)

        images = clip(images)
        o_h, o_w = images.shape[-2:]
        images = div_pad(images, (16, 16))
        images = normalize(images)
        _, _, new_h, new_w = images.shape

        images = images.reshape(b, t, c, new_h, new_w)
        # Preserve the unpadded user-visible temporal length for decode trim.
        images_bcthw_unpadded = rearrange(images, "b t c h w -> b c t h w")
        images = cut_videos(images)
        images_bthwc = rearrange(images, "b t c h w -> b t h w c")
        original_image_video = images_bcthw_unpadded

        # in case users a non-compatiable number for tiling
        def make_divisible(val, divisor):
            return max(divisor, round(val / divisor) * divisor)

        spatial_tile_size = make_divisible(spatial_tile_size, 32)
        spatial_overlap = make_divisible(spatial_overlap, 32)

        if spatial_overlap >= spatial_tile_size:
            spatial_overlap = max(0, spatial_tile_size - 8)

        args = {
            "tile_size": (spatial_tile_size, spatial_tile_size),
            "tile_overlap": (spatial_overlap, spatial_overlap),
            "temporal_size": temporal_tile_size,
            "temporal_overlap": temporal_overlap,
        }
        if enable_tiling:
            vae_model.img_dims = [o_h, o_w]
            vae_model.original_image_video = original_image_video
            images_bcthw = rearrange(images_bthwc, "b t h w c -> b c t h w")
            # Move input to the VAE's loaded execution device. VideoAutoencoderKLWrapper.encode
            # adopts x.device as self.device without moving x, so a CPU input against a
            # GPU-loaded VAE silently falls back to CPU encode. Use vae.patcher.load_device
            # (the device load_models_gpu loaded the wrapper to) when available; fall back
            # to the wrapper's parameter device.
            vae_device = getattr(getattr(vae, "patcher", None), "load_device", None)
            if vae_device is None:
                vae_device = next(vae_model.parameters()).device
            images_bcthw = images_bcthw.to(vae_device)
            latent = tiled_vae(images_bcthw, vae_model, **args, encode=True)
        else:
            vae_model.img_dims = [o_h, o_w]
            vae_model.original_image_video = original_image_video
            vae_model.tiled_args = {**args, "enable_tiling": False}
            latent = vae.encode(images_bthwc)

        clear_vae_memory(vae_model)
        #images = images.to(offload_device)
        #vae_model = vae_model.to(offload_device)

        vae_model.img_dims = [o_h, o_w]
        args["enable_tiling"] = enable_tiling
        vae_model.tiled_args = args
        vae_model.original_image_video = original_image_video

        latent = latent.unsqueeze(2) if latent.ndim == 4 else latent
        latent = rearrange(latent, "b c ... -> b ... c")

        latent = (latent - shift) * scale

        return io.NodeOutput({"samples": latent})

class SeedVR2Conditioning(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SeedVR2Conditioning",
            category="image/video",
            inputs=[
                io.Latent.Input("vae_conditioning"),
                io.Model.Input("model"),
                io.Float.Input("latent_noise_scale", default=0.0, step=0.001)
            ],
            outputs=[io.Conditioning.Output(display_name = "positive"),
                     io.Conditioning.Output(display_name = "negative"),
                     io.Latent.Output(display_name = "latent")],
        )

    @classmethod
    def execute(cls, vae_conditioning, model, latent_noise_scale) -> io.NodeOutput:

        vae_conditioning = vae_conditioning["samples"]
        device = vae_conditioning.device
        model_patcher = model
        model = _resolve_seedvr2_diffusion_model(model_patcher)
        model_patcher.disable_model_cfg1_optimization()
        pos_cond = model.positive_conditioning
        neg_cond = model.negative_conditioning

        # Fail-loud guard against silently-wrong output when a numz-format
        # DiT-only ``.safetensors`` (no ``positive_conditioning`` /
        # ``negative_conditioning`` keys) is loaded via ``UNETLoader``.
        # ``NaDiT.__init__`` zero-fills the buffers via ``torch.zeros`` (see
        # ``comfy/ldm/seedvr/model.py``); ``load_state_dict(strict=False)``
        # leaves them at zero when the keys are absent. Detect that state
        # here rather than at ``BaseModel.extra_conds`` (per sampling step,
        # wasteful) or at the resolver helper (mixes structural shape with
        # semantic content). Both buffers must be checked together — partial
        # bake regressions could populate one but not the other.
        if (
            pos_cond.float().abs().sum().item() == 0
            and neg_cond.float().abs().sum().item() == 0
        ):
            source_path = _describe_seedvr2_model_source(model_patcher)
            file_clause = (
                f"Source file: {source_path}. " if source_path else ""
            )
            raise RuntimeError(
                f"{_SEEDVR2_INVALID_MODEL_MSG_PREFIX}: positive_conditioning "
                f"and negative_conditioning buffers are zero-valued — model "
                f"file appears to be a numz-format DiT-only export missing "
                f"the SeedVR2 conditioning tensors. {file_clause}"
                f"Re-bake the file with ``positive_conditioning`` (58, 5120) "
                f"and ``negative_conditioning`` (64, 5120) keys at top level, "
                f"or load via CheckpointLoaderSimple from a bundled "
                f"checkpoint."
            )

        _apply_rope_freqs_float32_cast(model)

        noises = torch.randn_like(vae_conditioning, dtype=vae_conditioning.dtype).to(device)
        aug_noises =  torch.randn_like(vae_conditioning, dtype=vae_conditioning.dtype).to(device)
        aug_noises = noises * 0.1 + aug_noises * 0.05
        cond_noise_scale = latent_noise_scale
        t = (
            torch.tensor([1000.0])
            * cond_noise_scale
        ).to(device)
        shape = torch.tensor(vae_conditioning.shape[1:]).to(device)[None] # avoid batch dim
        t = timestep_transform(t, shape)
        cond = inter(vae_conditioning, aug_noises, t)
        condition = torch.stack([get_conditions(noise, c) for noise, c in zip(noises, cond)])
        condition = condition.movedim(-1, 1)
        noises = noises.movedim(-1, 1)

        pos_shape = pos_cond.shape[0]
        neg_shape = neg_cond.shape[0]
        diff = abs(pos_shape - neg_shape)
        if pos_shape > neg_shape:
            neg_cond = F.pad(neg_cond, (0, 0, 0, diff))
        else:
            pos_cond = F.pad(pos_cond, (0, 0, 0, diff))

        noises = rearrange(noises, "b c t h w -> b (c t) h w")
        condition = rearrange(condition, "b c t h w -> b (c t) h w")

        negative = [[neg_cond.unsqueeze(0), {"condition": condition}]]
        positive = [[pos_cond.unsqueeze(0), {"condition": condition}]]

        return io.NodeOutput(positive, negative, {"samples": noises})

# SeedVR2 latent / conditioning channel constants. The SeedVR2 conditioning
# stage collapses ``(B, C, T, H, W) -> (B, C*T, H, W)`` for both the latent
# (C=16) and the per-frame condition tensor (C=17 = 16 latent + 1 mask), as
# required by ``NaDiT.forward`` which un-collapses via
# ``view(B, 16, -1, H, W)`` and ``view(B, 17, -1, H, W)`` respectively.
_SEEDVR2_LATENT_CHANNELS = 16
_SEEDVR2_CONDITION_CHANNELS = 17


def _slice_collapsed_4d_along_t(tensor_4d: torch.Tensor, t_start: int,
                                 t_end: int, channels: int) -> torch.Tensor:
    """Slice a SeedVR2-style collapsed 4D tensor ``(B, channels*T, H, W)``
    along the latent T axis, returning ``(B, channels*(t_end - t_start), H, W)``.

    Reshape -> slice -> ``.contiguous()`` -> re-collapse. ``reshape`` is
    used for the un-collapse so non-contiguous incoming tensors from
    cropping or slicing nodes are accepted. The
    ``.contiguous()`` is mandatory: T-axis slicing of a 5D tensor produces a
    non-contiguous view, and the subsequent re-collapse requires contiguous
    storage.
    """
    B, CT, H, W = tensor_4d.shape
    if CT % channels != 0:
        raise ValueError(
            f"_slice_collapsed_4d_along_t: collapsed channel dim {CT} is not "
            f"divisible by channels={channels}; tensor shape {tuple(tensor_4d.shape)}."
        )
    T = CT // channels
    if not (0 <= t_start < t_end <= T):
        raise ValueError(
            f"_slice_collapsed_4d_along_t: slice [{t_start}:{t_end}] out of "
            f"range for T={T}."
        )
    new_T = t_end - t_start
    sliced = tensor_4d.reshape(B, channels, T, H, W)[:, :, t_start:t_end, :, :].contiguous()
    return sliced.reshape(B, channels * new_T, H, W)


def _slice_seedvr2_cond_along_t(cond_list, t_start: int, t_end: int):
    """Build a new SeedVR2 conditioning list with the per-frame ``condition``
    tensor sliced along the latent T axis.

    SeedVR2 conditioning entries have the shape
    ``[text_cond_tensor, options_dict]`` where ``options_dict["condition"]``
    is a 4D collapsed ``(B, 17*T, H, W)`` tensor; the text tensor itself has
    no temporal axis and is passed through unchanged. Other keys in the
    options dict (controlnets, etc.) are also passed through unchanged. If
    an entry has no ``"condition"`` key, the entry is forwarded verbatim.

    A new list of ``[text_cond, new_options_dict]`` pairs is returned; the
    original ``cond_list`` and its options dicts are not mutated.
    """
    new_list = []
    for entry in cond_list:
        text_cond, options = entry[0], entry[1]
        if "condition" not in options:
            new_list.append(entry)
            continue
        new_options = options.copy()
        new_options["condition"] = _slice_collapsed_4d_along_t(
            new_options["condition"], t_start, t_end,
            _SEEDVR2_CONDITION_CHANNELS,
        )
        new_list.append([text_cond, new_options])
    return new_list


def _slice_seedvr2_noise_mask_along_t(noise_mask: torch.Tensor,
                                      samples_4d: torch.Tensor,
                                      t_start: int,
                                      t_end: int):
    """Slice collapsed SeedVR2 masks and preserve standard masks.

    ``SetLatentNoiseMask`` produces ``(B, 1, H, W)`` masks that KSampler
    expands to the latent shape. Only masks already expanded to the full
    collapsed ``(B, 16*T, H, W)`` shape need temporal slicing here.
    """
    if noise_mask.ndim == samples_4d.ndim and noise_mask.shape[1] == samples_4d.shape[1]:
        return _slice_collapsed_4d_along_t(
            noise_mask, t_start, t_end, _SEEDVR2_LATENT_CHANNELS,
        )
    return noise_mask


def _concat_chunks_along_t(chunks_4d, channels: int) -> torch.Tensor:
    """Concatenate a list of SeedVR2-style collapsed 4D tensors
    ``(B, channels*T_i, H, W)`` along the latent T axis. Each chunk is
    un-collapsed to 5D, concatenated on ``dim=2``, then re-collapsed to 4D.
    """
    if len(chunks_4d) == 0:
        raise ValueError("_concat_chunks_along_t: empty chunk list.")
    fives = []
    for ch in chunks_4d:
        B, CT, H, W = ch.shape
        if CT % channels != 0:
            raise ValueError(
                f"_concat_chunks_along_t: chunk shape {tuple(ch.shape)} "
                f"channel dim {CT} not divisible by channels={channels}."
            )
        T = CT // channels
        fives.append(ch.reshape(B, channels, T, H, W))
    cat = torch.cat(fives, dim=2).contiguous()
    B, C, T_total, H, W = cat.shape
    return cat.reshape(B, C * T_total, H, W)


def _hann_blend_weights_1d(overlap: int, device, dtype) -> torch.Tensor:
    """Build a 1D crossfade weight tensor of length ``overlap`` for the
    *previous* chunk's contribution; the current chunk's weight is
    ``1 - w_prev``.

    Mirrors the numz ``blend_overlapping_frames`` shape
    (AInVFX/numz fork ``src/core/generation_utils.py``,
    ``blend_overlapping_frames``): a Hann window with a ``[1/3, 2/3]``
    dead-band when ``overlap >= 3``, and a plain linear ramp when
    ``overlap < 3`` (the dead-band would collapse the transition for
    very small overlap counts). The numz reference operates on
    pixel-space tensors ``[overlap, H, W, C]``; this 1D form is
    reshaped by the caller to broadcast across the latent's
    ``(B, C, T_overlap, H, W)`` axes.
    """
    if overlap < 1:
        raise ValueError(
            f"_hann_blend_weights_1d: overlap must be >= 1; got {overlap}."
        )
    if overlap >= 3:
        t = torch.linspace(0.0, 1.0, steps=overlap, device=device, dtype=dtype)
        blend_start = 1.0 / 3.0
        blend_end = 2.0 / 3.0
        u = ((t - blend_start) / (blend_end - blend_start)).clamp(0.0, 1.0)
        return 0.5 + 0.5 * torch.cos(torch.pi * u)
    return torch.linspace(1.0, 0.0, steps=overlap, device=device, dtype=dtype)


def _blend_overlap_region(prev_tail_5d: torch.Tensor,
                          cur_head_5d: torch.Tensor) -> torch.Tensor:
    """Blend two 5D ``(B, C, T_overlap, H, W)`` tensors of equal shape
    using a 1D Hann/linear ramp along the T axis. ``prev_tail_5d``
    receives the descending weight; ``cur_head_5d`` receives
    ``1 - w_prev``.

    The caller is responsible for ensuring both inputs have identical
    shape and dtype/device.
    """
    if prev_tail_5d.shape != cur_head_5d.shape:
        raise ValueError(
            f"_blend_overlap_region: shape mismatch "
            f"prev {tuple(prev_tail_5d.shape)} vs "
            f"cur {tuple(cur_head_5d.shape)}."
        )
    overlap = int(prev_tail_5d.shape[2])
    w_prev_1d = _hann_blend_weights_1d(
        overlap, prev_tail_5d.device, prev_tail_5d.dtype,
    )
    # Reshape to (1, 1, overlap, 1, 1) for broadcast across B, C, H, W.
    w_prev = w_prev_1d.view(1, 1, overlap, 1, 1)
    w_cur = 1.0 - w_prev
    return prev_tail_5d * w_prev + cur_head_5d * w_cur


def _concat_chunks_with_overlap_blend(chunk_specs, channels: int,
                                      overlap_latent: int) -> torch.Tensor:
    """Concatenate temporally-overlapping chunks back into a single
    collapsed 4D tensor, blending overlap regions with a Hann/linear
    crossfade.

    ``chunk_specs`` is a list of ``(t_start, t_end, chunk_4d)`` tuples
    in source-latent T coordinates. ``overlap_latent == 0`` is a fast
    path that delegates to plain concatenation (and produces output
    bit-identical to ``_concat_chunks_along_t`` of the same chunks).

    The blend at each pair of adjacent chunks acts on the actual
    overlap region width ``min(prev_end - cur_start, current chunk
    length)``, which may be smaller than ``overlap_latent`` when the
    final chunk is a runt shorter than the configured overlap.
    """
    if len(chunk_specs) == 0:
        raise ValueError("_concat_chunks_with_overlap_blend: empty chunk list.")
    if overlap_latent < 0:
        raise ValueError(
            f"_concat_chunks_with_overlap_blend: overlap_latent must be "
            f">= 0; got {overlap_latent}."
        )

    # Validate channel divisibility once and capture per-chunk T.
    chunk_5d = []
    for t_start, t_end, ch in chunk_specs:
        B, CT, H, W = ch.shape
        if CT % channels != 0:
            raise ValueError(
                f"_concat_chunks_with_overlap_blend: chunk shape "
                f"{tuple(ch.shape)} channel dim {CT} not divisible "
                f"by channels={channels}."
            )
        T = CT // channels
        if t_end - t_start != T:
            raise ValueError(
                f"_concat_chunks_with_overlap_blend: chunk T={T} mismatches "
                f"declared range [{t_start}:{t_end}]."
            )
        chunk_5d.append((t_start, t_end, ch.reshape(B, channels, T, H, W)))

    if overlap_latent == 0:
        # Fast path: pure concat in the caller-provided chunk order.
        return _concat_chunks_along_t(
            [c.reshape(c.shape[0], channels * c.shape[2], c.shape[3], c.shape[4])
             for _, _, c in chunk_5d],
            channels,
        )

    T_total = max(t_end for _, t_end, _ in chunk_5d)
    first_5d = chunk_5d[0][2]
    B = first_5d.shape[0]
    H = first_5d.shape[3]
    W = first_5d.shape[4]
    result = torch.empty(
        (B, channels, T_total, H, W),
        device=first_5d.device, dtype=first_5d.dtype,
    )
    filled_until = 0
    for i, (cs, ce, ct_5d) in enumerate(chunk_5d):
        chunk_T = int(ct_5d.shape[2])
        if i == 0:
            result[:, :, cs:ce, :, :] = ct_5d
            filled_until = ce
            continue
        # Overlap region width is bounded by both the previous fill
        # frontier and the current chunk's actual length (for runt
        # final chunks shorter than the configured overlap).
        overlap_len = min(filled_until - cs, chunk_T)
        if overlap_len > 0:
            prev_tail = result[:, :, cs:cs + overlap_len, :, :].contiguous()
            cur_head = ct_5d[:, :, :overlap_len, :, :].contiguous()
            blended = _blend_overlap_region(prev_tail, cur_head)
            result[:, :, cs:cs + overlap_len, :, :] = blended
            tail_start = cs + overlap_len
            tail_end = ce
            if tail_end > tail_start:
                result[:, :, tail_start:tail_end, :, :] = (
                    ct_5d[:, :, overlap_len:, :, :]
                )
        else:
            # Disjoint chunks (overlap_latent set but this pair did not
            # actually overlap, e.g. step_latent equal to chunk_latent
            # in a degenerate config). Treat as concat.
            result[:, :, cs:ce, :, :] = ct_5d
        filled_until = ce

    return result.contiguous().reshape(B, channels * T_total, H, W)


def _run_standard_sample(model, seed: int, steps: int, cfg: float,
                         sampler_name: str, scheduler: str,
                         positive, negative, latent_image: dict,
                         denoise: float) -> dict:
    """Single-shot delegation that mirrors the standard ``common_ksampler``
    flow (``nodes.py:common_ksampler``): generate noise from seed, run
    ``comfy.sample.sample``, return a latent dict. Used by the
    ProgressiveSampler short-circuit when the full sequence fits in one
    chunk so chunking introduces no overhead for small videos.
    """
    samples_in = latent_image["samples"]
    samples_in = comfy.sample.fix_empty_latent_channels(
        model, samples_in, latent_image.get("downscale_ratio_spacial", None),
    )
    batch_inds = latent_image.get("batch_index", None)
    noise = comfy.sample.prepare_noise(samples_in, seed, batch_inds)
    noise_mask = latent_image.get("noise_mask", None)
    samples = comfy.sample.sample(
        model, noise, steps, cfg, sampler_name, scheduler,
        positive, negative, samples_in,
        denoise=denoise, noise_mask=noise_mask, seed=seed,
    )
    out = latent_image.copy()
    out.pop("downscale_ratio_spacial", None)
    out["samples"] = samples
    return out


class SeedVR2ProgressiveSampler(io.ComfyNode):
    """Sequential temporal chunking sampler for SeedVR2 native.

    Drop-in replacement for ``KSampler`` in SeedVR2 native workflows that
    OOM on long sequences. The latent enters the sampler in SeedVR2's
    collapsed form ``(B, 16*T, H, W)`` (collapsed by ``SeedVR2Conditioning``
    at ``rearrange(b c t h w -> b (c t) h w)``); this node slices that
    tensor along the temporal axis, runs the configured inner sampler
    sequentially per chunk against the standard ``comfy.sample.sample``
    entry point, and concatenates per-chunk outputs back into a single
    ``(B, 16*T_total, H, W)`` latent.

    ``frames_per_chunk`` is expressed in pixel-frame units to match the
    SeedVR2 4n+1 constraint enforced upstream by ``cut_videos`` and the
    VAE's ``temporal_downsample_factor=4``. A pixel chunk size ``F``
    maps to ``(F - 1) // 4 + 1`` latent-frame chunks.

    Determinism contract: a single noise tensor is generated once from
    the user seed and sliced per chunk (rather than re-seeding each
    chunk), so a workflow that fits in a single chunk produces output
    identical to a workflow that fits in N chunks at the same seed,
    modulo the inherent T-axis chunk-boundary independence of the model.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SeedVR2ProgressiveSampler",
            category="sampling",
            inputs=[
                io.Model.Input("model"),
                io.Int.Input("seed", default=0, min=0,
                             max=0xffffffffffffffff,
                             control_after_generate=True),
                io.Int.Input("steps", default=20, min=1, max=10000),
                io.Float.Input("cfg", default=1.0, min=0.0, max=100.0,
                               step=0.1, round=0.01),
                io.Combo.Input("sampler_name",
                               options=comfy.samplers.SAMPLER_NAMES),
                io.Combo.Input("scheduler",
                               options=comfy.samplers.SCHEDULER_NAMES),
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Latent.Input("latent_image"),
                io.Float.Input("denoise", default=1.0, min=0.0, max=1.0,
                               step=0.01),
                io.Int.Input("frames_per_chunk", default=21, min=1,
                             max=16384, step=4),
                io.Int.Input("temporal_overlap", default=0, min=0,
                             max=16384,
                             tooltip="Latent-frame overlap between "
                                     "adjacent chunks; blended with a "
                                     "Hann window (linear for overlap "
                                     "< 3). 0 = no blend, pure concat. "
                                     "Must be < chunk_latent derived "
                                     "from frames_per_chunk; 1 latent "
                                     "frame corresponds to ~4 pixel "
                                     "frames."),
            ],
            outputs=[io.Latent.Output()],
        )

    @classmethod
    def execute(cls, model, seed, steps, cfg, sampler_name, scheduler,
                positive, negative, latent_image, denoise,
                frames_per_chunk, temporal_overlap) -> io.NodeOutput:
        # 4n+1 validation in pixel-frame domain. The SeedVR2 native pipeline
        # requires pixel-frame counts of the form 4n+1 (1, 5, 9, 13, ...),
        # imposed at ``cut_videos`` upstream and propagated through the VAE's
        # temporal_downsample_factor=4. Reject violations explicitly before
        # any model invocation; a silent rounding would mis-align chunk
        # boundaries with the 4n+1 lattice.
        if frames_per_chunk < 1 or (frames_per_chunk - 1) % 4 != 0:
            raise ValueError(
                f"SeedVR2ProgressiveSampler: frames_per_chunk must be a "
                f"4n+1 pixel-frame count (1, 5, 9, 13, 17, 21, ...); "
                f"got {frames_per_chunk}."
            )

        samples_4d = latent_image["samples"]
        samples_4d = comfy.sample.fix_empty_latent_channels(
            model, samples_4d,
            latent_image.get("downscale_ratio_spacial", None),
        )
        if samples_4d.ndim != 4:
            raise ValueError(
                f"SeedVR2ProgressiveSampler: expected 4D collapsed latent "
                f"(B, 16*T, H, W); got shape {tuple(samples_4d.shape)}."
            )
        B, CT, H, W = samples_4d.shape
        if CT % _SEEDVR2_LATENT_CHANNELS != 0:
            raise ValueError(
                f"SeedVR2ProgressiveSampler: collapsed channel dim {CT} is "
                f"not divisible by SeedVR2 latent channels "
                f"{_SEEDVR2_LATENT_CHANNELS}; latent does not appear to be "
                f"SeedVR2-shaped."
            )
        T_latent = CT // _SEEDVR2_LATENT_CHANNELS
        T_pixel = 4 * (T_latent - 1) + 1

        # Short-circuit: total fits in one chunk -> standard path with no
        # chunking overhead. Output of this branch is byte-identical to the
        # built-in KSampler given the same (model, seed, steps, cfg,
        # sampler_name, scheduler, positive, negative, latent_image,
        # denoise) tuple.
        if T_pixel <= frames_per_chunk:
            return io.NodeOutput(_run_standard_sample(
                model, seed, steps, cfg, sampler_name, scheduler,
                positive, negative, latent_image, denoise,
            ))

        # Map pixel chunk -> latent chunk. Each chunk's latent length is
        # at most ``chunk_latent``; the final chunk may be a runt that
        # is automatically 4n+1-aligned in the pixel domain by the
        # T_pixel = 4*(T_latent-1) + 1 mapping (every positive integer
        # T_latent corresponds to a valid 4n+1 pixel count).
        chunk_latent = (frames_per_chunk - 1) // 4 + 1

        # ``temporal_overlap`` is exposed in latent-frame units. The
        # validation here keeps the chunk loop's stride strictly
        # positive; without it a config like overlap >= chunk would
        # produce zero or negative stride and an infinite loop.
        if temporal_overlap < 0 or temporal_overlap >= chunk_latent:
            raise ValueError(
                f"SeedVR2ProgressiveSampler: temporal_overlap must be in "
                f"[0, chunk_latent) latent frames where chunk_latent="
                f"{chunk_latent} (derived from frames_per_chunk="
                f"{frames_per_chunk}); got {temporal_overlap}."
            )
        step_latent = chunk_latent - temporal_overlap

        # Generate full noise once from the user seed, then slice along T
        # per chunk. Using one global noise tensor (rather than re-seeding
        # per chunk) preserves seed-determinism across chunk-count
        # variations: the same (seed, total T_latent) always produces the
        # same noise samples regardless of how the work is partitioned.
        batch_inds = latent_image.get("batch_index", None)
        noise_full = comfy.sample.prepare_noise(samples_4d, seed, batch_inds)

        noise_mask = latent_image.get("noise_mask", None)

        # Build the flat list of chunk ranges first so the chunking
        # geometry is fully known before any sample call. Slicing and
        # ``comfy.sample.sample`` happen later in either the sequential
        # path or the worksplit workers; keeping the geometry pass
        # separate also lets the worksplit path round-robin chunks
        # across devices without re-walking the stride logic.
        chunk_ranges = []
        for chunk_start in range(0, T_latent, step_latent):
            chunk_end = min(chunk_start + chunk_latent, T_latent)
            if chunk_start >= chunk_end:
                # The final iteration of a stride that lands exactly on
                # T_latent produces a zero-length chunk; skip it.
                break
            chunk_ranges.append((chunk_start, chunk_end))
            if chunk_end >= T_latent:
                break

        # Per-chunk sample call. Used by both the sequential path and
        # by each worksplit worker so the slicing + sampler invocation
        # logic stays in one place. ``patcher`` is the ModelPatcher to
        # invoke for this chunk's sample call — the primary ``model``
        # on the sequential path, or a per-device clone on the
        # worksplit path.
        def _sample_one_chunk(patcher, chunk_start, chunk_end):
            samples_chunk = _slice_collapsed_4d_along_t(
                samples_4d, chunk_start, chunk_end,
                _SEEDVR2_LATENT_CHANNELS,
            )
            noise_chunk = _slice_collapsed_4d_along_t(
                noise_full, chunk_start, chunk_end,
                _SEEDVR2_LATENT_CHANNELS,
            )
            positive_chunk = _slice_seedvr2_cond_along_t(
                positive, chunk_start, chunk_end,
            )
            negative_chunk = _slice_seedvr2_cond_along_t(
                negative, chunk_start, chunk_end,
            )

            # Per-chunk noise_mask handling: standard masks are passed
            # through for KSampler expansion; pre-expanded collapsed
            # masks are sliced.
            chunk_noise_mask = None
            if noise_mask is not None:
                chunk_noise_mask = _slice_seedvr2_noise_mask_along_t(
                    noise_mask, samples_4d, chunk_start, chunk_end,
                )

            return comfy.sample.sample(
                patcher, noise_chunk, steps, cfg, sampler_name, scheduler,
                positive_chunk, negative_chunk, samples_chunk,
                denoise=denoise, noise_mask=chunk_noise_mask, seed=seed,
            )

        # Worksplit clones are attached upstream by
        # ``MultiGPU_WorkUnits`` (or any node that calls
        # ``create_multigpu_deepclones``). They live in the
        # additional-models registry under the ``"multigpu"`` key at
        # ``execute()`` time — ``model_options["multigpu_clones"]`` is
        # populated later, inside ``KSampler.outer_sample``, so we MUST
        # read from the registry here rather than from
        # ``model_options``.
        extra_clones = model.get_additional_models_with_key("multigpu")

        if not extra_clones:
            # Sequential path — byte-identical to the pre-worksplit
            # chunk loop. Each chunk runs on the primary model in
            # temporal order.
            chunk_specs = []
            for idx, (chunk_start, chunk_end) in enumerate(chunk_ranges):
                t0 = time.perf_counter()
                chunk_samples = _sample_one_chunk(model, chunk_start, chunk_end)
                t1 = time.perf_counter()
                logging.info(
                    f"INSTRUMENT_SEEDVR2_CHUNK_TIME path=standard "
                    f"chunk_idx={idx} chunk_start={chunk_start} "
                    f"chunk_end={chunk_end} device={model.load_device} "
                    f"duration_ms={(t1 - t0) * 1000.0:.2f}"
                )
                chunk_specs.append((chunk_start, chunk_end, chunk_samples))
        else:
            import comfy.multigpu

            # Worksplit path — round-robin chunks across
            # ``[primary_device, *clone_devices]`` and run sample()
            # calls in parallel through ``MultiGPUThreadPool``.
            #
            # Recursion / CFG-split guard:
            # The existing multigpu CFG-split machinery in
            # ``comfy/samplers.py`` (``_calc_cond_batch_multigpu``) is
            # triggered when ``KSampler.outer_sample`` →
            # ``prepare_model_patcher_multigpu_clones`` finds at least
            # one ``ModelPatcher`` with ``is_multigpu_base_clone=True``
            # in ``loaded_models`` and populates
            # ``model_options["multigpu_clones"]``. When that happens,
            # the sampler builds a 2-batch (cond+uncond) ``context``
            # tensor — SeedVR2's ``model.forward`` requires either
            # ``numel()==0`` (empty text, falls back to the model's
            # built-in conditioning buffers) or
            # ``shape[0]==2`` (cond+uncond). At cfg=1.0 with empty
            # text and no multigpu CFG-split, the single-device
            # ``_calc_cond_batch`` path produces ``numel()==0`` and
            # SeedVR2 takes its built-in conditioning fallback.
            #
            # For chunk-half worksplit we want every worker to take
            # exactly that single-device path so each chunk runs the
            # same code as the sequential baseline — just on a
            # different device. To make this deterministic we
            # temporarily flip ``is_multigpu_base_clone=False`` on
            # the primary and every extra clone for the duration of
            # the dispatch, then restore. The flag only controls the
            # filter inside ``prepare_model_patcher_multigpu_clones``;
            # flipping it has no other side effects.
            devices = [model.load_device]
            patcher_for_device = {model.load_device: model}
            for clone in extra_clones:
                devices.append(clone.load_device)
                patcher_for_device[clone.load_device] = clone

            # Round-robin assignment: chunk i → devices[i % N]. Devices
            # with no chunks (more devices than chunks) are skipped on
            # both submit AND drain to avoid blocking on an empty
            # result queue.
            per_device_ranges: dict = {dev: [] for dev in devices}
            for i, (cs, ce) in enumerate(chunk_ranges):
                per_device_ranges[devices[i % len(devices)]].append((i, cs, ce))

            def _worker_run_chunks(device, patcher, ranges):
                # Per-worker patcher: shallow clone (no weight copy),
                # strip the multigpu additional-models registry, and
                # force ``is_multigpu_base_clone=False`` so the
                # worker's ``outer_sample`` does not enter the
                # multigpu CFG-split path inside the sampler.
                worker_patcher = patcher.clone()
                worker_patcher.remove_additional_models("multigpu")
                worker_patcher.is_multigpu_base_clone = False
                # Propagate SeedVR2's CFG=1.0 contract: SeedVR2's
                # ``model.forward`` requires a 2-batch ``context``
                # tensor (cond+uncond concatenated along dim 0). The
                # primary patcher had ``disable_cfg1_optimization``
                # set by ``SeedVR2Conditioning.execute`` so its sample
                # call always processes both branches even at cfg=1.0.
                # Extra clones were created by ``MultiGPU_WorkUnits``
                # BEFORE that node ran, so their ``model_options`` do
                # not carry the flag — we propagate it explicitly here
                # so every worker (primary's clone OR extra clone's
                # clone) takes the same 2-batch sampling path. Without
                # this propagation, only the primary's clone runs
                # 2-batch and the others fail at
                # ``context.chunk(2, dim=0)`` inside the DiT.
                worker_patcher.disable_model_cfg1_optimization()
                results = []
                with comfy.model_management.cuda_device_context(device):
                    for (idx, cs, ce) in ranges:
                        t0 = time.perf_counter()
                        chunk_samples = _sample_one_chunk(worker_patcher, cs, ce)
                        t1 = time.perf_counter()
                        logging.info(
                            f"INSTRUMENT_SEEDVR2_CHUNK_TIME path=worksplit "
                            f"chunk_idx={idx} chunk_start={cs} "
                            f"chunk_end={ce} device={device} "
                            f"duration_ms={(t1 - t0) * 1000.0:.2f}"
                        )
                        results.append((idx, cs, ce, chunk_samples))
                return results

            # Flip the flag on every participating patcher BEFORE
            # dispatch so ``prepare_model_patcher_multigpu_clones``
            # cannot find any multigpu-base-clone in ``loaded_models``
            # no matter which worker observes the cache first. Restore
            # the original flags in ``finally`` so downstream nodes
            # (and the next graph execution) see the model exactly as
            # MultiGPU_WorkUnits left it.
            saved_flag_model = model.is_multigpu_base_clone
            saved_flag_extras = [c.is_multigpu_base_clone for c in extra_clones]
            model.is_multigpu_base_clone = False
            for c in extra_clones:
                c.is_multigpu_base_clone = False

            pool = comfy.multigpu.MultiGPUThreadPool(devices)
            dispatch_t0 = time.perf_counter()
            try:
                submitted_devices = []
                for dev in devices:
                    if per_device_ranges[dev]:
                        pool.submit(dev, _worker_run_chunks, dev,
                                    patcher_for_device[dev],
                                    per_device_ranges[dev])
                        submitted_devices.append(dev)

                # Drain every submitted device. Surfacing the first
                # worker exception (if any) as a chained RuntimeError
                # tells the caller which device failed and preserves
                # the original traceback via ``raise ... from``.
                per_device_results = {}
                first_error = None
                for dev in submitted_devices:
                    result, error = pool.get_result(dev)
                    if error is not None and first_error is None:
                        first_error = (dev, error)
                    per_device_results[dev] = result
            finally:
                pool.shutdown()
                model.is_multigpu_base_clone = saved_flag_model
                for c, f in zip(extra_clones, saved_flag_extras):
                    c.is_multigpu_base_clone = f
            dispatch_t1 = time.perf_counter()

            if first_error is not None:
                err_dev, err = first_error
                raise RuntimeError(
                    f"SeedVR2ProgressiveSampler: worksplit worker on "
                    f"{err_dev} raised an exception during chunk "
                    f"sampling."
                ) from err

            # Reassemble by chunk_idx so the temporal blender sees
            # chunks in order regardless of which device produced
            # them.
            indexed = []
            for dev in submitted_devices:
                indexed.extend(per_device_results[dev])
            indexed.sort(key=lambda x: x[0])

            logging.info(
                f"INSTRUMENT_SEEDVR2_DISPATCH_TIME path=worksplit "
                f"n_chunks={len(chunk_ranges)} "
                f"n_devices={len(submitted_devices)} "
                f"parallel_wall_ms={(dispatch_t1 - dispatch_t0) * 1000.0:.2f}"
            )

            chunk_specs = [(cs, ce, chunk_samples)
                           for (_, cs, ce, chunk_samples) in indexed]

        final = _concat_chunks_with_overlap_blend(
            chunk_specs, _SEEDVR2_LATENT_CHANNELS, temporal_overlap,
        )

        out = latent_image.copy()
        out.pop("downscale_ratio_spacial", None)
        out["samples"] = final
        return io.NodeOutput(out)


class SeedVRExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            SeedVR2Conditioning,
            SeedVR2InputProcessing,
            SeedVR2ProgressiveSampler,
        ]

async def comfy_entrypoint() -> SeedVRExtension:
    return SeedVRExtension()
