from typing_extensions import override
from comfy_api.latest import ComfyExtension, io
import torch
import math
from einops import rearrange

import gc
import comfy.model_management
from comfy.ldm.seedvr.vae import tiled_vae

import torch.nn.functional as F
from torchvision.transforms import functional as TVF
from torchvision.transforms import Lambda, Normalize
from torchvision.transforms.functional import InterpolationMode

import folder_paths


_SEEDVR2_INVALID_MODEL_MSG_PREFIX = (
    "SeedVR2Conditioning: model object does not match expected SeedVR2 structure"
)

# Private sentinel for getattr default: distinguishes "attribute missing"
# from "attribute present but None" so the failure message is accurate.
_ATTR_MISSING = object()

_SEEDVR2_PROMPT_EMBED_CANDIDATES = {
    "positive": [
        ("embeddings", "SEEDVR2/pos_emb.pt"),
        ("embeddings", "pos_emb.pt"),
        ("custom_nodes", "ComfyUI-SeedVR2-Canonical/vendor/SeedVR/pos_emb.pt"),
        ("custom_nodes", "ComfyUI-SeedVR2_VideoUpscaler/pos_emb.pt"),
    ],
    "negative": [
        ("embeddings", "SEEDVR2/neg_emb.pt"),
        ("embeddings", "neg_emb.pt"),
        ("custom_nodes", "ComfyUI-SeedVR2-Canonical/vendor/SeedVR/neg_emb.pt"),
        ("custom_nodes", "ComfyUI-SeedVR2_VideoUpscaler/neg_emb.pt"),
    ],
}


def _load_seedvr2_prompt_embed(kind, device, dtype):
    for folder_name, filename in _SEEDVR2_PROMPT_EMBED_CANDIDATES[kind]:
        path = folder_paths.get_full_path(folder_name, filename)
        if path is not None:
            embed = torch.load(path, map_location="cpu", weights_only=True)
            if not torch.is_tensor(embed) or embed.ndim != 2:
                raise RuntimeError(
                    f"SeedVR2Conditioning: {kind} prompt embedding at {path} "
                    f"must be a rank-2 torch.Tensor; got {type(embed).__name__} "
                    f"with shape {getattr(embed, 'shape', None)}."
                )
            return embed.to(device=device, dtype=dtype)
    searched = ", ".join(
        f"{folder_name}:{filename}"
        for folder_name, filename in _SEEDVR2_PROMPT_EMBED_CANDIDATES[kind]
    )
    raise FileNotFoundError(
        f"SeedVR2Conditioning: missing {kind} prompt embedding. "
        f"Searched {searched}."
    )


def _load_seedvr2_prompt_embeds(device, dtype):
    return (
        _load_seedvr2_prompt_embed("positive", device, dtype),
        _load_seedvr2_prompt_embed("negative", device, dtype),
    )


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
                io.Int.Input("temporal_tile_size", default=16, min=1, max=16384, step=4),
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
            latent = tiled_vae(images_bcthw, vae_model, **args, encode=True)
        else:
            vae_model.img_dims = [o_h, o_w]
            vae_model.original_image_video = original_image_video
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
        pos_cond, neg_cond = _load_seedvr2_prompt_embeds(device, model.positive_conditioning.dtype)

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

class SeedVRExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            SeedVR2Conditioning,
            SeedVR2InputProcessing
        ]

async def comfy_entrypoint() -> SeedVRExtension:
    return SeedVRExtension()
