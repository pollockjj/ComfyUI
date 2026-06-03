# pylint: disable=attribute-defined-outside-init
import logging
from typing import Any

from comfy.isolation.proxies.base import (
    IS_CHILD_PROCESS,
    BaseProxy,
    BaseRegistry,
    detach_if_grad,
)
from comfy.isolation.model_patcher_proxy import ModelPatcherProxy, ModelPatcherRegistry

logger = logging.getLogger(__name__)


class FirstStageModelRegistry(BaseRegistry[Any]):
    _type_prefix = "first_stage_model"

    async def get_property(self, instance_id: str, name: str) -> Any:
        obj = self._get_instance(instance_id)
        return getattr(obj, name)

    async def has_property(self, instance_id: str, name: str) -> bool:
        obj = self._get_instance(instance_id)
        return hasattr(obj, name)


class FirstStageModelProxy(BaseProxy[FirstStageModelRegistry]):
    _registry_class = FirstStageModelRegistry
    __module__ = "comfy.ldm.models.autoencoder"

    def __getattr__(self, name: str) -> Any:
        try:
            return self._call_rpc("get_property", name)
        except Exception as e:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'"
            ) from e

    def __repr__(self) -> str:
        return f"<FirstStageModelProxy {self._instance_id}>"


class VAERegistry(BaseRegistry[Any]):
    _type_prefix = "vae"

    async def get_patcher_id(self, instance_id: str) -> str:
        vae = self._get_instance(instance_id)
        return ModelPatcherRegistry().register(vae.patcher)

    async def get_first_stage_model_id(self, instance_id: str) -> str:
        vae = self._get_instance(instance_id)
        return FirstStageModelRegistry().register(vae.first_stage_model)

    async def encode(self, instance_id: str, pixels: Any) -> Any:
        return detach_if_grad(self._get_instance(instance_id).encode(pixels))

    async def encode_tiled(
        self,
        instance_id: str,
        pixels: Any,
        tile_x: int = 512,
        tile_y: int = 512,
        overlap: int = 64,
    ) -> Any:
        return detach_if_grad(
            self._get_instance(instance_id).encode_tiled(
                pixels, tile_x=tile_x, tile_y=tile_y, overlap=overlap
            )
        )

    async def model_size(self, instance_id: str) -> Any:
        return self._get_instance(instance_id).model_size()

    async def throw_exception_if_invalid(self, instance_id: str) -> None:
        self._get_instance(instance_id).throw_exception_if_invalid()

    async def vae_encode_crop_pixels(self, instance_id: str, pixels: Any) -> Any:
        return detach_if_grad(
            self._get_instance(instance_id).vae_encode_crop_pixels(pixels)
        )

    async def vae_output_dtype(self, instance_id: str) -> Any:
        return self._get_instance(instance_id).vae_output_dtype()

    async def decode_tiled_(
        self,
        instance_id: str,
        samples: Any,
        tile_x: int = 64,
        tile_y: int = 64,
        overlap: int = 16,
    ) -> Any:
        return detach_if_grad(
            self._get_instance(instance_id).decode_tiled_(
                samples, tile_x=tile_x, tile_y=tile_y, overlap=overlap
            )
        )

    async def decode_tiled_1d(
        self, instance_id: str, samples: Any, tile_x: int = 256, overlap: int = 32
    ) -> Any:
        return detach_if_grad(
            self._get_instance(instance_id).decode_tiled_1d(
                samples, tile_x=tile_x, overlap=overlap
            )
        )

    async def decode_tiled_3d(
        self,
        instance_id: str,
        samples: Any,
        tile_t: int = 999,
        tile_x: int = 32,
        tile_y: int = 32,
        overlap: Any = (1, 8, 8),
    ) -> Any:
        return detach_if_grad(
            self._get_instance(instance_id).decode_tiled_3d(
                samples, tile_t=tile_t, tile_x=tile_x, tile_y=tile_y, overlap=overlap
            )
        )

    async def encode_tiled_(
        self,
        instance_id: str,
        pixel_samples: Any,
        tile_x: int = 512,
        tile_y: int = 512,
        overlap: int = 64,
    ) -> Any:
        return detach_if_grad(
            self._get_instance(instance_id).encode_tiled_(
                pixel_samples, tile_x=tile_x, tile_y=tile_y, overlap=overlap
            )
        )

    async def encode_tiled_1d(
        self,
        instance_id: str,
        samples: Any,
        tile_x: int = 256 * 2048,
        overlap: int = 64 * 2048,
    ) -> Any:
        return detach_if_grad(
            self._get_instance(instance_id).encode_tiled_1d(
                samples, tile_x=tile_x, overlap=overlap
            )
        )

    async def encode_tiled_3d(
        self,
        instance_id: str,
        samples: Any,
        tile_t: int = 9999,
        tile_x: int = 512,
        tile_y: int = 512,
        overlap: Any = (1, 64, 64),
    ) -> Any:
        return detach_if_grad(
            self._get_instance(instance_id).encode_tiled_3d(
                samples, tile_t=tile_t, tile_x=tile_x, tile_y=tile_y, overlap=overlap
            )
        )

    async def decode(self, instance_id: str, samples: Any, **kwargs: Any) -> Any:
        return detach_if_grad(self._get_instance(instance_id).decode(samples, **kwargs))

    async def decode_tiled(
        self,
        instance_id: str,
        samples: Any,
        tile_x: int = 64,
        tile_y: int = 64,
        overlap: int = 16,
        **kwargs: Any,
    ) -> Any:
        return detach_if_grad(
            self._get_instance(instance_id).decode_tiled(
                samples, tile_x=tile_x, tile_y=tile_y, overlap=overlap, **kwargs
            )
        )

    async def get_property(self, instance_id: str, name: str) -> Any:
        return getattr(self._get_instance(instance_id), name)

    async def get_sd(self, instance_id: str) -> Any:
        return self._get_instance(instance_id).get_sd()

    async def memory_used_encode(self, instance_id: str, shape: Any, dtype: Any) -> int:
        return self._get_instance(instance_id).memory_used_encode(shape, dtype)

    async def memory_used_decode(self, instance_id: str, shape: Any, dtype: Any) -> int:
        return self._get_instance(instance_id).memory_used_decode(shape, dtype)

    async def process_input(self, instance_id: str, image: Any) -> Any:
        return detach_if_grad(self._get_instance(instance_id).process_input(image))

    async def process_output(self, instance_id: str, image: Any) -> Any:
        return detach_if_grad(self._get_instance(instance_id).process_output(image))

    async def spacial_compression_decode(self, instance_id: str) -> Any:
        return self._get_instance(instance_id).spacial_compression_decode()

    async def spacial_compression_encode(self, instance_id: str) -> Any:
        return self._get_instance(instance_id).spacial_compression_encode()

    async def temporal_compression_decode(self, instance_id: str) -> Any:
        return self._get_instance(instance_id).temporal_compression_decode()


class VAEProxy(BaseProxy[VAERegistry]):
    _registry_class = VAERegistry
    __module__ = "comfy.sd"

    @property
    def patcher(self) -> ModelPatcherProxy:
        if not hasattr(self, "_patcher_proxy"):
            patcher_id = self._call_rpc("get_patcher_id")
            self._patcher_proxy = ModelPatcherProxy(patcher_id, manage_lifecycle=False)
        return self._patcher_proxy

    @property
    def first_stage_model(self) -> FirstStageModelProxy:
        if not hasattr(self, "_first_stage_model_proxy"):
            fsm_id = self._call_rpc("get_first_stage_model_id")
            self._first_stage_model_proxy = FirstStageModelProxy(
                fsm_id, manage_lifecycle=False
            )
        return self._first_stage_model_proxy

    @property
    def vae_dtype(self) -> Any:
        return self._get_property("vae_dtype")

    def model_size(self) -> Any:
        return self._call_rpc("model_size")

    def throw_exception_if_invalid(self) -> None:
        self._call_rpc("throw_exception_if_invalid")

    def vae_encode_crop_pixels(self, pixels: Any) -> Any:
        return self._call_rpc("vae_encode_crop_pixels", pixels)

    def vae_output_dtype(self) -> Any:
        return self._call_rpc("vae_output_dtype")

    def decode_tiled_(
        self, samples: Any, tile_x: int = 64, tile_y: int = 64, overlap: int = 16
    ) -> Any:
        return self._call_rpc("decode_tiled_", samples, tile_x, tile_y, overlap)

    def decode_tiled_1d(
        self, samples: Any, tile_x: int = 256, overlap: int = 32
    ) -> Any:
        return self._call_rpc("decode_tiled_1d", samples, tile_x, overlap)

    def decode_tiled_3d(
        self,
        samples: Any,
        tile_t: int = 999,
        tile_x: int = 32,
        tile_y: int = 32,
        overlap: Any = (1, 8, 8),
    ) -> Any:
        return self._call_rpc(
            "decode_tiled_3d", samples, tile_t, tile_x, tile_y, overlap
        )

    def encode_tiled_(
        self, pixel_samples: Any, tile_x: int = 512, tile_y: int = 512, overlap: int = 64
    ) -> Any:
        return self._call_rpc("encode_tiled_", pixel_samples, tile_x, tile_y, overlap)

    def encode_tiled_1d(
        self,
        samples: Any,
        tile_x: int = 256 * 2048,
        overlap: int = 64 * 2048,
    ) -> Any:
        return self._call_rpc("encode_tiled_1d", samples, tile_x, overlap)

    def encode_tiled_3d(
        self,
        samples: Any,
        tile_t: int = 9999,
        tile_x: int = 512,
        tile_y: int = 512,
        overlap: Any = (1, 64, 64),
    ) -> Any:
        return self._call_rpc(
            "encode_tiled_3d", samples, tile_t, tile_x, tile_y, overlap
        )

    def encode(self, pixels: Any) -> Any:
        return self._call_rpc("encode", pixels)

    def encode_tiled(
        self, pixels: Any, tile_x: int = 512, tile_y: int = 512, overlap: int = 64
    ) -> Any:
        return self._call_rpc("encode_tiled", pixels, tile_x, tile_y, overlap)

    def decode(self, samples: Any, **kwargs: Any) -> Any:
        return self._call_rpc("decode", samples, **kwargs)

    def decode_tiled(
        self,
        samples: Any,
        tile_x: int = 64,
        tile_y: int = 64,
        overlap: int = 16,
        **kwargs: Any,
    ) -> Any:
        return self._call_rpc(
            "decode_tiled", samples, tile_x, tile_y, overlap, **kwargs
        )

    def get_sd(self) -> Any:
        return self._call_rpc("get_sd")

    def _get_property(self, name: str) -> Any:
        return self._call_rpc("get_property", name)

    @property
    def latent_dim(self) -> int:
        return self._get_property("latent_dim")

    @property
    def latent_channels(self) -> int:
        return self._get_property("latent_channels")

    @property
    def downscale_ratio(self) -> Any:
        return self._get_property("downscale_ratio")

    @property
    def upscale_ratio(self) -> Any:
        return self._get_property("upscale_ratio")

    @property
    def output_channels(self) -> int:
        return self._get_property("output_channels")

    @property
    def not_video(self) -> bool:
        return self._get_property("not_video")

    @property
    def device(self) -> Any:
        return self._get_property("device")

    @property
    def working_dtypes(self) -> Any:
        return self._get_property("working_dtypes")

    @property
    def disable_offload(self) -> bool:
        return self._get_property("disable_offload")

    @property
    def size(self) -> Any:
        return self._get_property("size")

    @property
    def audio_sample_rate(self) -> Any:
        return self._get_property("audio_sample_rate")

    @property
    def audio_sample_rate_output(self) -> Any:
        return self._get_property("audio_sample_rate_output")

    @property
    def autoencoder(self) -> Any:
        return self._get_property("autoencoder")

    @property
    def conv_out_channels(self) -> Any:
        return self._get_property("conv_out_channels")

    @property
    def crop_input(self) -> Any:
        return self._get_property("crop_input")

    @property
    def downscale_index_formula(self) -> Any:
        return self._get_property("downscale_index_formula")

    @property
    def extra_1d_channel(self) -> Any:
        return self._get_property("extra_1d_channel")

    @property
    def output_device(self) -> Any:
        return self._get_property("output_device")

    @property
    def pad_channel_value(self) -> Any:
        return self._get_property("pad_channel_value")

    @property
    def upscale_index_formula(self) -> Any:
        return self._get_property("upscale_index_formula")

    def memory_used_encode(self, shape: Any, dtype: Any) -> int:
        return self._call_rpc("memory_used_encode", shape, dtype)

    def memory_used_decode(self, shape: Any, dtype: Any) -> int:
        return self._call_rpc("memory_used_decode", shape, dtype)

    def process_input(self, image: Any) -> Any:
        return self._call_rpc("process_input", image)

    def process_output(self, image: Any) -> Any:
        return self._call_rpc("process_output", image)

    def spacial_compression_decode(self) -> Any:
        return self._call_rpc("spacial_compression_decode")

    def spacial_compression_encode(self) -> Any:
        return self._call_rpc("spacial_compression_encode")

    def temporal_compression_decode(self) -> Any:
        return self._call_rpc("temporal_compression_decode")


if not IS_CHILD_PROCESS:
    _VAE_REGISTRY_SINGLETON = VAERegistry()
    _FIRST_STAGE_MODEL_REGISTRY_SINGLETON = FirstStageModelRegistry()
