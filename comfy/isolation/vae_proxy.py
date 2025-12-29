import logging
from typing import Any

from comfy.isolation.proxies.base import (
    IS_CHILD_PROCESS,
    BaseProxy,
    BaseRegistry,
    detach_if_grad,
)

logger = logging.getLogger(__name__)


class VAERegistry(BaseRegistry[Any]):
    _type_prefix = "vae"

    async def encode(self, instance_id: str, pixels: Any) -> Any:
        vae = self._get_instance(instance_id)
        return detach_if_grad(vae.encode(pixels))

    async def encode_tiled(
        self, instance_id: str, pixels: Any, tile_x: int = 512, tile_y: int = 512, overlap: int = 64
    ) -> Any:
        vae = self._get_instance(instance_id)
        return detach_if_grad(vae.encode_tiled(pixels, tile_x=tile_x, tile_y=tile_y, overlap=overlap))

    async def decode(self, instance_id: str, samples: Any, **kwargs: Any) -> Any:
        vae = self._get_instance(instance_id)
        return detach_if_grad(vae.decode(samples, **kwargs))

    async def decode_tiled(
        self, instance_id: str, samples: Any, tile_x: int = 64, tile_y: int = 64, overlap: int = 16, **kwargs: Any
    ) -> Any:
        vae = self._get_instance(instance_id)
        return detach_if_grad(vae.decode_tiled(samples, tile_x=tile_x, tile_y=tile_y, overlap=overlap, **kwargs))

    async def get_property(self, instance_id: str, name: str) -> Any:
        vae = self._get_instance(instance_id)
        return getattr(vae, name)

    async def memory_used_encode(self, instance_id: str, shape: Any, dtype: Any) -> int:
        vae = self._get_instance(instance_id)
        return vae.memory_used_encode(shape, dtype)

    async def memory_used_decode(self, instance_id: str, shape: Any, dtype: Any) -> int:
        vae = self._get_instance(instance_id)
        return vae.memory_used_decode(shape, dtype)

    async def process_input(self, instance_id: str, image: Any) -> Any:
        vae = self._get_instance(instance_id)
        return detach_if_grad(vae.process_input(image))

    async def process_output(self, instance_id: str, image: Any) -> Any:
        vae = self._get_instance(instance_id)
        return detach_if_grad(vae.process_output(image))


class VAEProxy(BaseProxy[VAERegistry]):
    _registry_class = VAERegistry
    __module__ = "comfy.sd"

    def encode(self, pixels: Any) -> Any:
        return self._call_rpc("encode", pixels)

    def encode_tiled(self, pixels: Any, tile_x: int = 512, tile_y: int = 512, overlap: int = 64) -> Any:
        return self._call_rpc("encode_tiled", pixels, tile_x, tile_y, overlap)

    def decode(self, samples: Any, **kwargs: Any) -> Any:
        return self._call_rpc("decode", samples, **kwargs)

    def decode_tiled(
        self, samples: Any, tile_x: int = 64, tile_y: int = 64, overlap: int = 16, **kwargs: Any
    ) -> Any:
        return self._call_rpc("decode_tiled", samples, tile_x, tile_y, overlap, **kwargs)

    def get_sd(self) -> Any:
        return self._call_rpc("get_sd")

    # Wrapper for property access
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
    def check_not_vide(self) -> bool:
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

    def memory_used_encode(self, shape: Any, dtype: Any) -> int:
        return self._call_rpc("memory_used_encode", shape, dtype)

    def memory_used_decode(self, shape: Any, dtype: Any) -> int:
        return self._call_rpc("memory_used_decode", shape, dtype)

    def process_input(self, image: Any) -> Any:
        return self._call_rpc("process_input", image)

    def process_output(self, image: Any) -> Any:
        return self._call_rpc("process_output", image)


if not IS_CHILD_PROCESS:
    _VAE_REGISTRY_SINGLETON = VAERegistry()


