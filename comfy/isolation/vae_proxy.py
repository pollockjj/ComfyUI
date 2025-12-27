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

    async def get_sd(self, instance_id: str) -> Any:
        vae = self._get_instance(instance_id)
        return vae.get_sd()


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


if not IS_CHILD_PROCESS:
    _VAE_REGISTRY_SINGLETON = VAERegistry()


