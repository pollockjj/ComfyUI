import logging
from typing import Any, Optional

from comfy.isolation.proxies.base import (
    IS_CHILD_PROCESS,
    BaseProxy,
    BaseRegistry,
    detach_if_grad,
    get_thread_loop,
    run_coro_in_new_loop,
)

logger = logging.getLogger(__name__)


class CLIPRegistry(BaseRegistry[Any]):
    _type_prefix = "clip"

    async def get_ram_usage(self, instance_id: str) -> int:
        clip = self._get_instance(instance_id)
        return clip.get_ram_usage()

    async def clip_layer(self, instance_id: str, layer_idx: int) -> None:
        clip = self._get_instance(instance_id)
        clip.clip_layer(layer_idx)

    async def set_tokenizer_option(self, instance_id: str, option_name: str, value: Any) -> None:
        clip = self._get_instance(instance_id)
        clip.set_tokenizer_option(option_name, value)

    async def tokenize(self, instance_id: str, text: str, return_word_ids: bool = False, **kwargs: Any) -> Any:
        clip = self._get_instance(instance_id)
        return clip.tokenize(text, return_word_ids=return_word_ids, **kwargs)

    async def encode(self, instance_id: str, text: str) -> Any:
        clip = self._get_instance(instance_id)
        return detach_if_grad(clip.encode(text))

    async def encode_from_tokens(
        self, instance_id: str, tokens: Any, return_pooled: bool = False, return_dict: bool = False
    ) -> Any:
        clip = self._get_instance(instance_id)
        return detach_if_grad(
            clip.encode_from_tokens(tokens, return_pooled=return_pooled, return_dict=return_dict)
        )

    async def encode_from_tokens_scheduled(
        self,
        instance_id: str,
        tokens: Any,
        unprojected: bool = False,
        add_dict: Optional[dict] = None,
        show_pbar: bool = True,
    ) -> Any:
        clip = self._get_instance(instance_id)
        add_dict = add_dict or {}
        return detach_if_grad(
            clip.encode_from_tokens_scheduled(
                tokens, unprojected=unprojected, add_dict=add_dict, show_pbar=show_pbar
            )
        )

    async def add_patches(
        self, instance_id: str, patches: Any, strength_patch: float = 1.0, strength_model: float = 1.0
    ) -> Any:
        clip = self._get_instance(instance_id)
        return clip.add_patches(patches, strength_patch=strength_patch, strength_model=strength_model)

    async def get_key_patches(self, instance_id: str) -> Any:
        clip = self._get_instance(instance_id)
        return clip.get_key_patches()

    async def load_sd(self, instance_id: str, sd: dict, full_model: bool = False) -> Any:
        clip = self._get_instance(instance_id)
        return clip.load_sd(sd, full_model=full_model)

    async def get_sd(self, instance_id: str) -> Any:
        clip = self._get_instance(instance_id)
        return clip.get_sd()

    async def clone(self, instance_id: str) -> str:
        clip = self._get_instance(instance_id)
        new_clip = clip.clone()
        return self.register(new_clip)


class CLIPProxy(BaseProxy[CLIPRegistry]):
    _registry_class = CLIPRegistry
    __module__ = "comfy.sd"

    def get_ram_usage(self) -> int:
        return self._call_rpc("get_ram_usage")

    def clip_layer(self, layer_idx: int) -> None:
        return self._call_rpc("clip_layer", layer_idx)

    def set_tokenizer_option(self, option_name: str, value: Any) -> None:
        return self._call_rpc("set_tokenizer_option", option_name, value)

    def tokenize(self, text: str, return_word_ids: bool = False, **kwargs: Any) -> Any:
        return self._call_rpc("tokenize", text, return_word_ids=return_word_ids, **kwargs)

    def encode(self, text: str) -> Any:
        return self._call_rpc("encode", text)

    def encode_from_tokens(
        self, tokens: Any, return_pooled: bool = False, return_dict: bool = False
    ) -> Any:
        return self._call_rpc("encode_from_tokens", tokens, return_pooled=return_pooled, return_dict=return_dict)

    def encode_from_tokens_scheduled(
        self,
        tokens: Any,
        unprojected: bool = False,
        add_dict: Optional[dict] = None,
        show_pbar: bool = True,
    ) -> Any:
        add_dict = add_dict or {}
        return self._call_rpc(
            "encode_from_tokens_scheduled",
            tokens,
            unprojected=unprojected,
            add_dict=add_dict,
            show_pbar=show_pbar,
        )

    def add_patches(self, patches: Any, strength_patch: float = 1.0, strength_model: float = 1.0) -> Any:
        return self._call_rpc("add_patches", patches, strength_patch=strength_patch, strength_model=strength_model)

    def get_key_patches(self) -> Any:
        return self._call_rpc("get_key_patches")

    def load_sd(self, sd: dict, full_model: bool = False) -> Any:
        return self._call_rpc("load_sd", sd, full_model=full_model)

    def get_sd(self) -> Any:
        return self._call_rpc("get_sd")

    def clone(self) -> "CLIPProxy":
        new_id = self._call_rpc("clone")
        return CLIPProxy(new_id, self._registry, manage_lifecycle=not IS_CHILD_PROCESS)


# Registry instantiated in host_hooks.initialize_host_process; keep optional safety
if not IS_CHILD_PROCESS:
    _CLIP_REGISTRY_SINGLETON = CLIPRegistry()

