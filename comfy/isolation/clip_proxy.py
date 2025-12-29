"""
Stateless RPC pattern for CLIP instances.
Inherits from BaseRegistry/BaseProxy for standardized isolation.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from comfy.isolation.proxies.base import (
    IS_CHILD_PROCESS,
    BaseProxy,
    BaseRegistry,
    detach_if_grad,
)
from comfy.isolation.model_patcher_proxy import ModelPatcherProxy

logger = logging.getLogger(__name__)


class CLIPRegistry(BaseRegistry[Any]):
    _type_prefix = "clip"

    # =========================================================================
    # Core RPC Methods
    # =========================================================================

    async def get_ram_usage(self, instance_id: str) -> int:
        clip = self._get_instance(instance_id)
        return clip.get_ram_usage()

    async def get_patcher_id(self, instance_id: str) -> str:
        clip = self._get_instance(instance_id)
        # Ensure the associated ModelPatcher is registered in its own registry
        from comfy.isolation.model_patcher_proxy import ModelPatcherRegistry
        mp_registry = ModelPatcherRegistry()
        return mp_registry.register(clip.patcher)

    async def load_model(self, instance_id: str) -> None:
        clip = self._get_instance(instance_id)
        clip.load_model()
        # Return None; Proxy handles patcher access via dedicated property
        return None

    async def clip_layer(self, instance_id: str, layer_idx: int) -> None:
        clip = self._get_instance(instance_id)
        clip.clip_layer(layer_idx)

    # =========================================================================
    # Tokenizer / Text Encoding
    # =========================================================================

    async def set_tokenizer_option(self, instance_id: str, option_name: str, value: Any) -> None:
        clip = self._get_instance(instance_id)
        clip.set_tokenizer_option(option_name, value)

    async def get_property(self, instance_id: str, name: str) -> Any:
        clip = self._get_instance(instance_id)
        return getattr(clip, name)

    async def set_property(self, instance_id: str, name: str, value: Any) -> None:
        clip = self._get_instance(instance_id)
        setattr(clip, name, value)

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

    # =========================================================================
    # Patching / State
    # =========================================================================

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

    @property
    def patcher(self) -> ModelPatcherProxy:
        if not hasattr(self, "_patcher_proxy"):
            # Lazy load the patcher proxy
            patcher_id = self._call_rpc("get_patcher_id")
            self._patcher_proxy = ModelPatcherProxy(patcher_id, manage_lifecycle=False)
        return self._patcher_proxy

    @patcher.setter
    def patcher(self, value: Any) -> None:
        if isinstance(value, ModelPatcherProxy):
            self._patcher_proxy = value
        else:
            logger.warning(f"Attempted to set CLIPProxy.patcher to non-proxy object: {value}")

    def load_model(self) -> ModelPatcherProxy:
        self._call_rpc("load_model")
        return self.patcher

    @property
    def layer_idx(self) -> Optional[int]:
        return self._call_rpc("get_property", "layer_idx")

    @layer_idx.setter
    def layer_idx(self, value: Optional[int]) -> None:
        self._call_rpc("set_property", "layer_idx", value)

    @property
    def tokenizer_options(self) -> dict:
         return self._call_rpc("get_property", "tokenizer_options")

    @tokenizer_options.setter
    def tokenizer_options(self, value: dict) -> None:
        self._call_rpc("set_property", "tokenizer_options", value)

    @property
    def use_clip_schedule(self) -> bool:
        return self._call_rpc("get_property", "use_clip_schedule")

    @use_clip_schedule.setter
    def use_clip_schedule(self, value: bool) -> None:
        self._call_rpc("set_property", "use_clip_schedule", value)

    @property
    def apply_hooks_to_conds(self) -> Any:
        return self._call_rpc("get_property", "apply_hooks_to_conds")

    @apply_hooks_to_conds.setter
    def apply_hooks_to_conds(self, value: Any) -> None:
        self._call_rpc("set_property", "apply_hooks_to_conds", value)

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
        res = self._call_rpc("encode_from_tokens", tokens, return_pooled=return_pooled, return_dict=return_dict)
        # Rehydrate tuple if needed (RPC converts tuples to lists)
        if return_pooled and isinstance(res, list) and not return_dict:
            return tuple(res)
        return res

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

    def clone(self) -> CLIPProxy:
        new_id = self._call_rpc("clone")
        return CLIPProxy(new_id, self._registry, manage_lifecycle=not IS_CHILD_PROCESS)


# Registry instantiated in host_hooks.initialize_host_process; keep optional safety
if not IS_CHILD_PROCESS:
    _CLIP_REGISTRY_SINGLETON = CLIPRegistry()
