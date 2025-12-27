"""
Stateless RPC pattern for ModelPatcher instances.
Inherits from BaseRegistry/BaseProxy for standardized isolation.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Optional, List, Set, Dict

# Try to import AutoPatcherEjector for use_ejected; fallback if not available locally
try:
    from comfy.model_patcher import AutoPatcherEjector
except ImportError:
    class AutoPatcherEjector:
        def __init__(self, model, skip_and_inject_on_exit_only=False):
            self.model = model
            self.skip_and_inject_on_exit_only = skip_and_inject_on_exit_only
            self.prev_skip_injection = False
            self.was_injected = False
        def __enter__(self):
            self.was_injected = False
            self.prev_skip_injection = self.model.skip_injection
            if self.skip_and_inject_on_exit_only:
                self.model.skip_injection = True
            if self.model.is_injected:
                self.model.eject_model()
                self.was_injected = True
        def __exit__(self, *args):
            if self.skip_and_inject_on_exit_only:
                self.model.skip_injection = self.prev_skip_injection
                self.model.inject_model()
            if self.was_injected and not self.model.skip_injection:
                self.model.inject_model()
            self.model.skip_injection = self.prev_skip_injection

from comfy.isolation.proxies.base import (
    IS_CHILD_PROCESS,
    BaseProxy,
    BaseRegistry,
    detach_if_grad,
)

logger = logging.getLogger(__name__)


class ModelPatcherRegistry(BaseRegistry[Any]):
    _type_prefix = "model"

    # =========================================================================
    # Core RPC Methods
    # =========================================================================

    async def clone(self, instance_id: str) -> str:
        instance = self._get_instance(instance_id)
        new_model = instance.clone()
        return self.register(new_model)

    async def is_clone(self, instance_id: str, other: Any) -> bool:
        instance = self._get_instance(instance_id)
        if hasattr(other, "model"):
            return instance.is_clone(other)
        return False
        
    async def clone_has_same_weights_by_id(self, instance_id: str, other_id: str) -> bool:
        instance = self._get_instance(instance_id)
        other = self._get_instance(other_id)
        return instance.clone_has_same_weights(other)

    async def get_model_object(self, instance_id: str, name: str) -> Any:
        instance = self._get_instance(instance_id)
        result = instance.get_model_object(name)
        if name == "model_sampling":
            from comfy.isolation.model_sampling_proxy import ModelSamplingRegistry, ModelSamplingProxy
            registry = ModelSamplingRegistry()
            sampling_id = registry.register(result)
            return ModelSamplingProxy(sampling_id, registry)
        return detach_if_grad(result)

    async def get_model_options(self, instance_id: str) -> dict:
        instance = self._get_instance(instance_id)
        import copy
        return copy.deepcopy(instance.model_options)

    async def set_model_options(self, instance_id: str, options: dict) -> None:
        self._get_instance(instance_id).model_options = options

    # =========================================================================
    # Device / Memory / Properties
    # =========================================================================

    async def get_load_device(self, instance_id: str) -> Any:
        return self._get_instance(instance_id).load_device

    async def get_offload_device(self, instance_id: str) -> Any:
        return self._get_instance(instance_id).offload_device

    async def current_loaded_device(self, instance_id: str) -> Any:
        return self._get_instance(instance_id).current_loaded_device()

    async def get_size(self, instance_id: str) -> int:
        return self._get_instance(instance_id).size

    async def model_size(self, instance_id: str) -> Any:
        return self._get_instance(instance_id).model_size()

    async def loaded_size(self, instance_id: str) -> Any:
        return self._get_instance(instance_id).loaded_size()
        
    async def get_ram_usage(self, instance_id: str) -> int:
        return self._get_instance(instance_id).get_ram_usage()

    async def lowvram_patch_counter(self, instance_id: str) -> int:
        return self._get_instance(instance_id).lowvram_patch_counter()
        
    async def memory_required(self, instance_id: str, input_shape: Any) -> Any:
        return self._get_instance(instance_id).memory_required(input_shape)

    async def model_dtype(self, instance_id: str) -> Any:
        return self._get_instance(instance_id).model_dtype()

    # =========================================================================
    # Load / Patch / State
    # =========================================================================

    async def model_patches_to(self, instance_id: str, device: Any) -> Any:
        return self._get_instance(instance_id).model_patches_to(device)

    async def partially_load(self, instance_id: str, device: Any, extra_memory: Any, force_patch_weights: bool = False) -> Any:
        return self._get_instance(instance_id).partially_load(device, extra_memory, force_patch_weights=force_patch_weights)
        
    async def partially_unload(self, instance_id: str, device_to: Any, memory_to_free: int = 0, force_patch_weights: bool = False) -> int:
        return self._get_instance(instance_id).partially_unload(device_to, memory_to_free, force_patch_weights)
    
    async def load(self, instance_id: str, device_to: Any = None, lowvram_model_memory: int = 0, force_patch_weights: bool = False, full_load: bool = False) -> None:
        self._get_instance(instance_id).load(device_to, lowvram_model_memory, force_patch_weights, full_load)
        
    async def patch_model(self, instance_id: str, device_to: Any = None, lowvram_model_memory: int = 0, load_weights: bool = True, force_patch_weights: bool = False) -> None:
        self._get_instance(instance_id).patch_model(device_to, lowvram_model_memory, load_weights, force_patch_weights)
        
    async def unpatch_model(self, instance_id: str, device_to: Any = None, unpatch_weights: bool = True) -> None:
        self._get_instance(instance_id).unpatch_model(device_to, unpatch_weights)
        
    async def detach(self, instance_id: str, unpatch_all: bool = True) -> None:
        self._get_instance(instance_id).detach(unpatch_all)

    async def prepare_state(self, instance_id: str, timestep: Any) -> Any:
        instance = self._get_instance(instance_id)
        cp = getattr(instance.model, "current_patcher", instance)
        return cp.prepare_state(timestep)

    async def pre_run(self, instance_id: str) -> None:
        self._get_instance(instance_id).pre_run()

    async def cleanup(self, instance_id: str) -> None:
        self._get_instance(instance_id).cleanup()

    # =========================================================================
    # Hook / Injection Management
    # =========================================================================

    async def apply_hooks(self, instance_id: str, hooks: Any) -> Any:
        instance = self._get_instance(instance_id)
        cp = getattr(instance.model, "current_patcher", instance)
        return cp.apply_hooks(hooks=hooks)
        
    async def clean_hooks(self, instance_id: str) -> None:
        self._get_instance(instance_id).clean_hooks()

    async def restore_hook_patches(self, instance_id: str) -> None:
        self._get_instance(instance_id).restore_hook_patches()
        
    async def unpatch_hooks(self, instance_id: str, whitelist_keys_set: Optional[set] = None) -> None:
        self._get_instance(instance_id).unpatch_hooks(whitelist_keys_set)

    async def register_all_hook_patches(self, instance_id: str, hooks: Any, target_dict: Any, model_options: Any, registered: Any) -> None:
        from types import SimpleNamespace
        import comfy.hooks
        instance = self._get_instance(instance_id)
        # Handle SimpleNamespace hooks from RPC
        if isinstance(hooks, SimpleNamespace):
             hooks = comfy.hooks.HookGroup() if not getattr(hooks, 'hooks', None) else hooks
             if hasattr(hooks, 'hooks') and hooks.hooks:
                 logger.warning("Skipping register_all_hook_patches: hooks came as SimpleNamespace")
                 return
        instance.register_all_hook_patches(hooks, target_dict, model_options, registered)

    async def get_hook_mode(self, instance_id: str) -> Any:
        return getattr(self._get_instance(instance_id), "hook_mode", None)

    async def set_hook_mode(self, instance_id: str, value: Any) -> None:
        setattr(self._get_instance(instance_id), "hook_mode", value)

    async def inject_model(self, instance_id: str) -> None:
        self._get_instance(instance_id).inject_model()

    async def eject_model(self, instance_id: str) -> None:
        self._get_instance(instance_id).eject_model()
        
    async def get_is_injected(self, instance_id: str) -> bool:
        return self._get_instance(instance_id).is_injected
        
    async def set_skip_injection(self, instance_id: str, value: bool) -> None:
        self._get_instance(instance_id).skip_injection = value

    async def get_skip_injection(self, instance_id: str) -> bool:
        return self._get_instance(instance_id).skip_injection

    # =========================================================================
    # Configuration Setters (Easy RPCs)
    # =========================================================================

    async def set_model_sampler_cfg_function(self, instance_id: str, sampler_cfg_function: Any, disable_cfg1_optimization: bool = False) -> None:
        self._get_instance(instance_id).set_model_sampler_cfg_function(sampler_cfg_function, disable_cfg1_optimization)

    async def set_model_sampler_post_cfg_function(self, instance_id: str, post_cfg_function: Any, disable_cfg1_optimization: bool = False) -> None:
        self._get_instance(instance_id).set_model_sampler_post_cfg_function(post_cfg_function, disable_cfg1_optimization)

    async def set_model_sampler_pre_cfg_function(self, instance_id: str, pre_cfg_function: Any, disable_cfg1_optimization: bool = False) -> None:
        self._get_instance(instance_id).set_model_sampler_pre_cfg_function(pre_cfg_function, disable_cfg1_optimization)
    
    async def set_model_sampler_calc_cond_batch_function(self, instance_id: str, fn: Any) -> None:
        self._get_instance(instance_id).set_model_sampler_calc_cond_batch_function(fn)

    async def set_model_unet_function_wrapper(self, instance_id: str, unet_wrapper_function: Any) -> None:
        self._get_instance(instance_id).set_model_unet_function_wrapper(unet_wrapper_function)

    async def set_model_denoise_mask_function(self, instance_id: str, denoise_mask_function: Any) -> None:
        self._get_instance(instance_id).set_model_denoise_mask_function(denoise_mask_function)

    async def set_model_patch(self, instance_id: str, patch: Any, name: str) -> None:
        self._get_instance(instance_id).set_model_patch(patch, name)

    async def set_model_patch_replace(self, instance_id: str, patch: Any, name: str, block_name: str, number: int, transformer_index: Optional[int] = None) -> None:
        self._get_instance(instance_id).set_model_patch_replace(patch, name, block_name, number, transformer_index)
        
    async def set_model_input_block_patch(self, instance_id: str, patch: Any) -> None:
        self._get_instance(instance_id).set_model_input_block_patch(patch)
        
    async def set_model_input_block_patch_after_skip(self, instance_id: str, patch: Any) -> None:
        self._get_instance(instance_id).set_model_input_block_patch_after_skip(patch)

    async def set_model_output_block_patch(self, instance_id: str, patch: Any) -> None:
        self._get_instance(instance_id).set_model_output_block_patch(patch)
        
    async def set_model_emb_patch(self, instance_id: str, patch: Any) -> None:
        self._get_instance(instance_id).set_model_emb_patch(patch)
        
    async def set_model_forward_timestep_embed_patch(self, instance_id: str, patch: Any) -> None:
        self._get_instance(instance_id).set_model_forward_timestep_embed_patch(patch)
        
    async def set_model_double_block_patch(self, instance_id: str, patch: Any) -> None:
        self._get_instance(instance_id).set_model_double_block_patch(patch)
        
    async def set_model_post_input_patch(self, instance_id: str, patch: Any) -> None:
        self._get_instance(instance_id).set_model_post_input_patch(patch)

    async def set_model_rope_options(self, instance_id: str, options: dict) -> None:
        self._get_instance(instance_id).set_model_rope_options(**options)

    async def set_model_compute_dtype(self, instance_id: str, dtype: Any) -> None:
        self._get_instance(instance_id).set_model_compute_dtype(dtype)
        
    async def add_object_patch(self, instance_id: str, name: str, obj: Any) -> None:
        self._get_instance(instance_id).add_object_patch(name, obj)
        
    # =========================================================================
    # Wrappers / Callbacks / Attachments
    # =========================================================================

    async def add_weight_wrapper(self, instance_id: str, name: str, function: Any) -> None:
        self._get_instance(instance_id).add_weight_wrapper(name, function)
        
    async def add_wrapper_with_key(self, instance_id: str, wrapper_type: Any, key: str, fn: Any) -> None:
        self._get_instance(instance_id).add_wrapper_with_key(wrapper_type, key, fn)

    async def remove_wrappers_with_key(self, instance_id: str, wrapper_type: str, key: str) -> None:
        self._get_instance(instance_id).remove_wrappers_with_key(wrapper_type, key)
        
    async def get_wrappers(self, instance_id: str) -> Any:
        return {} # Not fully serializable; keep minimal for now

    async def add_callback_with_key(self, instance_id: str, call_type: str, key: str, callback: Any) -> None:
        self._get_instance(instance_id).add_callback_with_key(call_type, key, callback)
    
    async def remove_callbacks_with_key(self, instance_id: str, call_type: str, key: str) -> None:
        self._get_instance(instance_id).remove_callbacks_with_key(call_type, key)
        
    async def get_callbacks(self, instance_id: str) -> Any:
        return {} # Not fully serializable

    async def set_attachments(self, instance_id: str, key: str, attachment: Any) -> None:
        self._get_instance(instance_id).set_attachments(key, attachment)

    async def get_attachment(self, instance_id: str, key: str) -> Any:
        return self._get_instance(instance_id).get_attachment(key)

    async def remove_attachments(self, instance_id: str, key: str) -> None:
        self._get_instance(instance_id).remove_attachments(key)

    async def set_injections(self, instance_id: str, key: str, injections: Any) -> None:
        self._get_instance(instance_id).set_injections(key, injections)
    
    async def get_injections(self, instance_id: str, key: str) -> Any:
        return self._get_instance(instance_id).get_injections(key)

    async def remove_injections(self, instance_id: str, key: str) -> None:
        self._get_instance(instance_id).remove_injections(key)
        
    async def set_additional_models(self, instance_id: str, key: str, models: Any) -> None:
        self._get_instance(instance_id).set_additional_models(key, models)
        
    async def remove_additional_models(self, instance_id: str, key: str) -> None:
        self._get_instance(instance_id).remove_additional_models(key)
        
    async def get_nested_additional_models(self, instance_id: str) -> Any:
        return self._get_instance(instance_id).get_nested_additional_models()
        
    async def get_additional_models(self, instance_id: str) -> List[str]:
        models = self._get_instance(instance_id).get_additional_models()
        return [self.register(m) for m in models]

    async def model_patches_models(self, instance_id: str) -> Any:
        return self._get_instance(instance_id).model_patches_models()

    async def get_parent(self, instance_id: str) -> Any:
        return getattr(self._get_instance(instance_id), "parent", None)
        
    # =========================================================================
    # Weight Operations
    # =========================================================================

    async def patch_weight_to_device(self, instance_id: str, key: str, device_to: Any = None, inplace_update: bool = False) -> None:
        self._get_instance(instance_id).patch_weight_to_device(key, device_to, inplace_update)

    async def pin_weight_to_device(self, instance_id: str, key: str) -> None:
        self._get_instance(instance_id).pin_weight_to_device(key)

    async def unpin_weight(self, instance_id: str, key: str) -> None:
        self._get_instance(instance_id).unpin_weight(key)

    async def unpin_all_weights(self, instance_id: str) -> None:
        self._get_instance(instance_id).unpin_all_weights()
    
    async def calculate_weight(self, instance_id: str, patches: Any, weight: Any, key: str, intermediate_dtype: Any = float) -> Any:
        return detach_if_grad(self._get_instance(instance_id).calculate_weight(patches, weight, key, intermediate_dtype))

    # =========================================================================
    # Inner Model Access / Latent Processing
    # =========================================================================
    
    async def get_inner_model_attr(self, instance_id: str, name: str) -> Any:
        try:
            return getattr(self._get_instance(instance_id).model, name)
        except AttributeError:
            return None

    async def inner_model_memory_required(self, instance_id: str, args: tuple, kwargs: dict) -> Any:
        return self._get_instance(instance_id).model.memory_required(*args, **kwargs)

    async def inner_model_extra_conds_shapes(self, instance_id: str, args: tuple, kwargs: dict) -> Any:
        return self._get_instance(instance_id).model.extra_conds_shapes(*args, **kwargs)

    async def inner_model_extra_conds(self, instance_id: str, args: tuple, kwargs: dict) -> Any:
        return self._get_instance(instance_id).model.extra_conds(*args, **kwargs)

    async def inner_model_apply_model(self, instance_id: str, args: tuple, kwargs: dict) -> Any:
        instance = self._get_instance(instance_id)
        target = getattr(instance, "load_device", None)
        if target is None and args and hasattr(args[0], "device"):
            target = args[0].device
        elif target is None:
            for v in kwargs.values():
                if hasattr(v, "device"):
                    target = v.device
                    break

        def _move(obj):
            if target is None: return obj
            if isinstance(obj, (tuple, list)):
                return type(obj)(_move(o) for o in obj)
            if hasattr(obj, "to"):
                return obj.to(target)
            return obj

        moved_args = tuple(_move(a) for a in args)
        moved_kwargs = {k: _move(v) for k, v in kwargs.items()}
        result = instance.model.apply_model(*moved_args, **moved_kwargs)
        return detach_if_grad(_move(result))
        
    async def process_latent_in(self, instance_id: str, args: tuple, kwargs: dict) -> Any:
        return detach_if_grad(self._get_instance(instance_id).model.process_latent_in(*args, **kwargs))

    async def process_latent_out(self, instance_id: str, args: tuple, kwargs: dict) -> Any:
        instance = self._get_instance(instance_id)
        result = instance.model.process_latent_out(*args, **kwargs)
        try:
            target = None
            if args and hasattr(args[0], "device"):
                target = args[0].device
            elif kwargs:
                for v in kwargs.values():
                    if hasattr(v, "device"):
                        target = v.device
                        break
            if target is not None and hasattr(result, "to"):
                return detach_if_grad(result.to(target))
        except Exception:
            pass
        return detach_if_grad(result)

    async def scale_latent_inpaint(self, instance_id: str, args: tuple, kwargs: dict) -> Any:
        instance = self._get_instance(instance_id)
        result = instance.model.scale_latent_inpaint(*args, **kwargs)
        try:
            target = None
            if args and hasattr(args[0], "device"):
                target = args[0].device
            elif kwargs:
                for v in kwargs.values():
                    if hasattr(v, "device"):
                        target = v.device
                        break
            if target is not None and hasattr(result, "to"):
                return detach_if_grad(result.to(target))
        except Exception:
            pass
        return detach_if_grad(result)

    # =========================================================================
    # LoRA / High Level
    # =========================================================================

    async def load_lora(self, instance_id: str, lora_path: str, strength_model: float, clip_id: Optional[str] = None, strength_clip: float = 1.0) -> dict:
        import comfy.utils
        import comfy.sd
        import folder_paths
        from comfy.isolation.clip_proxy import CLIPRegistry

        model = self._get_instance(instance_id)
        clip = None
        if clip_id:
            clip = CLIPRegistry()._get_instance(clip_id)

        lora_full_path = folder_paths.get_full_path("loras", lora_path)
        if lora_full_path is None:
            raise ValueError(f"LoRA file not found: {lora_path}")

        lora = comfy.utils.load_torch_file(lora_full_path)
        new_model, new_clip = comfy.sd.load_lora_for_models(model, clip, lora, strength_model, strength_clip)

        new_model_id = self.register(new_model) if new_model else None
        new_clip_id = CLIPRegistry().register(new_clip) if (new_clip and clip_id) else None

        return {"model_id": new_model_id, "clip_id": new_clip_id}


class ModelPatcherProxy(BaseProxy[ModelPatcherRegistry]):
    _registry_class = ModelPatcherRegistry
    __module__ = "comfy.model_patcher"

    def _get_rpc(self) -> Any:
        if self._rpc_caller is None:
            from pyisolate._internal.shared import get_child_rpc_instance
            rpc = get_child_rpc_instance()
            if rpc is not None:
                self._rpc_caller = rpc.create_caller(self._registry_class, self._registry_class.get_remote_id())
            else:
                self._rpc_caller = self._registry
        return self._rpc_caller

    # =========================================================================
    # Core Methods & Properties
    # =========================================================================

    def is_clone(self, other: Any) -> bool:
        if isinstance(other, ModelPatcherProxy):
            return self._instance_id == other._instance_id
        return self._call_rpc("is_clone", other)

    def clone(self) -> ModelPatcherProxy:
        new_id = self._call_rpc("clone")
        return ModelPatcherProxy(new_id, self._registry, manage_lifecycle=not IS_CHILD_PROCESS)
        
    def clone_has_same_weights(self, clone: Any) -> bool:
        """Compare weights with another instance."""
        if isinstance(clone, ModelPatcherProxy):
            return self._call_rpc("clone_has_same_weights_by_id", clone._instance_id)
        if not IS_CHILD_PROCESS:
            return self._call_rpc("is_clone", clone) # Best effort on host
        return False

    def get_model_object(self, name: str) -> Any:
        return self._call_rpc("get_model_object", name)

    @property
    def model_options(self) -> dict:
        return self._call_rpc("get_model_options")

    @model_options.setter
    def model_options(self, value: dict) -> None:
        self._call_rpc("set_model_options", value)

    @property
    def load_device(self) -> Any:
        return self._call_rpc("get_load_device")

    @property
    def offload_device(self) -> Any:
        return self._call_rpc("get_offload_device")

    def current_loaded_device(self) -> Any:
        return self._call_rpc("current_loaded_device")

    @property
    def size(self) -> int:
        return self._call_rpc("get_size")
        
    def model_size(self) -> Any:
        return self._call_rpc("model_size")

    def loaded_size(self) -> Any:
        return self._call_rpc("loaded_size")
        
    def get_ram_usage(self) -> int:
        return self._call_rpc("get_ram_usage")
        
    def lowvram_patch_counter(self) -> int:
        return self._call_rpc("lowvram_patch_counter")

    def memory_required(self, input_shape: Any) -> Any:
        return self._call_rpc("memory_required", input_shape)

    def model_dtype(self) -> Any:
        return self._call_rpc("model_dtype")

    @property
    def hook_mode(self) -> Any:
        return self._call_rpc("get_hook_mode")

    @hook_mode.setter
    def hook_mode(self, value: Any) -> None:
        self._call_rpc("set_hook_mode", value)

    # =========================================================================
    # Configuration / Patches
    # =========================================================================
    
    def set_model_sampler_cfg_function(self, sampler_cfg_function: Any, disable_cfg1_optimization: bool = False) -> None:
        self._call_rpc("set_model_sampler_cfg_function", sampler_cfg_function, disable_cfg1_optimization)

    def set_model_sampler_post_cfg_function(self, post_cfg_function: Any, disable_cfg1_optimization: bool = False) -> None:
        self._call_rpc("set_model_sampler_post_cfg_function", post_cfg_function, disable_cfg1_optimization)

    def set_model_sampler_pre_cfg_function(self, pre_cfg_function: Any, disable_cfg1_optimization: bool = False) -> None:
        self._call_rpc("set_model_sampler_pre_cfg_function", pre_cfg_function, disable_cfg1_optimization)
    
    def set_model_sampler_calc_cond_batch_function(self, fn: Any) -> None:
        self._call_rpc("set_model_sampler_calc_cond_batch_function", fn)

    def set_model_unet_function_wrapper(self, unet_wrapper_function: Any) -> None:
        self._call_rpc("set_model_unet_function_wrapper", unet_wrapper_function)

    def set_model_denoise_mask_function(self, denoise_mask_function: Any) -> None:
        self._call_rpc("set_model_denoise_mask_function", denoise_mask_function)

    def set_model_patch(self, patch: Any, name: str) -> None:
        self._call_rpc("set_model_patch", patch, name)

    def set_model_patch_replace(self, patch: Any, name: str, block_name: str, number: int, transformer_index: Optional[int] = None) -> None:
        self._call_rpc("set_model_patch_replace", patch, name, block_name, number, transformer_index)

    def set_model_attn1_patch(self, patch: Any) -> None:
        self.set_model_patch(patch, "attn1_patch")

    def set_model_attn2_patch(self, patch: Any) -> None:
        self.set_model_patch(patch, "attn2_patch")
        
    def set_model_attn1_replace(self, patch: Any, block_name: str, number: int, transformer_index: Optional[int] = None) -> None:
        self.set_model_patch_replace(patch, "attn1", block_name, number, transformer_index)

    def set_model_attn2_replace(self, patch: Any, block_name: str, number: int, transformer_index: Optional[int] = None) -> None:
        self.set_model_patch_replace(patch, "attn2", block_name, number, transformer_index)

    def set_model_attn1_output_patch(self, patch: Any) -> None:
        self.set_model_patch(patch, "attn1_output_patch")

    def set_model_attn2_output_patch(self, patch: Any) -> None:
        self.set_model_patch(patch, "attn2_output_patch")
        
    def set_model_input_block_patch(self, patch: Any) -> None:
        self.set_model_patch(patch, "input_block_patch")
        
    def set_model_input_block_patch_after_skip(self, patch: Any) -> None:
        self.set_model_patch(patch, "input_block_patch_after_skip")

    def set_model_output_block_patch(self, patch: Any) -> None:
        self.set_model_patch(patch, "output_block_patch")
        
    def set_model_emb_patch(self, patch: Any) -> None:
        self.set_model_patch(patch, "emb_patch")
        
    def set_model_forward_timestep_embed_patch(self, patch: Any) -> None:
        self.set_model_patch(patch, "forward_timestep_embed_patch")
        
    def set_model_double_block_patch(self, patch: Any) -> None:
        self.set_model_patch(patch, "double_block")
        
    def set_model_post_input_patch(self, patch: Any) -> None:
        self.set_model_patch(patch, "post_input")

    def set_model_rope_options(self, scale_x=1.0, shift_x=0.0, scale_y=1.0, shift_y=0.0, scale_t=1.0, shift_t=0.0, **kwargs: Any) -> None:
        options = {
            "scale_x": scale_x, "shift_x": shift_x,
            "scale_y": scale_y, "shift_y": shift_y,
            "scale_t": scale_t, "shift_t": shift_t
        }
        options.update(kwargs)
        self._call_rpc("set_model_rope_options", options)

    def set_model_compute_dtype(self, dtype: Any) -> None:
        self._call_rpc("set_model_compute_dtype", dtype)
        
    def add_object_patch(self, name: str, obj: Any) -> None:
        self._call_rpc("add_object_patch", name, obj)
        
    # =========================================================================
    # Wrappers / Callbacks / Attachments
    # =========================================================================

    def add_weight_wrapper(self, name: str, function: Any) -> None:
        self._call_rpc("add_weight_wrapper", name, function)
        
    def add_wrapper_with_key(self, wrapper_type: Any, key: str, fn: Any) -> None:
        self._call_rpc("add_wrapper_with_key", wrapper_type, key, fn)

    def add_wrapper(self, wrapper_type: str, wrapper: Callable) -> None:
        self.add_wrapper_with_key(wrapper_type, None, wrapper)

    def remove_wrappers_with_key(self, wrapper_type: str, key: str) -> None:
        self._call_rpc("remove_wrappers_with_key", wrapper_type, key)
        
    @property
    def wrappers(self) -> Any:
        return self._call_rpc("get_wrappers")

    def add_callback_with_key(self, call_type: str, key: str, callback: Any) -> None:
        self._call_rpc("add_callback_with_key", call_type, key, callback)
    
    def add_callback(self, call_type: str, callback: Any) -> None:
        self.add_callback_with_key(call_type, None, callback)
    
    def remove_callbacks_with_key(self, call_type: str, key: str) -> None:
        self._call_rpc("remove_callbacks_with_key", call_type, key)
        
    @property
    def callbacks(self) -> Any:
        return self._call_rpc("get_callbacks")

    def set_attachments(self, key: str, attachment: Any) -> None:
        self._call_rpc("set_attachments", key, attachment)

    def get_attachment(self, key: str) -> Any:
        return self._call_rpc("get_attachment", key)

    def remove_attachments(self, key: str) -> None:
        self._call_rpc("remove_attachments", key)

    def set_injections(self, key: str, injections: Any) -> None:
        self._call_rpc("set_injections", key, injections)
    
    def get_injections(self, key: str) -> Any:
        return self._call_rpc("get_injections", key)

    def remove_injections(self, key: str) -> None:
        self._call_rpc("remove_injections", key)
        
    def set_additional_models(self, key: str, models: Any) -> None:
        ids = [m._instance_id for m in models]
        self._call_rpc("set_additional_models", key, ids)
        
    def remove_additional_models(self, key: str) -> None:
        self._call_rpc("remove_additional_models", key)
        
    def get_nested_additional_models(self) -> Any:
        return self._call_rpc("get_nested_additional_models")
        
    def get_additional_models(self) -> List[ModelPatcherProxy]:
        ids = self._call_rpc("get_additional_models")
        return [ModelPatcherProxy(mid, self._registry, manage_lifecycle=not IS_CHILD_PROCESS) for mid in ids]

    def model_patches_models(self) -> Any:
        return self._call_rpc("model_patches_models")

    @property
    def parent(self) -> Any:
        return self._call_rpc("get_parent")

    # =========================================================================
    # Device / Memory / Loading
    # =========================================================================

    def model_patches_to(self, device: Any) -> Any:
        return self._call_rpc("model_patches_to", device)

    def partially_load(self, device: Any, extra_memory: Any, force_patch_weights: bool = False) -> Any:
        return self._call_rpc("partially_load", device, extra_memory, force_patch_weights)
        
    def partially_unload(self, device_to: Any, memory_to_free: int = 0, force_patch_weights: bool = False) -> int:
        return self._call_rpc("partially_unload", device_to, memory_to_free, force_patch_weights)

    def load(self, device_to: Any = None, lowvram_model_memory: int = 0, force_patch_weights: bool = False, full_load: bool = False) -> None:
        self._call_rpc("load", device_to, lowvram_model_memory, force_patch_weights, full_load)

    def patch_model(self, device_to: Any = None, lowvram_model_memory: int = 0, load_weights: bool = True, force_patch_weights: bool = False) -> Any:
        self._call_rpc("patch_model", device_to, lowvram_model_memory, load_weights, force_patch_weights)
        return self

    def unpatch_model(self, device_to: Any = None, unpatch_weights: bool = True) -> None:
        self._call_rpc("unpatch_model", device_to, unpatch_weights)

    def detach(self, unpatch_all: bool = True) -> Any:
        self._call_rpc("detach", unpatch_all)
        return self.model

    # =========================================================================
    # Weight Operations
    # =========================================================================

    def patch_weight_to_device(self, key, device_to=None, inplace_update=False):
        self._call_rpc("patch_weight_to_device", key, device_to, inplace_update)

    def pin_weight_to_device(self, key):
        self._call_rpc("pin_weight_to_device", key)

    def unpin_weight(self, key):
        self._call_rpc("unpin_weight", key)

    def unpin_all_weights(self):
        self._call_rpc("unpin_all_weights")
        
    def calculate_weight(self, patches, weight, key, intermediate_dtype=None):
        return self._call_rpc("calculate_weight", patches, weight, key, intermediate_dtype)

    # =========================================================================
    # Lifecycle / Hooks / Injection
    # =========================================================================

    def inject_model(self) -> None:
        self._call_rpc("inject_model")

    def eject_model(self) -> None:
        self._call_rpc("eject_model")
        
    def use_ejected(self, skip_and_inject_on_exit_only: bool = False) -> Any:
        return AutoPatcherEjector(self, skip_and_inject_on_exit_only=skip_and_inject_on_exit_only)
        
    @property
    def is_injected(self) -> bool:
        return self._call_rpc("get_is_injected")

    @property
    def skip_injection(self) -> bool:
        return self._call_rpc("get_skip_injection")
        
    @skip_injection.setter
    def skip_injection(self, value: bool) -> None:
        self._call_rpc("set_skip_injection", value)

    def clean_hooks(self) -> None:
        self._call_rpc("clean_hooks")

    def pre_run(self) -> None:
        self._call_rpc("pre_run")
        
    def cleanup(self) -> None:
        self._call_rpc("cleanup")

    def restore_hook_patches(self) -> None:
        self._call_rpc("restore_hook_patches")
        
    def unpatch_hooks(self, whitelist_keys_set: Optional[Set[str]] = None) -> None:
        self._call_rpc("unpatch_hooks", whitelist_keys_set)

    def register_all_hook_patches(self, hooks: Any, target_dict: Any, model_options: Any, registered: Any) -> None:
        self._call_rpc("register_all_hook_patches", hooks, target_dict, model_options, registered)

    def apply_hooks(self, hooks: Any) -> Any:
        return self._call_rpc("apply_hooks", hooks)

    def prepare_state(self, timestep: Any) -> Any:
        return self._call_rpc("prepare_state", timestep)

    @property
    def model(self) -> _InnerModelProxy:
        return _InnerModelProxy(self)

    # =========================================================================
    # Guards & Hard Parts (Not Implemented yet)
    # =========================================================================

    @property
    def patches(self) -> Any:
        raise AttributeError("Direct access to 'patches' is not supported in isolated mode. See PASSDOWN_MODEL_PATCHER_HARD_PARTS.md")

    @property
    def object_patches(self) -> Any:
        raise AttributeError("Direct access to 'object_patches' is not supported in isolated mode.")

    def add_patches(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("add_patches() is not supported in isolated mode. Use load_lora() instead.")
        
    def load_lora(self, lora_path: str, strength_model: float, clip: Optional[Any] = None, strength_clip: float = 1.0) -> tuple:
        clip_id = None
        if clip is not None:
            # Handle both proxy types
            clip_id = getattr(clip, '_instance_id', getattr(clip, '_clip_id', None))
            
        result = self._call_rpc("load_lora", lora_path, strength_model, clip_id, strength_clip)
        
        new_model = None
        if result.get("model_id"):
            new_model = ModelPatcherProxy(result["model_id"], self._registry, manage_lifecycle=not IS_CHILD_PROCESS)
            
        new_clip = None
        if result.get("clip_id"):
            from comfy.isolation.clip_proxy import CLIPProxy
            new_clip = CLIPProxy(result["clip_id"])
            
        return (new_model, new_clip)


class _InnerModelProxy:
    def __init__(self, parent: ModelPatcherProxy):
        self._parent = parent

    def __getattr__(self, name: str) -> Any:
        if name.startswith('_'):
            raise AttributeError(name)
            
        if name in ('model_config', 'latent_format', 'model_type'):
            return self._parent._call_rpc("get_inner_model_attr", name)
            
        if name == 'load_device':
            return self._parent._call_rpc("get_inner_model_attr", 'load_device')

        if name == 'device':
            return self._parent._call_rpc("get_inner_model_attr", 'device')

        if name == 'current_patcher':
             return ModelPatcherProxy(self._parent._instance_id, self._parent._registry, manage_lifecycle=False)

        if name == 'model_sampling':
            return self._parent._call_rpc("get_model_object", "model_sampling")

        if name == 'extra_conds_shapes':
            return lambda *a, **k: self._parent._call_rpc("inner_model_extra_conds_shapes", a, k)
        if name == 'extra_conds':
            return lambda *a, **k: self._parent._call_rpc("inner_model_extra_conds", a, k)
        if name == 'memory_required':
            return lambda *a, **k: self._parent._call_rpc("inner_model_memory_required", a, k)
        if name == 'apply_model':
            return lambda *a, **k: self._parent._call_rpc("inner_model_apply_model", a, k)
        if name == 'process_latent_in':
            return lambda *a, **k: self._parent._call_rpc("process_latent_in", a, k)
        if name == 'process_latent_out':
            return lambda *a, **k: self._parent._call_rpc("process_latent_out", a, k)
        if name == 'scale_latent_inpaint':
            return lambda *a, **k: self._parent._call_rpc("scale_latent_inpaint", a, k)
            
        raise AttributeError(f"'{name}' not supported on isolated InnerModel")


def maybe_wrap_model_for_isolation(model_patcher: Any) -> Any:
    isolation_active = os.environ.get("PYISOLATE_ISOLATION_ACTIVE") == "1"
    is_child = os.environ.get("PYISOLATE_CHILD") == "1"
    
    if not isolation_active:
        return model_patcher
    
    if is_child:
        return model_patcher
    
    if isinstance(model_patcher, ModelPatcherProxy):
        return model_patcher
    
    registry = ModelPatcherRegistry()
    model_id = registry.register(model_patcher)
    logger.debug(f"Isolated ModelPatcher: {model_id}")
    return ModelPatcherProxy(model_id, registry, manage_lifecycle=True)