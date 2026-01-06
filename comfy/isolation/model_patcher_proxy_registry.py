# RPC server for ModelPatcher isolation (child process)
from __future__ import annotations

import logging
from typing import Any, Optional, List

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
    BaseRegistry,
    detach_if_grad,
)

logger = logging.getLogger(__name__)


class ModelPatcherRegistry(BaseRegistry[Any]):
    _type_prefix = "model"

    async def clone(self, instance_id: str) -> str:
        instance = self._get_instance(instance_id)
        new_model = instance.clone()
        return self.register(new_model)

    async def is_clone(self, instance_id: str, other: Any) -> bool:
        instance = self._get_instance(instance_id)
        if hasattr(other, "model"):
            return instance.is_clone(other)
        return False

    async def get_model_object(self, instance_id: str, name: str) -> Any:
        instance = self._get_instance(instance_id)
        if name == "model":
             return f"<ModelObject: {type(instance.model).__name__}>"
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
        opts = copy.deepcopy(instance.model_options)
        return self._sanitize_rpc_result(opts)

    async def set_model_options(self, instance_id: str, options: dict) -> None:
        self._get_instance(instance_id).model_options = options

    async def get_patcher_attr(self, instance_id: str, name: str) -> Any:
        return self._sanitize_rpc_result(getattr(self._get_instance(instance_id), name, None))

    async def model_state_dict(self, instance_id: str, filter_prefix=None) -> Any:
        instance = self._get_instance(instance_id)
        sd_keys = instance.model.state_dict().keys()
        return dict.fromkeys(sd_keys, None)

    def _sanitize_rpc_result(self, obj, seen=None):
        if seen is None:
            seen = set()
        if obj is None:
            return None
        if isinstance(obj, (bool, int, float, str)):
            if isinstance(obj, str) and len(obj) > 500000:
                return f"<Truncated String len={len(obj)}>"
            return obj
        obj_id = id(obj)
        if obj_id in seen:
            return None
        seen.add(obj_id)
        if isinstance(obj, (list, tuple)):
             return [self._sanitize_rpc_result(x, seen) for x in obj]
        if isinstance(obj, set):
             return [self._sanitize_rpc_result(x, seen) for x in obj]
        if isinstance(obj, dict):
             new_dict = {}
             for k, v in obj.items():
                 if isinstance(k, tuple):
                     import json
                     try:
                         key_str = "__pyisolate_key__" + json.dumps(list(k))
                         new_dict[key_str] = self._sanitize_rpc_result(v, seen)
                     except Exception:
                         new_dict[str(k)] = self._sanitize_rpc_result(v, seen)
                 else:
                     new_dict[str(k)] = self._sanitize_rpc_result(v, seen)
             return new_dict
        if hasattr(obj, "__dict__") and not hasattr(obj, "__get__") and not hasattr(obj, "__call__"):
             return self._sanitize_rpc_result(obj.__dict__, seen)
        if hasattr(obj, "items") and hasattr(obj, "get"):
             return {str(k): self._sanitize_rpc_result(v, seen) for k, v in obj.items()}
        return None

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

    async def model_patches_to(self, instance_id: str, device: Any) -> Any:
        return self._get_instance(instance_id).model_patches_to(device)

    async def partially_load(self, instance_id: str, device: Any, extra_memory: Any, force_patch_weights: bool = False) -> Any:
        return self._get_instance(instance_id).partially_load(device, extra_memory, force_patch_weights=force_patch_weights)

    async def partially_unload(self, instance_id: str, device_to: Any, memory_to_free: int = 0, force_patch_weights: bool = False) -> int:
        return self._get_instance(instance_id).partially_unload(device_to, memory_to_free, force_patch_weights)

    async def load(self, instance_id: str, device_to: Any = None, lowvram_model_memory: int = 0, force_patch_weights: bool = False, full_load: bool = False) -> None:
        self._get_instance(instance_id).load(device_to, lowvram_model_memory, force_patch_weights, full_load)

    async def patch_model(self, instance_id: str, device_to: Any = None, lowvram_model_memory: int = 0, load_weights: bool = True, force_patch_weights: bool = False) -> None:
        try:
            self._get_instance(instance_id).patch_model(device_to, lowvram_model_memory, load_weights, force_patch_weights)
        except AttributeError as e:
            logger.error(f"Isolation Error: Failed to patch model attribute: {e}. Skipping.")
            return

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
        if isinstance(hooks, SimpleNamespace) or hasattr(hooks, '__dict__'):
             hook_data = hooks.__dict__ if hasattr(hooks, '__dict__') else hooks
             new_hooks = comfy.hooks.HookGroup()
             if hasattr(hook_data, 'hooks'):
                   new_hooks.hooks = hook_data['hooks'] if isinstance(hook_data, dict) else hook_data.hooks
             hooks = new_hooks
        instance.register_all_hook_patches(hooks, target_dict, model_options, registered)

    async def get_hook_mode(self, instance_id: str) -> Any:
        return getattr(self._get_instance(instance_id), "hook_mode", None)

    async def set_hook_mode(self, instance_id: str, value: Any) -> None:
        setattr(self._get_instance(instance_id), "hook_mode", value)

    async def inject_model(self, instance_id: str) -> None:
        instance = self._get_instance(instance_id)
        try:
             instance.inject_model()
        except AttributeError as e:
             if "inject" in str(e):
                  logger.error("Isolation Error: Injector object lost method code during serialization. Cannot inject. Skipping.")
                  return
             raise e

    async def eject_model(self, instance_id: str) -> None:
        self._get_instance(instance_id).eject_model()

    async def get_is_injected(self, instance_id: str) -> bool:
        return self._get_instance(instance_id).is_injected

    async def set_skip_injection(self, instance_id: str, value: bool) -> None:
        self._get_instance(instance_id).skip_injection = value

    async def get_skip_injection(self, instance_id: str) -> bool:
        return self._get_instance(instance_id).skip_injection

    async def set_model_sampler_cfg_function(self, instance_id: str, sampler_cfg_function: Any, disable_cfg1_optimization: bool = False) -> None:
        if not callable(sampler_cfg_function):
             logger.error(f"set_model_sampler_cfg_function: Expected callable, got {type(sampler_cfg_function)}. Skipping.")
             return
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

    async def clone_has_same_weights_by_id(self, instance_id: str, other_id: str) -> bool:
        instance = self._get_instance(instance_id)
        other = self._get_instance(other_id)
        if not other:
            return False
        return instance.clone_has_same_weights(other)

    async def load_list_internal(self, instance_id: str, *args, **kwargs) -> Any:
        return self._get_instance(instance_id)._load_list(*args, **kwargs)

    async def is_clone_by_id(self, instance_id: str, other_id: str) -> bool:
         instance = self._get_instance(instance_id)
         other = self._get_instance(other_id)
         if hasattr(instance, "is_clone"):
              return instance.is_clone(other)
         return False

    async def add_object_patch(self, instance_id: str, name: str, obj: Any) -> None:
        self._get_instance(instance_id).add_object_patch(name, obj)

    async def add_weight_wrapper(self, instance_id: str, name: str, function: Any) -> None:
        self._get_instance(instance_id).add_weight_wrapper(name, function)

    async def add_wrapper_with_key(self, instance_id: str, wrapper_type: Any, key: str, fn: Any) -> None:
        self._get_instance(instance_id).add_wrapper_with_key(wrapper_type, key, fn)

    async def remove_wrappers_with_key(self, instance_id: str, wrapper_type: str, key: str) -> None:
        self._get_instance(instance_id).remove_wrappers_with_key(wrapper_type, key)

    async def get_wrappers(self, instance_id: str, wrapper_type: str = None, key: str = None) -> Any:
        if wrapper_type is None and key is None:
             return self._sanitize_rpc_result(getattr(self._get_instance(instance_id), "wrappers", {}))
        return self._sanitize_rpc_result(self._get_instance(instance_id).get_wrappers(wrapper_type, key))

    async def get_all_wrappers(self, instance_id: str, wrapper_type: str = None) -> Any:
        return self._sanitize_rpc_result(getattr(self._get_instance(instance_id), "get_all_wrappers", lambda x: [])(wrapper_type))

    async def add_callback_with_key(self, instance_id: str, call_type: str, key: str, callback: Any) -> None:
        self._get_instance(instance_id).add_callback_with_key(call_type, key, callback)

    async def remove_callbacks_with_key(self, instance_id: str, call_type: str, key: str) -> None:
        self._get_instance(instance_id).remove_callbacks_with_key(call_type, key)

    async def get_callbacks(self, instance_id: str, call_type: str = None, key: str = None) -> Any:
        if call_type is None and key is None:
             return self._sanitize_rpc_result(getattr(self._get_instance(instance_id), "callbacks", {}))
        return self._sanitize_rpc_result(self._get_instance(instance_id).get_callbacks(call_type, key))

    async def get_all_callbacks(self, instance_id: str, call_type: str = None) -> Any:
        return self._sanitize_rpc_result(getattr(self._get_instance(instance_id), "get_all_callbacks", lambda x: [])(call_type))

    async def set_attachments(self, instance_id: str, key: str, attachment: Any) -> None:
        self._get_instance(instance_id).set_attachments(key, attachment)

    async def get_attachment(self, instance_id: str, key: str) -> Any:
        return self._sanitize_rpc_result(self._get_instance(instance_id).get_attachment(key))

    async def remove_attachments(self, instance_id: str, key: str) -> None:
        self._get_instance(instance_id).remove_attachments(key)

    async def set_injections(self, instance_id: str, key: str, injections: Any) -> None:
        self._get_instance(instance_id).set_injections(key, injections)

    async def get_injections(self, instance_id: str, key: str) -> Any:
        return self._sanitize_rpc_result(self._get_instance(instance_id).get_injections(key))

    async def remove_injections(self, instance_id: str, key: str) -> None:
        self._get_instance(instance_id).remove_injections(key)

    async def set_additional_models(self, instance_id: str, key: str, models: Any) -> None:
        self._get_instance(instance_id).set_additional_models(key, models)

    async def remove_additional_models(self, instance_id: str, key: str) -> None:
        self._get_instance(instance_id).remove_additional_models(key)

    async def get_nested_additional_models(self, instance_id: str) -> Any:
        return self._sanitize_rpc_result(self._get_instance(instance_id).get_nested_additional_models())

    async def get_additional_models(self, instance_id: str) -> List[str]:
        models = self._get_instance(instance_id).get_additional_models()
        return [self.register(m) for m in models]

    async def get_additional_models_with_key(self, instance_id: str, key: str) -> Any:
         return self._sanitize_rpc_result(self._get_instance(instance_id).get_additional_models_with_key(key))

    async def model_patches_models(self, instance_id: str) -> Any:
        return self._sanitize_rpc_result(self._get_instance(instance_id).model_patches_models())

    async def get_patches(self, instance_id: str) -> Any:
        return self._sanitize_rpc_result(self._get_instance(instance_id).patches.copy())

    async def get_object_patches(self, instance_id: str) -> Any:
        return self._sanitize_rpc_result(self._get_instance(instance_id).object_patches.copy())

    async def add_patches(self, instance_id: str, patches: Any, strength_patch: float = 1.0, strength_model: float = 1.0) -> Any:
        return self._get_instance(instance_id).add_patches(patches, strength_patch, strength_model)

    async def get_key_patches(self, instance_id: str, filter_prefix: Optional[str] = None) -> Any:
        res = self._get_instance(instance_id).get_key_patches()
        if filter_prefix:
            res = {k: v for k, v in res.items() if k.startswith(filter_prefix)}
        safe_res = {}
        for k, v in res.items():
            safe_res[k] = [f"<Tensor shape={t.shape} dtype={t.dtype}>" if hasattr(t, "shape") else str(t) for t in v]
        return safe_res

    async def add_hook_patches(self, instance_id: str, hook: Any, patches: Any, strength_patch: float = 1.0, strength_model: float = 1.0) -> None:
        if hasattr(hook, 'hook_ref') and isinstance(hook.hook_ref, (dict, list, tuple)) and hasattr(hook.hook_ref, 'items'):
             try:
                 items = sorted(hook.hook_ref.items())
                 hook.hook_ref = tuple(items)
             except Exception:
                 hook.hook_ref = None
        self._get_instance(instance_id).add_hook_patches(hook, patches, strength_patch, strength_model)

    async def get_combined_hook_patches(self, instance_id: str, hooks: Any) -> Any:
        res = self._get_instance(instance_id).get_combined_hook_patches(hooks)
        return self._sanitize_rpc_result(res)

    async def clear_cached_hook_weights(self, instance_id: str) -> None:
        self._get_instance(instance_id).clear_cached_hook_weights()

    async def prepare_hook_patches_current_keyframe(self, instance_id: str, t: Any, hook_group: Any, model_options: Any) -> None:
        self._get_instance(instance_id).prepare_hook_patches_current_keyframe(t, hook_group, model_options)

    async def get_parent(self, instance_id: str) -> Any:
        return getattr(self._get_instance(instance_id), "parent", None)

    async def patch_weight_to_device(self, instance_id: str, key: str, device_to: Any = None, inplace_update: bool = False) -> None:
        self._get_instance(instance_id).patch_weight_to_device(key, device_to, inplace_update)

    async def pin_weight_to_device(self, instance_id: str, key: str) -> None:
        instance = self._get_instance(instance_id)
        if hasattr(instance, "pinned") and isinstance(instance.pinned, list):
            instance.pinned = set(instance.pinned)
        instance.pin_weight_to_device(key)

    async def unpin_weight(self, instance_id: str, key: str) -> None:
        instance = self._get_instance(instance_id)
        if hasattr(instance, "pinned") and isinstance(instance.pinned, list):
            instance.pinned = set(instance.pinned)
        instance.unpin_weight(key)

    async def unpin_all_weights(self, instance_id: str) -> None:
        instance = self._get_instance(instance_id)
        if hasattr(instance, "pinned") and isinstance(instance.pinned, list):
            instance.pinned = set(instance.pinned)
        instance.unpin_all_weights()

    async def calculate_weight(self, instance_id: str, patches: Any, weight: Any, key: str, intermediate_dtype: Any = float) -> Any:
        return detach_if_grad(self._get_instance(instance_id).calculate_weight(patches, weight, key, intermediate_dtype))

    async def get_inner_model_attr(self, instance_id: str, name: str) -> Any:
        try:
            return self._sanitize_rpc_result(getattr(self._get_instance(instance_id).model, name))
        except AttributeError:
            return None

    async def inner_model_memory_required(self, instance_id: str, args: tuple, kwargs: dict) -> Any:
        return self._get_instance(instance_id).model.memory_required(*args, **kwargs)

    async def inner_model_extra_conds_shapes(self, instance_id: str, args: tuple, kwargs: dict) -> Any:
        return self._get_instance(instance_id).model.extra_conds_shapes(*args, **kwargs)

    async def inner_model_extra_conds(self, instance_id: str, args: tuple, kwargs: dict) -> Any:
        return self._get_instance(instance_id).model.extra_conds(*args, **kwargs)

    async def inner_model_state_dict(self, instance_id: str, args: tuple, kwargs: dict) -> Any:
        sd = self._get_instance(instance_id).model.state_dict(*args, **kwargs)
        return {k: {'numel': v.numel(), 'element_size': v.element_size()} for k, v in sd.items()}

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
            if target is None:
                return obj
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
