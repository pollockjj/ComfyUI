from __future__ import annotations

import os
from typing import Any, Optional

from pyisolate import ProxiedSingleton

from .base import call_singleton_rpc
from .singleton_contract import SingletonProxyContract, install_singleton_module_proxy


MODEL_MANAGEMENT_PUBLIC_CALLABLES = (
    "get_supported_float8_types",
    "is_intel_xpu",
    "is_ascend_npu",
    "is_mlu",
    "is_ixuca",
    "is_wsl",
    "get_torch_device",
    "get_all_torch_devices",
    "get_gpu_device_options",
    "get_gpu_device_options_no_cpu",
    "resolve_gpu_device_option",
    "cuda_device_context",
    "get_total_memory",
    "mac_version",
    "is_oom",
    "raise_non_oom",
    "is_nvidia",
    "is_amd",
    "amd_min_version",
    "get_torch_device_name",
    "module_size",
    "mark_mmap_dirty",
    "free_pins",
    "ensure_pin_budget",
    "ensure_pin_registerable",
    "use_more_memory",
    "offloaded_memory",
    "extra_reserved_memory",
    "minimum_inference_memory",
    "free_memory",
    "load_models_gpu",
    "load_model_gpu",
    "loaded_models",
    "cleanup_models_gc",
    "archive_model_dtypes",
    "cleanup_models",
    "dtype_size",
    "unet_offload_device",
    "unet_inital_load_device",
    "maximum_vram_for_weights",
    "unet_dtype",
    "unet_manual_cast",
    "text_encoder_offload_device",
    "text_encoder_device",
    "text_encoder_initial_device",
    "text_encoder_dtype",
    "intermediate_device",
    "intermediate_dtype",
    "vae_device",
    "vae_offload_device",
    "vae_dtype",
    "get_autocast_device",
    "supports_dtype",
    "supports_cast",
    "pick_weight_dtype",
    "device_supports_non_blocking",
    "force_channels_last",
    "current_stream",
    "get_cast_buffer",
    "get_aimdo_cast_buffer",
    "get_pin_buffer",
    "resize_pin_buffer",
    "reset_cast_buffers",
    "get_offload_stream",
    "sync_stream",
    "cast_to_gathered",
    "cast_to",
    "cast_to_device",
    "pinned_hostbuf_size",
    "discard_cuda_async_error",
    "pin_memory",
    "unpin_memory",
    "sage_attention_enabled",
    "flash_attention_enabled",
    "xformers_enabled",
    "xformers_enabled_vae",
    "pytorch_attention_enabled",
    "pytorch_attention_enabled_vae",
    "pytorch_attention_flash_attention",
    "force_upcast_attention_dtype",
    "get_free_memory",
    "cpu_mode",
    "mps_mode",
    "is_device_type",
    "is_device_cpu",
    "is_device_mps",
    "is_device_xpu",
    "is_device_cuda",
    "is_directml_enabled",
    "should_use_fp16",
    "should_use_bf16",
    "supports_fp8_compute",
    "supports_nvfp4_compute",
    "supports_mxfp8_compute",
    "supports_fp64",
    "extended_fp16_support",
    "lora_compute_dtype",
    "synchronize",
    "soft_empty_cache",
    "unload_all_models",
    "unload_model_and_clones",
    "debug_memory_summary",
    "interrupt_current_processing",
    "processing_interrupted",
    "throw_exception_if_processing_interrupted",
)

MODEL_MANAGEMENT_CUSTOM_SYMBOLS = (
    "module_size",
    "archive_model_dtypes",
)

MODEL_MANAGEMENT_RELAY_SYMBOLS = tuple(
    name for name in MODEL_MANAGEMENT_PUBLIC_CALLABLES
    if name not in MODEL_MANAGEMENT_CUSTOM_SYMBOLS
)

MODEL_MANAGEMENT_SINGLETON_CONTRACT = SingletonProxyContract(
    proxy_name="ModelManagementProxy",
    target_name="comfy.model_management",
    target_public_symbols=MODEL_MANAGEMENT_PUBLIC_CALLABLES,
    relay_symbols=MODEL_MANAGEMENT_RELAY_SYMBOLS,
    custom_symbols=MODEL_MANAGEMENT_CUSTOM_SYMBOLS,
)


def _mm():
    import comfy.model_management

    return comfy.model_management


def _is_child_process() -> bool:
    return os.environ.get("PYISOLATE_CHILD") == "1"


class TorchDeviceProxy:
    def __init__(self, device_str: str):
        self._device_str = device_str
        if ":" in device_str:
            device_type, index = device_str.split(":", 1)
            self.type = device_type
            self.index = int(index)
        else:
            self.type = device_str
            self.index = None

    def __str__(self) -> str:
        return self._device_str

    def __repr__(self) -> str:
        return f"TorchDeviceProxy({self._device_str!r})"


def _serialize_value(value: Any) -> Any:
    value_type = type(value)
    if value_type.__module__ == "torch" and value_type.__name__ == "device":
        return {"__pyisolate_torch_device__": str(value)}
    if isinstance(value, TorchDeviceProxy):
        return {"__pyisolate_torch_device__": str(value)}
    if isinstance(value, tuple):
        return {"__pyisolate_tuple__": [_serialize_value(item) for item in value]}
    if isinstance(value, list):
        return [_serialize_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _serialize_value(inner) for key, inner in value.items()}
    return value


def _deserialize_value(value: Any) -> Any:
    if isinstance(value, dict):
        if "__pyisolate_torch_device__" in value:
            try:
                import torch
            except ImportError:
                return TorchDeviceProxy(value["__pyisolate_torch_device__"])
            return torch.device(value["__pyisolate_torch_device__"])
        if "__pyisolate_tuple__" in value:
            return tuple(_deserialize_value(item) for item in value["__pyisolate_tuple__"])
        return {key: _deserialize_value(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [_deserialize_value(item) for item in value]
    return value


def _normalize_argument(value: Any) -> Any:
    if isinstance(value, TorchDeviceProxy):
        import torch

        return torch.device(str(value))
    if isinstance(value, dict):
        if "__pyisolate_torch_device__" in value:
            import torch

            return torch.device(value["__pyisolate_torch_device__"])
        if "__pyisolate_tuple__" in value:
            return tuple(_normalize_argument(item) for item in value["__pyisolate_tuple__"])
        return {key: _normalize_argument(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [_normalize_argument(item) for item in value]
    return value


class ModelManagementProxy(ProxiedSingleton):
    """
    Exact-relay proxy for comfy.model_management.
    Child calls never import comfy.model_management directly; they serialize
    arguments, relay to host, and deserialize the host result back.
    """

    _rpc: Optional[Any] = None

    @classmethod
    def set_rpc(cls, rpc: Any) -> None:
        cls._rpc = rpc.create_caller(cls, cls.get_remote_id())

    @classmethod
    def clear_rpc(cls) -> None:
        cls._rpc = None

    @classmethod
    def _get_caller(cls) -> Any:
        if cls._rpc is None:
            raise RuntimeError("ModelManagementProxy RPC caller is not configured")
        return cls._rpc

    def _relay_call(self, method_name: str, *args: Any, **kwargs: Any) -> Any:
        payload = call_singleton_rpc(
            self._get_caller(),
            "rpc_call",
            method_name,
            _serialize_value(args),
            _serialize_value(kwargs),
        )
        return _deserialize_value(payload)

    def install_into(self, target_module: Any) -> dict[str, tuple[str, ...]]:
        return install_singleton_module_proxy(
            target_module,
            self,
            MODEL_MANAGEMENT_SINGLETON_CONTRACT,
        )

    def archive_model_dtypes(self, model: Any) -> None:
        for _name, module in model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                setattr(module, f"{param_name}_comfy_model_dtype", param.dtype)
            for buf_name, buf in module.named_buffers(recurse=False):
                setattr(module, f"{buf_name}_comfy_model_dtype", buf.dtype)

    def module_size(self, module: Any) -> int:
        module_mem = 0
        state_dict = module.state_dict()
        for key in state_dict:
            module_mem += state_dict[key].nbytes
        return module_mem

    @property
    def VRAMState(self):
        return _mm().VRAMState

    @property
    def CPUState(self):
        return _mm().CPUState

    @property
    def OOM_EXCEPTION(self):
        return _mm().OOM_EXCEPTION

    def __getattr__(self, name: str):
        if _is_child_process():
            def child_method(*args: Any, **kwargs: Any) -> Any:
                return self._relay_call(name, *args, **kwargs)

            return child_method
        return getattr(_mm(), name)

    async def rpc_call(self, method_name: str, args: Any, kwargs: Any) -> Any:
        normalized_args = _normalize_argument(_deserialize_value(args))
        normalized_kwargs = _normalize_argument(_deserialize_value(kwargs))
        method = getattr(_mm(), method_name)
        result = method(*normalized_args, **normalized_kwargs)
        return _serialize_value(result)
