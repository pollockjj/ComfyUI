import comfy.model_management as mm
from pyisolate import ProxiedSingleton


class ModelManagementProxy(ProxiedSingleton):
    def get_supported_float8_types(self):
        return mm.get_supported_float8_types()

    @property
    def FLOAT8_TYPES(self):
        return mm.FLOAT8_TYPES

    def is_intel_xpu(self):
        return mm.is_intel_xpu()

    def is_ascend_npu(self):
        return mm.is_ascend_npu()

    def is_mlu(self):
        return mm.is_mlu()

    def is_ixuca(self):
        return mm.is_ixuca()

    def get_torch_device(self):
        return mm.get_torch_device()

    def get_total_memory(self, dev=None, torch_total_too=False):
        return mm.get_total_memory(dev, torch_total_too)

    def mac_version(self):
        return mm.mac_version()

    @property
    def OOM_EXCEPTION(self):
        return mm.OOM_EXCEPTION

    @property
    def XFORMERS_IS_AVAILABLE(self):
        return mm.XFORMERS_IS_AVAILABLE

    @property
    def XFORMERS_VERSION(self):
        return mm.XFORMERS_VERSION

    @property
    def XFORMERS_ENABLED_VAE(self):
        return mm.XFORMERS_ENABLED_VAE

    def is_nvidia(self):
        return mm.is_nvidia()

    def is_amd(self):
        return mm.is_amd()

    def amd_min_version(self, device=None, min_rdna_version=0):
        return mm.amd_min_version(device, min_rdna_version)

    @property
    def MIN_WEIGHT_MEMORY_RATIO(self):
        return mm.MIN_WEIGHT_MEMORY_RATIO

    @property
    def ENABLE_PYTORCH_ATTENTION(self):
        return mm.ENABLE_PYTORCH_ATTENTION

    @property
    def SUPPORT_FP8_OPS(self):
        return mm.SUPPORT_FP8_OPS

    @property
    def AMD_RDNA2_AND_OLDER_ARCH(self):
        return mm.AMD_RDNA2_AND_OLDER_ARCH

    @property
    def PRIORITIZE_FP16(self):
        return mm.PRIORITIZE_FP16

    @property
    def FORCE_FP32(self):
        return mm.FORCE_FP32

    @property
    def DISABLE_SMART_MEMORY(self):
        return mm.DISABLE_SMART_MEMORY

    def get_torch_device_name(self, device):
        return mm.get_torch_device_name(device)

    def module_size(self, module):
        return mm.module_size(module)

    def use_more_memory(self, extra_memory, loaded_models, device):
        return mm.use_more_memory(extra_memory, loaded_models, device)

    def offloaded_memory(self, loaded_models, device):
        return mm.offloaded_memory(loaded_models, device)

    @property
    def WINDOWS(self):
        return mm.WINDOWS

    @property
    def EXTRA_RESERVED_VRAM(self):
        return mm.EXTRA_RESERVED_VRAM

    def extra_reserved_memory(self):
        return mm.extra_reserved_memory()

    def minimum_inference_memory(self):
        return mm.minimum_inference_memory()

    def free_memory(self, memory_required, device, keep_loaded=None):
        if keep_loaded is None:
            keep_loaded = []
        return mm.free_memory(memory_required, device, keep_loaded)

    def load_models_gpu(self, models, memory_required=0, force_patch_weights=False, minimum_memory_required=None, force_full_load=False):
        return mm.load_models_gpu(models, memory_required, force_patch_weights, minimum_memory_required, force_full_load)

    def load_model_gpu(self, model):
        return mm.load_model_gpu(model)

    def loaded_models(self, only_currently_used=False):
        return mm.loaded_models(only_currently_used)

    def cleanup_models_gc(self):
        return mm.cleanup_models_gc()

    def cleanup_models(self):
        return mm.cleanup_models()

    def dtype_size(self, dtype):
        return mm.dtype_size(dtype)

    def unet_offload_device(self):
        return mm.unet_offload_device()

    def unet_inital_load_device(self, parameters, dtype):
        return mm.unet_inital_load_device(parameters, dtype)

    def maximum_vram_for_weights(self, device=None):
        return mm.maximum_vram_for_weights(device)

    def unet_dtype(self, device=None, model_params=0, supported_dtypes=None, weight_dtype=None):
        if supported_dtypes is None:
            supported_dtypes = [mm.torch.float16, mm.torch.bfloat16, mm.torch.float32]
        return mm.unet_dtype(device, model_params, supported_dtypes, weight_dtype)

    def unet_manual_cast(self, weight_dtype, inference_device, supported_dtypes=None):
        if supported_dtypes is None:
            supported_dtypes = [mm.torch.float16, mm.torch.bfloat16, mm.torch.float32]
        return mm.unet_manual_cast(weight_dtype, inference_device, supported_dtypes)

    def text_encoder_offload_device(self):
        return mm.text_encoder_offload_device()

    def text_encoder_device(self):
        return mm.text_encoder_device()

    def text_encoder_initial_device(self, load_device, offload_device, model_size=0):
        return mm.text_encoder_initial_device(load_device, offload_device, model_size)

    def text_encoder_dtype(self, device=None):
        return mm.text_encoder_dtype(device)

    def intermediate_device(self):
        return mm.intermediate_device()

    def vae_device(self):
        return mm.vae_device()

    def vae_offload_device(self):
        return mm.vae_offload_device()

    def vae_dtype(self, device=None, allowed_dtypes=None):
        if allowed_dtypes is None:
            allowed_dtypes = []
        return mm.vae_dtype(device, allowed_dtypes)

    def get_autocast_device(self, dev):
        return mm.get_autocast_device(dev)

    def supports_dtype(self, device, dtype):
        return mm.supports_dtype(device, dtype)

    def supports_cast(self, device, dtype):
        return mm.supports_cast(device, dtype)

    def pick_weight_dtype(self, dtype, fallback_dtype, device=None):
        return mm.pick_weight_dtype(dtype, fallback_dtype, device)

    def device_supports_non_blocking(self, device):
        return mm.device_supports_non_blocking(device)

    def force_channels_last(self):
        return mm.force_channels_last()

    @property
    def STREAMS(self):
        return mm.STREAMS

    @property
    def NUM_STREAMS(self):
        return mm.NUM_STREAMS

    def current_stream(self, device):
        return mm.current_stream(device)

    def get_offload_stream(self, device):
        return mm.get_offload_stream(device)

    def sync_stream(self, device, stream):
        return mm.sync_stream(device, stream)

    def cast_to(self, weight, dtype=None, device=None, non_blocking=False, copy=False, stream=None):
        return mm.cast_to(weight, dtype, device, non_blocking, copy, stream)

    def cast_to_device(self, tensor, device, dtype, copy=False):
        return mm.cast_to_device(tensor, device, dtype, copy)

    @property
    def PINNED_MEMORY(self):
        return mm.PINNED_MEMORY

    @property
    def TOTAL_PINNED_MEMORY(self):
        return mm.TOTAL_PINNED_MEMORY

    @property
    def MAX_PINNED_MEMORY(self):
        return mm.MAX_PINNED_MEMORY

    @property
    def PINNING_ALLOWED_TYPES(self):
        return mm.PINNING_ALLOWED_TYPES

    def pin_memory(self, tensor):
        return mm.pin_memory(tensor)

    def unpin_memory(self, tensor):
        return mm.unpin_memory(tensor)

    def sage_attention_enabled(self):
        return mm.sage_attention_enabled()

    def flash_attention_enabled(self):
        return mm.flash_attention_enabled()

    def xformers_enabled(self):
        return mm.xformers_enabled()

    def xformers_enabled_vae(self):
        return mm.xformers_enabled_vae()

    def pytorch_attention_enabled(self):
        return mm.pytorch_attention_enabled()

    def pytorch_attention_enabled_vae(self):
        return mm.pytorch_attention_enabled_vae()

    def pytorch_attention_flash_attention(self):
        return mm.pytorch_attention_flash_attention()

    def force_upcast_attention_dtype(self):
        return mm.force_upcast_attention_dtype()

    def get_free_memory(self, dev=None, torch_free_too=False):
        return mm.get_free_memory(dev, torch_free_too)

    def cpu_mode(self):
        return mm.cpu_mode()

    def mps_mode(self):
        return mm.mps_mode()

    def is_device_type(self, device, type):
        return mm.is_device_type(device, type)

    def is_device_cpu(self, device):
        return mm.is_device_cpu(device)

    def is_device_mps(self, device):
        return mm.is_device_mps(device)

    def is_device_xpu(self, device):
        return mm.is_device_xpu(device)

    def is_device_cuda(self, device):
        return mm.is_device_cuda(device)

    def is_directml_enabled(self):
        return mm.is_directml_enabled()

    def should_use_fp16(self, device=None, model_params=0, prioritize_performance=True, manual_cast=False):
        return mm.should_use_fp16(device, model_params, prioritize_performance, manual_cast)

    def should_use_bf16(self, device=None, model_params=0, prioritize_performance=True, manual_cast=False):
        return mm.should_use_bf16(device, model_params, prioritize_performance, manual_cast)

    def supports_fp8_compute(self, device=None):
        return mm.supports_fp8_compute(device)

    def extended_fp16_support(self):
        return mm.extended_fp16_support()

    def soft_empty_cache(self, force=False):
        mm.soft_empty_cache(force=force)

    def unload_all_models(self):
        return mm.unload_all_models()

    def interrupt_current_processing(self, value=True):
        return mm.interrupt_current_processing(value)

    def processing_interrupted(self):
        return mm.processing_interrupted()

    def throw_exception_if_processing_interrupted(self):
        return mm.throw_exception_if_processing_interrupted()
