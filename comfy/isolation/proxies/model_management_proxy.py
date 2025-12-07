import comfy.model_management as mm


class ModelManagementProxy:
    def get_torch_device(self):
        return mm.get_torch_device()

    def get_torch_device_name(self, device) -> str:
        return mm.get_torch_device_name(device)

    def soft_empty_cache(self, force=False):
        mm.soft_empty_cache(force=force)

    def unet_dtype(self, device=None, model_params=0, supported_dtypes=None, weight_dtype=None):
        if supported_dtypes is None:
            supported_dtypes = [mm.torch.float16, mm.torch.bfloat16, mm.torch.float32]
        return mm.unet_dtype(device, model_params, supported_dtypes, weight_dtype)

    def unet_offload_device(self):
        return mm.unet_offload_device()

    def free_memory(self, memory_required, device, keep_loaded=None):
        if keep_loaded is None:
            keep_loaded = []
        return mm.free_memory(memory_required, device, keep_loaded)

    def load_models_gpu(self, models, memory_required=0, force_patch_weights=False, minimum_memory_required=None, force_full_load=False):
        return mm.load_models_gpu(models, memory_required, force_patch_weights, minimum_memory_required, force_full_load)

    def load_model_gpu(self, model):
        return mm.load_model_gpu(model)

    def should_use_fp16(self, device=None, model_params=0, prioritize_performance=True, manual_cast=False):
        return mm.should_use_fp16(device, model_params, prioritize_performance, manual_cast)

    def should_use_bf16(self, device=None, model_params=0, prioritize_performance=True, manual_cast=False):
        return mm.should_use_bf16(device, model_params, prioritize_performance, manual_cast)

    def text_encoder_offload_device(self):
        return mm.text_encoder_offload_device()

    def intermediate_device(self):
        return mm.intermediate_device()

    def module_size(self, module):
        return mm.module_size(module)

    @property
    def OOM_EXCEPTION(self):
        return mm.OOM_EXCEPTION

    @property
    def XFORMERS_IS_AVAILABLE(self):
        return mm.XFORMERS_IS_AVAILABLE

