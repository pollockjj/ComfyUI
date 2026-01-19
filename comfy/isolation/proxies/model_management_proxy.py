import comfy.model_management as mm
from pyisolate import ProxiedSingleton

class ModelManagementProxy(ProxiedSingleton):
    """
    Proxy for comfy.model_management.
    Explicitly implements methods to ensure correct RPC delegation 
    and prevent silent fallback to local execution.
    """

    # Explicitly expose Enums/Classes as properties
    @property
    def VRAMState(self):
        return mm.VRAMState

    @property
    def CPUState(self):
        return mm.CPUState

    @property
    def OOM_EXCEPTION(self):
        return mm.OOM_EXCEPTION

    # -------------------------------------------------------------------------
    # Memory Management Core (The "Leak Fixers")
    # -------------------------------------------------------------------------

    def load_models_gpu(self, models, memory_required=0, force_patch_weights=False, minimum_memory_required=None, force_full_load=False):
        if IS_CHILD_PROCESS:
            return self._call_rpc("load_models_gpu", models, memory_required, force_patch_weights, minimum_memory_required, force_full_load)
        return mm.load_models_gpu(models, memory_required, force_patch_weights, minimum_memory_required, force_full_load)

    def cleanup_models_gc(self):
        if IS_CHILD_PROCESS:
            return self._call_rpc("cleanup_models_gc")
        return mm.cleanup_models_gc()

    def unload_all_models(self):
        if IS_CHILD_PROCESS:
            return self._call_rpc("unload_all_models")
        return mm.unload_all_models()

    def free_memory(self, memory_required, device, keep_loaded=[]):
        if IS_CHILD_PROCESS:
            return self._call_rpc("free_memory", memory_required, device, keep_loaded)
        return mm.free_memory(memory_required, device, keep_loaded)

    def soft_empty_cache(self, force=False):
        if IS_CHILD_PROCESS:
            return self._call_rpc("soft_empty_cache", force)
        return mm.soft_empty_cache(force)

    def load_model_gpu(self, model):
        if IS_CHILD_PROCESS:
            return self._call_rpc("load_model_gpu", model)
        return mm.load_model_gpu(model)

    def unload_model(self, model):
        if IS_CHILD_PROCESS:
            return self._call_rpc("unload_model", model)
        return mm.unload_model(model)

    # -------------------------------------------------------------------------
    # Device & Stats Queries
    # -------------------------------------------------------------------------

    def get_torch_device(self):
        if IS_CHILD_PROCESS:
            return self._call_rpc("get_torch_device")
        return mm.get_torch_device()

    def get_torch_device_name(self, device):
        if IS_CHILD_PROCESS:
            # Device objects often can't be serialized simply if they are torch.device
            # But they should picklable.
            return self._call_rpc("get_torch_device_name", device)
        return mm.get_torch_device_name(device)

    def get_total_memory(self, device=None, torch_total_too=False):
        if IS_CHILD_PROCESS:
            return self._call_rpc("get_total_memory", device, torch_total_too)
        return mm.get_total_memory(device, torch_total_too)

    def get_free_memory(self, device=None, torch_free_too=False):
        if IS_CHILD_PROCESS:
            return self._call_rpc("get_free_memory", device, torch_free_too)
        return mm.get_free_memory(device, torch_free_too)

    def vram_usage(self):
        if IS_CHILD_PROCESS:
            return self._call_rpc("vram_usage")
        return mm.vram_usage()

    # -------------------------------------------------------------------------
    # Feature Flags & Capabilities
    # -------------------------------------------------------------------------

    def should_use_fp16(self, device=None, model_params=0, prioritize_performance=True, manual_cast=False):
        if IS_CHILD_PROCESS:
            return self._call_rpc("should_use_fp16", device, model_params, prioritize_performance, manual_cast)
        return mm.should_use_fp16(device, model_params, prioritize_performance, manual_cast)

    def should_use_bf16(self, device=None, model_params=0, prioritize_performance=True, manual_cast=False):
        if IS_CHILD_PROCESS:
            return self._call_rpc("should_use_bf16", device, model_params, prioritize_performance, manual_cast)
        return mm.should_use_bf16(device, model_params, prioritize_performance, manual_cast)

    def supports_fp8_compute(self, device=None):
        if IS_CHILD_PROCESS:
            return self._call_rpc("supports_fp8_compute", device)
        return mm.supports_fp8_compute(device)

    def supports_nvfp4_compute(self, device=None):
        if IS_CHILD_PROCESS:
            return self._call_rpc("supports_nvfp4_compute", device)
        return mm.supports_nvfp4_compute(device)

    def xformers_enabled(self):
        if IS_CHILD_PROCESS:
            return self._call_rpc("xformers_enabled")
        return mm.xformers_enabled()

    def pytorch_attention_enabled(self):
        if IS_CHILD_PROCESS:
            return self._call_rpc("pytorch_attention_enabled")
        return mm.pytorch_attention_enabled()

    def is_device_mps(self, device):
        if IS_CHILD_PROCESS:
            return self._call_rpc("is_device_mps", device)
        return mm.is_device_mps(device)

    # -------------------------------------------------------------------------
    # Interrupts
    # -------------------------------------------------------------------------

    def interrupt_current_processing(self, value=True):
        if IS_CHILD_PROCESS:
            return self._call_rpc("interrupt_current_processing", value)
        return mm.interrupt_current_processing(value)

    def processing_interrupted(self):
        if IS_CHILD_PROCESS:
            return self._call_rpc("processing_interrupted")
        return mm.processing_interrupted()

    def throw_exception_if_processing_interrupted(self):
        if IS_CHILD_PROCESS:
            return self._call_rpc("throw_exception_if_processing_interrupted")
        return mm.throw_exception_if_processing_interrupted()
