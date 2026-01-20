"""
ModelManagementProxy - RPC proxy for comfy.model_management.

This is a ProxiedSingleton that exposes model_management functions to isolated Child processes.

Architecture:
- On Host: This class is instantiated directly, methods execute locally
- On Child: pyisolate replaces the singleton with a CallWrapper that intercepts
  method calls and forwards them via RPC to Host

All methods MUST be async for pyisolate's CallWrapper to work.
"""
import os

import comfy.model_management as mm
from pyisolate import ProxiedSingleton


def _resolve_proxy_to_real_model(model):
    """On Host, convert a ModelPatcherProxy to its real ModelPatcher."""
    from comfy.isolation.model_patcher_proxy import ModelPatcherProxy
    from comfy.isolation.model_patcher_proxy_registry import get_real_model_patcher_registry

    if isinstance(model, ModelPatcherProxy):
        # Get the real model from registry
        registry = get_real_model_patcher_registry()
        if registry:
            try:
                return registry._get_instance(model._instance_id)
            except (ValueError, KeyError):
                pass
        # Fallback: try via proxy's registry reference
        try:
            return model._registry._get_instance(model._instance_id)
        except (ValueError, KeyError):
            return model  # Last resort: return proxy
    return model


class ModelManagementProxy(ProxiedSingleton):
    """
    Proxy for comfy.model_management module functions.

    Exposes critical VRAM management functions to isolated Child processes.
    All methods are async and execute on Host - pyisolate handles RPC from Child.
    """

    # -------------------------------------------------------------------------
    # Memory Management Core (The "Leak Fixers")
    # -------------------------------------------------------------------------

    async def load_models_gpu(self, models, memory_required=0, force_patch_weights=False, minimum_memory_required=None, force_full_load=False):
        """Load models to GPU. Resolves proxies to real models on Host."""
        import logging
        logging.info(f"[ModelManagementProxy.load_models_gpu] Called with {len(models)} models")
        resolved_models = [_resolve_proxy_to_real_model(m) for m in models]
        logging.info(f"[ModelManagementProxy.load_models_gpu] Resolved to: {[type(m).__name__ for m in resolved_models]}")
        return mm.load_models_gpu(resolved_models, memory_required, force_patch_weights, minimum_memory_required, force_full_load)

    async def cleanup_models_gc(self):
        """Cleanup unreferenced models."""
        return mm.cleanup_models_gc()

    async def unload_all_models(self):
        """Unload all models from GPU."""
        return mm.unload_all_models()

    async def free_memory(self, memory_required, device, keep_loaded=None):
        """Free GPU memory by unloading models."""
        if keep_loaded is None:
            keep_loaded = []
        return mm.free_memory(memory_required, device, keep_loaded)

    async def soft_empty_cache(self, force=False):
        """Soft empty CUDA cache."""
        return mm.soft_empty_cache(force)

    async def load_model_gpu(self, model):
        """Load a single model to GPU."""
        resolved_model = _resolve_proxy_to_real_model(model)
        return mm.load_model_gpu(resolved_model)

    async def unload_model(self, model):
        """Unload a single model from GPU."""
        return mm.unload_model(model)

    # -------------------------------------------------------------------------
    # Device & Stats Queries
    # -------------------------------------------------------------------------

    async def get_torch_device(self):
        """Get the torch device for inference."""
        return mm.get_torch_device()

    async def get_torch_device_name(self, device):
        """Get the name of a torch device."""
        return mm.get_torch_device_name(device)

    async def get_total_memory(self, device=None, torch_total_too=False):
        """Get total GPU memory."""
        # Device may not serialize correctly across RPC - resolve on Host
        # Also handle explicit cpu/string devices by using Host's torch device
        if device is None or (hasattr(device, 'type') and device.type == 'cpu') or device == 'cpu':
            device = mm.get_torch_device()
        return mm.get_total_memory(device, torch_total_too)

    async def get_free_memory(self, device=None, torch_free_too=False):
        """Get free GPU memory."""
        # Device may not serialize correctly across RPC - resolve on Host
        # Also handle explicit cpu/string devices by using Host's torch device
        if device is None or (hasattr(device, 'type') and device.type == 'cpu') or device == 'cpu':
            device = mm.get_torch_device()
        return mm.get_free_memory(device, torch_free_too)

    async def vram_usage(self):
        """Get current VRAM usage."""
        return mm.vram_usage()

    # -------------------------------------------------------------------------
    # Feature Flags & Capabilities
    # -------------------------------------------------------------------------

    async def should_use_fp16(self, device=None, model_params=0, prioritize_performance=True, manual_cast=False):
        """Check if fp16 should be used."""
        return mm.should_use_fp16(device, model_params, prioritize_performance, manual_cast)

    async def should_use_bf16(self, device=None, model_params=0, prioritize_performance=True, manual_cast=False):
        """Check if bf16 should be used."""
        return mm.should_use_bf16(device, model_params, prioritize_performance, manual_cast)

    async def supports_fp8_compute(self, device=None):
        """Check if fp8 compute is supported."""
        return mm.supports_fp8_compute(device)

    async def supports_nvfp4_compute(self, device=None):
        """Check if nvfp4 compute is supported."""
        return mm.supports_nvfp4_compute(device)

    async def xformers_enabled(self):
        """Check if xformers is enabled."""
        return mm.xformers_enabled()

    async def pytorch_attention_enabled(self):
        """Check if pytorch attention is enabled."""
        return mm.pytorch_attention_enabled()

    async def is_device_mps(self, device):
        """Check if device is MPS (Apple Silicon)."""
        return mm.is_device_mps(device)

    # -------------------------------------------------------------------------
    # Interrupts
    # -------------------------------------------------------------------------

    async def interrupt_current_processing(self, value=True):
        """Interrupt current processing."""
        return mm.interrupt_current_processing(value)

    async def processing_interrupted(self):
        """Check if processing was interrupted."""
        return mm.processing_interrupted()

    async def throw_exception_if_processing_interrupted(self):
        """Throw exception if processing was interrupted."""
        # Check flag first, only raise if actually interrupted
        # This avoids "RPC dispatch failed" traceback from pyisolate
        if mm.processing_interrupted():
            raise mm.InterruptProcessingException()

    # -------------------------------------------------------------------------
    # Snapshot / Polling
    # -------------------------------------------------------------------------

    async def get_current_loaded_models_snapshot(self):
        """Return snapshot of current_loaded_models for polling from Child."""
        result = []
        for i, lm in enumerate(mm.current_loaded_models):
            # Filter to cuda:0 only
            if str(lm.device) != "cuda:0":
                continue
            if lm.model is None:
                result.append({"index": i, "name": "DEAD", "id": None, "used": False, "mb": 0})
            else:
                name = lm.model.model.__class__.__name__ if hasattr(lm.model, 'model') else type(lm.model).__name__
                try:
                    size_mb = lm.model_memory() / (1024 * 1024)
                except:
                    size_mb = 0
                result.append({
                    "index": i,
                    "name": name,
                    "id": id(lm.model),
                    "used": lm.currently_used,
                    "mb": size_mb
                })
        return result
