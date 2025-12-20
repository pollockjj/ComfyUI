"""
RPC proxy/registry for ModelSampling instances.
Mirrors the VAE proxy pattern but with a minimal surface for sampling functions.
"""
from __future__ import annotations

import asyncio
import logging
import os
import threading
from typing import Any, Dict

from pyisolate import ProxiedSingleton

logger = logging.getLogger(__name__)

IS_CHILD_PROCESS = os.environ.get("PYISOLATE_CHILD") == "1"
_thread_local = threading.local()


def _get_thread_loop() -> asyncio.AbstractEventLoop:
    loop = getattr(_thread_local, "loop", None)
    if loop is None or loop.is_closed():
        loop = asyncio.new_event_loop()
        _thread_local.loop = loop
    return loop


def _run_coro_in_new_loop(coro):
    result_box: Dict[str, Any] = {}
    exc_box: Dict[str, BaseException] = {}

    def runner():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result_box["value"] = loop.run_until_complete(coro)
        except Exception as exc:  # noqa: BLE001
            exc_box["exc"] = exc
        finally:
            loop.close()

    t = threading.Thread(target=runner, daemon=True)
    t.start()
    t.join()
    if "exc" in exc_box:
        raise exc_box["exc"]
    return result_box.get("value")


def _detach_if_grad(obj):
    try:
        import torch
    except Exception:
        return obj

    if isinstance(obj, torch.Tensor):
        return obj.detach() if obj.requires_grad else obj
    if isinstance(obj, (list, tuple)):
        return type(obj)(_detach_if_grad(x) for x in obj)
    if isinstance(obj, dict):
        return {k: _detach_if_grad(v) for k, v in obj.items()}
    return obj


def _prefer_device(*tensors):
    """Return a preferred device from the first CUDA tensor, else first tensor device, else None."""
    for t in tensors:
        try:
            import torch
            if isinstance(t, torch.Tensor):
                if t.is_cuda:
                    return t.device
        except Exception:
            return None
    for t in tensors:
        try:
            import torch
            if isinstance(t, torch.Tensor):
                return t.device
        except Exception:
            return None
    return None


def _to_device(obj, device):
    try:
        import torch
    except Exception:
        return obj
    if device is None:
        return obj
    if isinstance(obj, torch.Tensor):
        if obj.device != device:
            return obj.to(device)
        return obj
    if isinstance(obj, (list, tuple)):
        converted = [_to_device(x, device) for x in obj]
        return type(obj)(converted) if isinstance(obj, tuple) else converted
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    return obj


class ModelSamplingRegistry(ProxiedSingleton):
    def __init__(self) -> None:
        if hasattr(ProxiedSingleton, "__init__") and ProxiedSingleton is not object:
            super().__init__()
        self._registry: Dict[str, Any] = {}
        self._id_map: Dict[int, str] = {}
        self._counter = 0
        self._lock = threading.Lock()
        logger.debug("[ModelSamplingRegistry] Initialized")

    def register(self, sampling) -> str:
        with self._lock:
            obj_id = id(sampling)
            if obj_id in self._id_map:
                existing = self._id_map[obj_id]
                logger.debug("[ModelSamplingRegistry] Reusing %s", existing)
                return existing
            instance_id = f"modelsampling_{self._counter}"
            self._counter += 1
            self._registry[instance_id] = sampling
            self._id_map[obj_id] = instance_id
            logger.debug("[ModelSamplingRegistry] Registered %s", instance_id)
            return instance_id

    def unregister_sync(self, instance_id: str) -> None:
        with self._lock:
            sampling = self._registry.pop(instance_id, None)
            if sampling is not None:
                self._id_map.pop(id(sampling), None)
                logger.debug("[ModelSamplingRegistry] Unregistered %s", instance_id)

    def _get(self, instance_id: str):
        if IS_CHILD_PROCESS:
            raise RuntimeError("[ModelSamplingRegistry] Accessed in child process")
        with self._lock:
            return self._registry.get(instance_id)

    def _get_instance(self, instance_id: str):
        return self._get(instance_id)

    async def calculate_input(self, instance_id: str, sigma, noise):
        sampling = self._get(instance_id)
        if sampling is None:
            raise ValueError(f"ModelSampling {instance_id} not found")
        return _detach_if_grad(sampling.calculate_input(sigma, noise))

    async def calculate_denoised(self, instance_id: str, sigma, model_output, model_input):
        sampling = self._get(instance_id)
        if sampling is None:
            raise ValueError(f"ModelSampling {instance_id} not found")
        return _detach_if_grad(sampling.calculate_denoised(sigma, model_output, model_input))

    async def noise_scaling(self, instance_id: str, sigma, noise, latent_image, max_denoise: bool = False):
        sampling = self._get(instance_id)
        if sampling is None:
            raise ValueError(f"ModelSampling {instance_id} not found")
        return _detach_if_grad(sampling.noise_scaling(sigma, noise, latent_image, max_denoise=max_denoise))

    async def inverse_noise_scaling(self, instance_id: str, sigma, latent):
        sampling = self._get(instance_id)
        if sampling is None:
            raise ValueError(f"ModelSampling {instance_id} not found")
        return _detach_if_grad(sampling.inverse_noise_scaling(sigma, latent))

    async def timestep(self, instance_id: str, sigma):
        sampling = self._get(instance_id)
        if sampling is None:
            raise ValueError(f"ModelSampling {instance_id} not found")
        return sampling.timestep(sigma)

    async def sigma(self, instance_id: str, timestep):
        sampling = self._get(instance_id)
        if sampling is None:
            raise ValueError(f"ModelSampling {instance_id} not found")
        return sampling.sigma(timestep)

    async def percent_to_sigma(self, instance_id: str, percent: float):
        sampling = self._get(instance_id)
        if sampling is None:
            raise ValueError(f"ModelSampling {instance_id} not found")
        return sampling.percent_to_sigma(percent)

    async def get_attr(self, instance_id: str, name: str):
        sampling = self._get(instance_id)
        if sampling is None:
            raise ValueError(f"ModelSampling {instance_id} not found")
        return getattr(sampling, name)


class ModelSamplingProxy:
    __module__ = "comfy.isolation.model_sampling_proxy"

    def __init__(self, instance_id: str):
        self._instance_id = instance_id
        self._rpc = None

    def _get_rpc(self):
        if self._rpc is None:
            from pyisolate._internal.shared import get_child_rpc_instance
            rpc = get_child_rpc_instance()
            if rpc is not None:
                self._rpc = rpc.create_caller(ModelSamplingRegistry, ModelSamplingRegistry.get_remote_id())
            else:
                # Host-side access: local registry shim implementing coroutine methods
                registry = ModelSamplingRegistry()

                class _LocalCaller:
                    def calculate_input(self_inner, instance_id, sigma, noise):
                        return registry.calculate_input(instance_id, sigma, noise)

                    def calculate_denoised(self_inner, instance_id, sigma, model_output, model_input):
                        return registry.calculate_denoised(instance_id, sigma, model_output, model_input)

                    def noise_scaling(self_inner, instance_id, sigma, noise, latent_image, max_denoise=False):
                        return registry.noise_scaling(instance_id, sigma, noise, latent_image, max_denoise)

                    def inverse_noise_scaling(self_inner, instance_id, sigma, latent):
                        return registry.inverse_noise_scaling(instance_id, sigma, latent)

                    def timestep(self_inner, instance_id, sigma):
                        return registry.timestep(instance_id, sigma)

                    def sigma(self_inner, instance_id, timestep):
                        return registry.sigma(instance_id, timestep)

                    def percent_to_sigma(self_inner, instance_id, percent):
                        return registry.percent_to_sigma(instance_id, percent)

                    def get_attr(self_inner, instance_id, name):
                        return registry.get_attr(instance_id, name)

                self._rpc = _LocalCaller()
        return self._rpc

    def calculate_input(self, sigma, noise):
        return self._call('calculate_input', sigma, noise)

    def calculate_denoised(self, sigma, model_output, model_input):
        return self._call('calculate_denoised', sigma, model_output, model_input)

    def noise_scaling(self, sigma, noise, latent_image, max_denoise: bool = False):
        return self._call('noise_scaling', sigma, noise, latent_image, max_denoise)

    def inverse_noise_scaling(self, sigma, latent):
        return self._call('inverse_noise_scaling', sigma, latent)

    def timestep(self, sigma):
        return self._call('timestep', sigma)

    def sigma(self, timestep):
        return self._call('sigma', timestep)

    def percent_to_sigma(self, percent: float):
        return self._call('percent_to_sigma', percent)

    def get_attr(self, name: str):
        return self._call('get_attr', name)

    def __getstate__(self):
        return {"_instance_id": self._instance_id}

    def __setstate__(self, state):
        self._instance_id = state.get("_instance_id")
        self._rpc = None
        logger.debug("[Child] ModelSamplingProxy %s initialized", self._instance_id)

    def __repr__(self):
        return f"<ModelSamplingProxy {self._instance_id}>"

    def __getattr__(self, name: str):
        """Fallback attribute access for scalar/array fields (e.g., sigma_max)."""
        return self._call('get_attr', name)

    def _call(self, method_name: str, *args):
        rpc = self._get_rpc()
        method = getattr(rpc, method_name)
        result = method(self._instance_id, *args)
        if asyncio.iscoroutine(result):
            try:
                asyncio.get_running_loop()
                return _run_coro_in_new_loop(result)
            except RuntimeError:
                loop = _get_thread_loop()
                return loop.run_until_complete(result)
        return result
