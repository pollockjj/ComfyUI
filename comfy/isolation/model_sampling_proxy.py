from __future__ import annotations

import asyncio
import logging
from typing import Any

from comfy.isolation.proxies.base import (
    IS_CHILD_PROCESS,
    BaseProxy,
    BaseRegistry,
    detach_if_grad,
    get_thread_loop,
    run_coro_in_new_loop,
)

logger = logging.getLogger(__name__)


def _prefer_device(*tensors: Any) -> Any:
    try:
        import torch
    except Exception:
        return None
    for t in tensors:
        if isinstance(t, torch.Tensor) and t.is_cuda:
            return t.device
    for t in tensors:
        if isinstance(t, torch.Tensor):
            return t.device
    return None


def _to_device(obj: Any, device: Any) -> Any:
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


class ModelSamplingRegistry(BaseRegistry[Any]):
    _type_prefix = "modelsampling"

    async def calculate_input(self, instance_id: str, sigma: Any, noise: Any) -> Any:
        sampling = self._get_instance(instance_id)
        return detach_if_grad(sampling.calculate_input(sigma, noise))

    async def calculate_denoised(
        self, instance_id: str, sigma: Any, model_output: Any, model_input: Any
    ) -> Any:
        sampling = self._get_instance(instance_id)
        return detach_if_grad(sampling.calculate_denoised(sigma, model_output, model_input))

    async def noise_scaling(
        self, instance_id: str, sigma: Any, noise: Any, latent_image: Any, max_denoise: bool = False
    ) -> Any:
        sampling = self._get_instance(instance_id)
        return detach_if_grad(sampling.noise_scaling(sigma, noise, latent_image, max_denoise=max_denoise))

    async def inverse_noise_scaling(self, instance_id: str, sigma: Any, latent: Any) -> Any:
        sampling = self._get_instance(instance_id)
        return detach_if_grad(sampling.inverse_noise_scaling(sigma, latent))

    async def timestep(self, instance_id: str, sigma: Any) -> Any:
        sampling = self._get_instance(instance_id)
        return sampling.timestep(sigma)

    async def sigma(self, instance_id: str, timestep: Any) -> Any:
        sampling = self._get_instance(instance_id)
        return sampling.sigma(timestep)

    async def percent_to_sigma(self, instance_id: str, percent: float) -> Any:
        sampling = self._get_instance(instance_id)
        return sampling.percent_to_sigma(percent)

    async def get_attr(self, instance_id: str, name: str) -> Any:
        sampling = self._get_instance(instance_id)
        return getattr(sampling, name)


class ModelSamplingProxy(BaseProxy[ModelSamplingRegistry]):
    _registry_class = ModelSamplingRegistry
    __module__ = "comfy.isolation.model_sampling_proxy"

    def _get_rpc(self) -> Any:
        if self._rpc_caller is None:
            from pyisolate._internal.shared import get_child_rpc_instance

            rpc = get_child_rpc_instance()
            if rpc is not None:
                self._rpc_caller = rpc.create_caller(
                    ModelSamplingRegistry, ModelSamplingRegistry.get_remote_id()
                )
            else:
                registry = ModelSamplingRegistry()

                class _LocalCaller:
                    def calculate_input(self_inner: Any, instance_id: str, sigma: Any, noise: Any) -> Any:
                        return registry.calculate_input(instance_id, sigma, noise)

                    def calculate_denoised(
                        self_inner: Any, instance_id: str, sigma: Any, model_output: Any, model_input: Any
                    ) -> Any:
                        return registry.calculate_denoised(instance_id, sigma, model_output, model_input)

                    def noise_scaling(
                        self_inner: Any,
                        instance_id: str,
                        sigma: Any,
                        noise: Any,
                        latent_image: Any,
                        max_denoise: bool = False,
                    ) -> Any:
                        return registry.noise_scaling(instance_id, sigma, noise, latent_image, max_denoise)

                    def inverse_noise_scaling(self_inner: Any, instance_id: str, sigma: Any, latent: Any) -> Any:
                        return registry.inverse_noise_scaling(instance_id, sigma, latent)

                    def timestep(self_inner: Any, instance_id: str, sigma: Any) -> Any:
                        return registry.timestep(instance_id, sigma)

                    def sigma(self_inner: Any, instance_id: str, timestep: Any) -> Any:
                        return registry.sigma(instance_id, timestep)

                    def percent_to_sigma(self_inner: Any, instance_id: str, percent: float) -> Any:
                        return registry.percent_to_sigma(instance_id, percent)

                    def get_attr(self_inner: Any, instance_id: str, name: str) -> Any:
                        return registry.get_attr(instance_id, name)

                self._rpc_caller = _LocalCaller()
        return self._rpc_caller

    def _call(self, method_name: str, *args: Any) -> Any:
        rpc = self._get_rpc()
        method = getattr(rpc, method_name)
        result = method(self._instance_id, *args)
        if asyncio.iscoroutine(result):
            try:
                asyncio.get_running_loop()
                return run_coro_in_new_loop(result)
            except RuntimeError:
                loop = get_thread_loop()
                return loop.run_until_complete(result)
        return result

    def calculate_input(self, sigma: Any, noise: Any) -> Any:
        return self._call("calculate_input", sigma, noise)

    def calculate_denoised(self, sigma: Any, model_output: Any, model_input: Any) -> Any:
        return self._call("calculate_denoised", sigma, model_output, model_input)

    def noise_scaling(self, sigma: Any, noise: Any, latent_image: Any, max_denoise: bool = False) -> Any:
        return self._call("noise_scaling", sigma, noise, latent_image, max_denoise)

    def inverse_noise_scaling(self, sigma: Any, latent: Any) -> Any:
        return self._call("inverse_noise_scaling", sigma, latent)

    def timestep(self, sigma: Any) -> Any:
        return self._call("timestep", sigma)

    def sigma(self, timestep: Any) -> Any:
        return self._call("sigma", timestep)

    def percent_to_sigma(self, percent: float) -> Any:
        return self._call("percent_to_sigma", percent)

    def get_attr(self, name: str) -> Any:
        return self._call("get_attr", name)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return self._call("get_attr", name)

