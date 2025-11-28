"""
Stateless RPC pattern for model_sampling instances.

This module provides:
1. ModelSamplingRegistry - Host-side registry of ModelSampling instances (ProxiedSingleton)
2. ModelSamplingProxy - Picklable handle that forwards calls via RPC

Architecture:
- Host process creates ModelSampling instances and registers them in the registry
- Isolated processes receive ModelSamplingProxy objects that contain only an ID string
- All method calls are forwarded via RPC to the host process
- Lifecycle is managed via weakref.finalize to prevent memory leaks

Critical Implementation Rules:
1. ALL proxy methods MUST use RpcBridge.run_sync() - never loop.run_until_complete()
2. NO direct access to registry._registry from proxy methods (it's an RPC proxy in child)
3. Registry operations MUST be protected by threading.Lock
4. Host-side proxies manage lifecycle; child-side proxies do not
"""

import logging
import os
import threading
import weakref
from typing import Any, Dict, Optional, TYPE_CHECKING

from .rpc_bridge import get_rpc_bridge

# Import PyIsolate at runtime only when needed
if TYPE_CHECKING:
    from pyisolate import ProxiedSingleton
else:
    try:
        from pyisolate import ProxiedSingleton
    except ImportError:
        class ProxiedSingleton:
            """Fallback when PyIsolate not available."""
            pass

logger = logging.getLogger(__name__)

# Flag to detect child process context
IS_CHILD_PROCESS = os.environ.get("PYISOLATE_CHILD") == "1"


class ModelSamplingRegistry(ProxiedSingleton):
    """
    Host-side registry of ModelSampling instances.
    
    Provides RPC methods for isolated processes to call ModelSampling
    methods without serializing the actual instances.
    
    Thread-safe: All registry operations are protected by a lock.
    """
    
    def __init__(self):
        if hasattr(ProxiedSingleton, "__init__") and ProxiedSingleton is not object:
            super().__init__()
        self._registry: Dict[str, Any] = {}
        self._counter = 0
        self._lock = threading.Lock()
    
    def register(self, model_sampling_instance) -> str:
        """
        Register a model_sampling instance and return its ID.
        
        Thread-safe. Called during model creation.
        
        Args:
            model_sampling_instance: Instance from model_sampling() factory
        
        Returns:
            Unique ID string for this instance
        
        Raises:
            RuntimeError: If called from child process
        """
        if IS_CHILD_PROCESS:
            raise RuntimeError(
                "📚 [PyIsolate][ModelSampling] FAIL-LOUD: Cannot register in child process"
            )
        
        with self._lock:
            instance_id = f"model_sampling_{self._counter}"
            self._counter += 1
            self._registry[instance_id] = model_sampling_instance
        
        logger.debug(
            "📚 [PyIsolate][ModelSampling] ✅ Registered %s (type: %s)",
            instance_id,
            type(model_sampling_instance).__name__,
        )
        return instance_id
    
    def unregister_sync(self, instance_id: str) -> None:
        """
        Synchronous unregister for use in weakref.finalize callbacks.
        
        Thread-safe. Called during garbage collection of host-side proxy.
        
        Args:
            instance_id: The ID of the instance to unregister
        """
        with self._lock:
            if instance_id in self._registry:
                del self._registry[instance_id]
                logger.debug(
                    "📚 [PyIsolate][ModelSampling] Unregistered %s (finalizer)",
                    instance_id,
                )
    
    def _get_instance(self, instance_id: str):
        """
        Get instance by ID. Internal use only.
        
        Thread-safe read access.
        
        Raises:
            ValueError: If instance_id not found
        """
        with self._lock:
            if instance_id not in self._registry:
                raise ValueError(f"Unknown model_sampling instance: {instance_id}")
            return self._registry[instance_id]
    
    # --- RPC Methods (async, called from isolated processes) ---
    
    async def calculate_input(self, instance_id: str, sigma, noise):
        """RPC: model_sampling.calculate_input()"""
        instance = self._get_instance(instance_id)
        return instance.calculate_input(sigma, noise)
    
    async def calculate_denoised(self, instance_id: str, sigma, model_output, model_input):
        """RPC: model_sampling.calculate_denoised()"""
        instance = self._get_instance(instance_id)
        return instance.calculate_denoised(sigma, model_output, model_input)
    
    async def noise_scaling(self, instance_id: str, sigma, noise, latent_image, max_denoise=False):
        """RPC: model_sampling.noise_scaling()"""
        instance = self._get_instance(instance_id)
        return instance.noise_scaling(sigma, noise, latent_image, max_denoise)
    
    async def timestep(self, instance_id: str, sigma):
        """RPC: model_sampling.timestep()"""
        instance = self._get_instance(instance_id)
        return instance.timestep(sigma)
    
    async def sigma(self, instance_id: str, timestep):
        """RPC: model_sampling.sigma()"""
        instance = self._get_instance(instance_id)
        return instance.sigma(timestep)
    
    async def percent_to_sigma(self, instance_id: str, percent: float) -> float:
        """RPC: model_sampling.percent_to_sigma()"""
        instance = self._get_instance(instance_id)
        return instance.percent_to_sigma(percent)
    
    async def get_sigma_min(self, instance_id: str):
        """RPC: Get sigma_min property."""
        instance = self._get_instance(instance_id)
        val = instance.sigma_min
        return val.item() if hasattr(val, "item") else val
    
    async def get_sigma_max(self, instance_id: str):
        """RPC: Get sigma_max property."""
        instance = self._get_instance(instance_id)
        val = instance.sigma_max
        return val.item() if hasattr(val, "item") else val
    
    async def get_sigma_data(self, instance_id: str):
        """RPC: Get sigma_data property."""
        instance = self._get_instance(instance_id)
        return getattr(instance, "sigma_data", None)
    
    async def get_sigmas(self, instance_id: str):
        """RPC: Get sigmas tensor."""
        instance = self._get_instance(instance_id)
        return instance.sigmas
    
    async def get_log_sigmas(self, instance_id: str):
        """RPC: Get log_sigmas tensor."""
        instance = self._get_instance(instance_id)
        return instance.log_sigmas


class ModelSamplingProxy:
    """
    Lightweight, picklable handle to a ModelSampling instance.
    
    Forwards all method calls via RPC to the host process.
    Only the instance_id string is serialized.
    
    Critical: ALL methods use RpcBridge.run_sync() for RPC calls.
    Never access self._registry._registry directly (it doesn't exist in child).
    
    Args:
        instance_id: Unique ID from ModelSamplingRegistry.register()
        registry: The ModelSamplingRegistry (may be RPC proxy in child)
        manage_lifecycle: If True, attach weakref.finalize for cleanup (host only)
    """
    
    def __init__(
        self,
        instance_id: str,
        registry: ModelSamplingRegistry,
        manage_lifecycle: bool = False,
    ):
        self._instance_id = instance_id
        self._registry = registry
        self._manage_lifecycle = manage_lifecycle
        
        # Cached properties (lazily loaded via RPC)
        self._cached_sigma_min: Optional[float] = None
        self._cached_sigma_max: Optional[float] = None
        self._cached_sigma_data: Optional[float] = None
        
        # Lifecycle management: only host-side proxy cleans up
        if manage_lifecycle and not IS_CHILD_PROCESS:
            # Note: weakref.finalize calls unregister_sync when this proxy is GC'd
            self._finalizer = weakref.finalize(
                self,
                registry.unregister_sync,
                instance_id,
            )
            logger.debug(
                "📚 [PyIsolate][ModelSampling] Lifecycle managed for %s",
                instance_id,
            )
    
    def __reduce__(self):
        """
        Custom pickle support - only serialize the instance_id.
        
        On unpickle, the proxy reconnects to the registry via RPC.
        manage_lifecycle=False because child proxies don't clean up.
        """
        return (
            _reconstruct_model_sampling_proxy,
            (self._instance_id,),
        )
    
    def _run_rpc(self, coro):
        """
        Execute an async RPC call synchronously.
        
        Uses the global RpcBridge to avoid nested event loop errors.
        """
        bridge = get_rpc_bridge()
        return bridge.run_sync(coro)
    
    # --- Forwarded Methods (all use RPC) ---
    
    def calculate_input(self, sigma, noise):
        """Forward to registry via RPC."""
        return self._run_rpc(
            self._registry.calculate_input(self._instance_id, sigma, noise)
        )
    
    def calculate_denoised(self, sigma, model_output, model_input):
        """Forward to registry via RPC."""
        return self._run_rpc(
            self._registry.calculate_denoised(
                self._instance_id, sigma, model_output, model_input
            )
        )
    
    def noise_scaling(self, sigma, noise, latent_image, max_denoise=False):
        """Forward to registry via RPC."""
        return self._run_rpc(
            self._registry.noise_scaling(
                self._instance_id, sigma, noise, latent_image, max_denoise
            )
        )
    
    def timestep(self, sigma):
        """Forward to registry via RPC."""
        return self._run_rpc(
            self._registry.timestep(self._instance_id, sigma)
        )
    
    def sigma(self, timestep):
        """Forward to registry via RPC."""
        return self._run_rpc(
            self._registry.sigma(self._instance_id, timestep)
        )
    
    def percent_to_sigma(self, percent: float) -> float:
        """Forward to registry via RPC."""
        return self._run_rpc(
            self._registry.percent_to_sigma(self._instance_id, percent)
        )
    
    # --- Properties (cached after first RPC call) ---
    
    @property
    def sigma_min(self):
        """Get sigma_min (cached after first access)."""
        if self._cached_sigma_min is None:
            self._cached_sigma_min = self._run_rpc(
                self._registry.get_sigma_min(self._instance_id)
            )
        return self._cached_sigma_min
    
    @property
    def sigma_max(self):
        """Get sigma_max (cached after first access)."""
        if self._cached_sigma_max is None:
            self._cached_sigma_max = self._run_rpc(
                self._registry.get_sigma_max(self._instance_id)
            )
        return self._cached_sigma_max
    
    @property
    def sigma_data(self):
        """Get sigma_data (cached after first access)."""
        if self._cached_sigma_data is None:
            self._cached_sigma_data = self._run_rpc(
                self._registry.get_sigma_data(self._instance_id)
            )
        return self._cached_sigma_data
    
    @property
    def sigmas(self):
        """Get sigmas tensor (NOT cached - may be large)."""
        return self._run_rpc(
            self._registry.get_sigmas(self._instance_id)
        )
    
    @property
    def log_sigmas(self):
        """Get log_sigmas tensor (NOT cached - may be large)."""
        return self._run_rpc(
            self._registry.get_log_sigmas(self._instance_id)
        )


def _reconstruct_model_sampling_proxy(instance_id: str) -> ModelSamplingProxy:
    """
    Pickle reconstruction helper.
    
    Called when unpickling a ModelSamplingProxy in the child process.
    The registry will be accessed via RPC automatically through ProxiedSingleton.
    
    Args:
        instance_id: The ID of the registered ModelSampling instance
    
    Returns:
        A new ModelSamplingProxy connected to the host's registry
    """
    try:
        registry = ModelSamplingRegistry()
        # manage_lifecycle=False: child proxies don't trigger cleanup
        return ModelSamplingProxy(instance_id, registry, manage_lifecycle=False)
    except Exception as e:
        logger.error(
            "📚 [PyIsolate][ModelSampling] Failed to reconstruct proxy for %s: %s",
            instance_id,
            e,
        )
        raise
