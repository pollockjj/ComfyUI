"""ProxiedSingleton for model_sampling instances to fix pickle errors."""

import logging
import os
from typing import Dict, Any, Optional, TYPE_CHECKING

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

# Flag to prevent registration in child processes
IS_CHILD_PROCESS = os.environ.get("PYISOLATE_CHILD") == "1"


class ModelSamplingRegistry(ProxiedSingleton):
    """
    Registry of model_sampling instances accessible across process boundaries.
    
    Isolated nodes get remote handles to host's model_sampling instances
    instead of trying to pickle/unpickle the dynamic ModelSampling classes.
    """
    
    def __init__(self):
        if hasattr(ProxiedSingleton, "__init__") and ProxiedSingleton is not object:
            super().__init__()
        self._registry: Dict[str, Any] = {}
        self._counter = 0
    
    def register(self, model_sampling_instance) -> str:
        """
        Register a model_sampling instance and return its ID.
        
        NOTE: Synchronous method - called during model creation before event loop exists.
        
        Args:
            model_sampling_instance: Instance from model_sampling() factory
        
        Returns:
            Unique ID string for this instance
        """
        if IS_CHILD_PROCESS:
            raise RuntimeError(
                "📚 [PyIsolate][ModelSampling] FAIL-LOUD: Cannot register model_sampling in child process"
            )
        
        instance_id = f"model_sampling_{self._counter}"
        self._counter += 1
        self._registry[instance_id] = model_sampling_instance
        
        logger.info(
            "📚 [PyIsolate][ModelSampling] ✅ Registered instance %s (type: %s)",
            instance_id,
            type(model_sampling_instance).__name__,
        )
        return instance_id
    
    async def calculate_input(self, instance_id: str, sigma, noise):
        """Proxy for model_sampling.calculate_input()"""
        if instance_id not in self._registry:
            raise ValueError(f"Unknown model_sampling instance: {instance_id}")
        
        instance = self._registry[instance_id]
        return instance.calculate_input(sigma, noise)
    
    async def calculate_denoised(self, instance_id: str, sigma, model_output, model_input):
        """Proxy for model_sampling.calculate_denoised()"""
        if instance_id not in self._registry:
            raise ValueError(f"Unknown model_sampling instance: {instance_id}")
        
        instance = self._registry[instance_id]
        return instance.calculate_denoised(sigma, model_output, model_input)
    
    async def noise_scaling(self, instance_id: str, sigma, noise, latent_image, max_denoise=False):
        """Proxy for model_sampling.noise_scaling()"""
        if instance_id not in self._registry:
            raise ValueError(f"Unknown model_sampling instance: {instance_id}")
        
        instance = self._registry[instance_id]
        return instance.noise_scaling(sigma, noise, latent_image, max_denoise)
    
    async def timestep(self, instance_id: str, sigma):
        """Proxy for model_sampling.timestep()"""
        if instance_id not in self._registry:
            raise ValueError(f"Unknown model_sampling instance: {instance_id}")
        
        instance = self._registry[instance_id]
        return instance.timestep(sigma)
    
    async def sigma(self, instance_id: str, timestep):
        """Proxy for model_sampling.sigma()"""
        if instance_id not in self._registry:
            raise ValueError(f"Unknown model_sampling instance: {instance_id}")
        
        instance = self._registry[instance_id]
        return instance.sigma(timestep)
    
    async def percent_to_sigma(self, instance_id: str, percent: float) -> float:
        """Proxy for model_sampling.percent_to_sigma()"""
        if instance_id not in self._registry:
            raise ValueError(f"Unknown model_sampling instance: {instance_id}")
        
        instance = self._registry[instance_id]
        return instance.percent_to_sigma(percent)
    
    async def get_properties(self, instance_id: str) -> Dict[str, Any]:
        """Get read-only properties (sigma_min, sigma_max, etc.)"""
        if instance_id not in self._registry:
            raise ValueError(f"Unknown model_sampling instance: {instance_id}")
        
        instance = self._registry[instance_id]
        return {
            "sigma_min": instance.sigma_min.item() if hasattr(instance.sigma_min, "item") else instance.sigma_min,
            "sigma_max": instance.sigma_max.item() if hasattr(instance.sigma_max, "item") else instance.sigma_max,
            "sigma_data": getattr(instance, "sigma_data", None),
        }
    
    async def get_sigmas(self, instance_id: str):
        """Get sigmas tensor."""
        if instance_id not in self._registry:
            raise ValueError(f"Unknown model_sampling instance: {instance_id}")
        
        instance = self._registry[instance_id]
        return instance.sigmas
    
    async def get_log_sigmas(self, instance_id: str):
        """Get log_sigmas tensor."""
        if instance_id not in self._registry:
            raise ValueError(f"Unknown model_sampling instance: {instance_id}")
        
        instance = self._registry[instance_id]
        return instance.log_sigmas
    
    async def unregister(self, instance_id: str):
        """Remove instance from registry."""
        if instance_id in self._registry:
            del self._registry[instance_id]
            logger.debug("📚 [PyIsolate][ModelSampling] Unregistered %s", instance_id)


class ModelSamplingProxy:
    """
    Lightweight proxy object that can be pickled and sent across processes.
    Delegates all calls to ModelSamplingRegistry via RPC.
    """
    
    def __init__(self, instance_id: str, registry: ModelSamplingRegistry):
        self._instance_id = instance_id
        self._registry = registry
        self._properties = None  # Lazy-loaded
    
    def __reduce__(self):
        """Custom pickle support - only serialize the ID."""
        return (
            _reconstruct_model_sampling_proxy,
            (self._instance_id,)
        )
    
    def calculate_input(self, sigma, noise):
        """Forward to registry."""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # No event loop in current thread - create one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(
            self._registry.calculate_input(self._instance_id, sigma, noise)
        )
    
    def calculate_denoised(self, sigma, model_output, model_input):
        """Forward to registry."""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(
            self._registry.calculate_denoised(self._instance_id, sigma, model_output, model_input)
        )
    
    def noise_scaling(self, sigma, noise, latent_image, max_denoise=False):
        """Forward to registry."""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(
            self._registry.noise_scaling(self._instance_id, sigma, noise, latent_image, max_denoise)
        )
    
    def timestep(self, sigma):
        """Forward to real instance - SYNCHRONOUS (host process only)."""
        # We're always in host process here, just access registry directly
        if self._instance_id not in self._registry._registry:
            raise ValueError(f"Unknown model_sampling instance: {self._instance_id}")
        
        instance = self._registry._registry[self._instance_id]
        return instance.timestep(sigma)
    
    def sigma(self, timestep):
        """Forward to real instance - SYNCHRONOUS (host process only)."""
        # We're always in host process here, just access registry directly
        if self._instance_id not in self._registry._registry:
            raise ValueError(f"Unknown model_sampling instance: {self._instance_id}")
        
        instance = self._registry._registry[self._instance_id]
        return instance.sigma(timestep)
    
    def percent_to_sigma(self, percent: float):
        """Forward to real instance - SYNCHRONOUS (host process only)."""
        # We're always in host process here, just access registry directly
        if self._instance_id not in self._registry._registry:
            raise ValueError(f"Unknown model_sampling instance: {self._instance_id}")
        
        instance = self._registry._registry[self._instance_id]
        return instance.percent_to_sigma(percent)
    
    @property
    def sigma_min(self):
        """Cached property access."""
        if self._properties is None:
            import asyncio
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            self._properties = loop.run_until_complete(
                self._registry.get_properties(self._instance_id)
            )
        return self._properties["sigma_min"]
    
    @property
    def sigma_max(self):
        """Cached property access."""
        if self._properties is None:
            import asyncio
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            self._properties = loop.run_until_complete(
                self._registry.get_properties(self._instance_id)
            )
        return self._properties["sigma_max"]
    
    @property
    def sigma_data(self):
        """Cached property access - SYNCHRONOUS."""
        if self._properties is None:
            # Direct registry access in host process
            if self._instance_id not in self._registry._registry:
                raise ValueError(f"Unknown model_sampling instance: {self._instance_id}")
            
            instance = self._registry._registry[self._instance_id]
            self._properties = {
                "sigma_min": instance.sigma_min.item() if hasattr(instance.sigma_min, "item") else instance.sigma_min,
                "sigma_max": instance.sigma_max.item() if hasattr(instance.sigma_max, "item") else instance.sigma_max,
                "sigma_data": getattr(instance, "sigma_data", None),
            }
        return self._properties["sigma_data"]
    
    @property
    def sigmas(self):
        """Direct tensor access - NOT cached, fetched on demand."""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(
            self._registry.get_sigmas(self._instance_id)
        )
    
    @property
    def log_sigmas(self):
        """Direct tensor access - NOT cached, fetched on demand."""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(
            self._registry.get_log_sigmas(self._instance_id)
        )


def _reconstruct_model_sampling_proxy(instance_id: str):
    """Pickle reconstruction helper."""
    # In child process, registry will be accessed via RPC automatically
    # No need to call use_remote() - just instantiate and it will connect
    try:
        registry = ModelSamplingRegistry()
        return ModelSamplingProxy(instance_id, registry)
    except Exception as e:
        import logging
        logging.error(f"📚 [PyIsolate][ModelSampling] Failed to reconstruct proxy: {e}")
        raise
