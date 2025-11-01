"""DistributedModelWrapper - Delegates model operations to distributed workers.

Parent process holds meta device model (0 bytes), workers hold FSDP2 shards.
Forward pass and other operations delegated via RPC to workers.

Based on Raylight comfy_dist patterns, adapted for ComfyUI core.
"""

import logging

LOG_PREFIX = "⚡ [Parallel-Attention]"


class DistributedModelWrapper:
    """Wrapper for distributed model with meta device parent.
    
    Architecture:
    - Parent process: Meta device model (0 bytes, preserves structure/properties)
    - Worker processes: FSDP2-sharded model (actual weights split across GPUs)
    - Operations: Delegated to workers via RPC
    
    ComfyUI Integration:
    - Exposes same interface as regular MODEL
    - Properties (latent_format, dtype) read from meta parent
    - Forward pass delegated to workers
    - Memory reporting accounts for sharding
    
    Based on Raylight's distributed wrapper pattern.
    """
    
    def __init__(self, executor, parent_model):
        """Initialize distributed wrapper.
        
        Args:
            executor: MultiprocExecutor for worker communication
            parent_model: Meta device model (0 bytes, structure only)
        """
        self._executor = executor
        self._parent = parent_model
        
        logging.info(f"{LOG_PREFIX} [Wrapper] Created with meta parent")
    
    @property
    def model(self):
        """Return parent model for ComfyUI compatibility."""
        return self._parent
    
    @property
    def latent_format(self):
        """Get latent format from parent."""
        return self._parent.latent_format
    
    def get_dtype(self):
        """Get dtype from parent."""
        return self._parent.get_dtype()
    
    def __call__(self, *args, **kwargs):
        """Forward pass - delegate to workers.
        
        NOT IMPLEMENTED YET - Phase 2.3
        """
        raise NotImplementedError(
            "Forward pass not implemented yet. "
            "Phase 2.2.1 validates loading only. "
            "Forward pass in Phase 2.3."
        )
