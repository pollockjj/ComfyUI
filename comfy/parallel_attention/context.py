"""
Parallel Attention Context - Single Source of Truth

This module provides a unified context object for parallel attention state management.
All parallel attention data flows through this single object, making it 100% transparent
to the rest of ComfyUI.

Design Principle: ONE object, attached ONCE, populated progressively.
"""

from typing import Optional, Any
import logging


class ParallelAttentionContext:
    """
    Single source of truth for all parallel attention state.
    
    Attached to ModelPatcher and preserved through clones.
    Populated progressively through the model lifecycle:
    1. Structure capture (sd.py)
    2. Worker initialization (config node)
    3. Tensor sharding (model_management.py)
    
    The rest of ComfyUI sees this as an opaque object.
    """
    
    def __init__(self):
        """Initialize empty context. Populated as we progress through lifecycle."""
        # Phase A: Structure Capture (sd.py)
        self.checkpoint_path: Optional[str] = None
        self.meta_model: Optional[Any] = None  # Meta device model (0GB)
        self.model_type: Optional[str] = None  # "flux", "wan", etc.
        self.policy: Optional[Any] = None  # FSDP2 sharding policy
        
        # Phase B: Worker Initialization (config node)
        self.executor: Optional[Any] = None  # FSDP2Executor with workers
        self.device_mesh: Optional[Any] = None  # DeviceMesh topology
        self.world_size: int = 0
        self.backend: str = "nccl"
        
        # Phase C: Tensor Sharding (model_management.py)
        self.sharded: bool = False
        self.vram_per_gpu: float = 0.0
        self.total_params: int = 0
        self.sharded_params: int = 0
        
        # Lifecycle
        self.enabled: bool = True
        self.phase: str = "init"
    
    def is_ready_for_sharding(self) -> bool:
        """Check if we have everything needed for FSDP2 sharding."""
        return all([
            self.checkpoint_path is not None,
            self.meta_model is not None,
            self.model_type is not None,
            self.policy is not None,
            self.executor is not None,
        ])
    
    def __repr__(self) -> str:
        """Debug representation."""
        return (
            f"ParallelAttentionContext("
            f"model_type={self.model_type}, "
            f"phase={self.phase}, "
            f"sharded={self.sharded}, "
            f"world_size={self.world_size})"
        )
    
    def log_state(self, prefix: str = "⚡ [Parallel-Attention][Context]"):
        """Log current state for debugging."""
        logging.info(f"{prefix} {self}")
        logging.info(f"{prefix}   checkpoint_path: {self.checkpoint_path}")
        logging.info(f"{prefix}   meta_model: {'✓' if self.meta_model else '✗'}")
        logging.info(f"{prefix}   model_type: {self.model_type}")
        logging.info(f"{prefix}   policy: {'✓' if self.policy else '✗'}")
        logging.info(f"{prefix}   executor: {'✓' if self.executor else '✗'}")
        logging.info(f"{prefix}   ready_for_sharding: {self.is_ready_for_sharding()}")
