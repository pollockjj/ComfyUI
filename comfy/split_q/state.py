"""
Split-Q State Management
Persistent CUDA streams and events for dual-GPU parallel attention.
"""

import torch
import logging
from dataclasses import dataclass, field
from typing import Callable, Optional, List

_logger = logging.getLogger('comfy')


@dataclass
class SplitQState:
    """
    State object cached in model_options to manage Split-Q parallelism.
    Instantiated once per sampling run. Holds persistent CUDA streams/events
    to avoid per-call recreation overhead.
    
    Architecture: Hybrid State-Seam (Blueprint Section 1)
    - Created by: CFGGuider.outer_sample hook (post-pre_run)
    - Consumed by: optimized_attention_override (per-attention-call)
    """
    enabled: bool = True
    validation_mode: bool = False
    
    # Devices
    device_0: torch.device = field(default_factory=lambda: torch.device("cuda:0"))
    device_1: torch.device = field(default_factory=lambda: torch.device("cuda:1"))
    
    # Original function for fallback/validation
    original_attn_func: Optional[Callable] = None
    
    # Persistent CUDA objects (created in __post_init__)
    stream_0: Optional[torch.cuda.Stream] = None
    stream_1: Optional[torch.cuda.Stream] = None
    
    # Events for pipeline synchronization (Blueprint Table 1)
    event_k_replicated: Optional[torch.cuda.Event] = None
    event_v_replicated: Optional[torch.cuda.Event] = None
    event_attn_0_done: Optional[torch.cuda.Event] = None
    event_attn_1_done: Optional[torch.cuda.Event] = None
    
    # Telemetry (Phase 3)
    perf_timings: List[float] = field(default_factory=list)
    
    def __post_init__(self):
        """
        Create persistent CUDA objects on initialization.
        Called automatically by dataclass after __init__.
        
        Rationale (Blueprint Section 3.1):
        - Stream/event creation is non-trivial CPU overhead
        - Attention is called hundreds of times per sample
        - Create once, reuse across all calls
        """
        if self.enabled:
            try:
                # Create streams on respective devices
                self.stream_0 = torch.cuda.Stream(device=self.device_0)
                self.stream_1 = torch.cuda.Stream(device=self.device_1)
                
                # Create synchronization events
                self.event_k_replicated = torch.cuda.Event()
                self.event_v_replicated = torch.cuda.Event()
                self.event_attn_0_done = torch.cuda.Event()
                self.event_attn_1_done = torch.cuda.Event()
                
                _logger.info("⚡ [split-q][state] CUDA streams and events initialized successfully")
                _logger.info(f"⚡ [split-q][state] device_0={self.device_0}, device_1={self.device_1}")
                
            except Exception as e:
                _logger.error(f"⚡ [split-q][state] Failed to initialize CUDA streams/events: {e}")
                _logger.error(f"⚡ [split-q][state] Disabling Split-Q")
                self.enabled = False
    
    def is_ready(self) -> bool:
        """
        Check if state is valid for parallel execution.
        
        Returns:
            bool: True if all required objects are initialized
        """
        ready = (
            self.enabled and
            self.original_attn_func is not None and
            self.stream_0 is not None and
            self.stream_1 is not None
        )
        
        if not ready and self.enabled:
            _logger.warning("⚡ [split-q][state] is_ready() check failed")
            _logger.warning(f"⚡ [split-q][state] enabled={self.enabled}, "
                          f"original_attn_func={'set' if self.original_attn_func else 'None'}, "
                          f"stream_0={'set' if self.stream_0 else 'None'}, "
                          f"stream_1={'set' if self.stream_1 else 'None'}")
        
        return ready
    
    def __repr__(self):
        """Debug representation."""
        return (f"SplitQState(enabled={self.enabled}, "
                f"validation_mode={self.validation_mode}, "
                f"ready={self.is_ready()})")
