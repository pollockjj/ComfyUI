"""Worker context for storing distributed state across RPC calls."""

from typing import Optional


class WorkerContext:
    """Global context for worker process state.
    
    Stores model and distributed state for access across RPC calls.
    This allows forward_pass handler to access the model loaded by
    load_fsdp_model handler.
    """
    model_patcher: Optional[object] = None
    device_mesh: Optional[object] = None
    rank: Optional[int] = None
    world_size: Optional[int] = None
