from __future__ import annotations

from typing import Optional, Any
import comfy.utils
from pyisolate import ProxiedSingleton

import os

class UtilsProxy(ProxiedSingleton):
    """
    Proxy for comfy.utils.
    Primarily handles the PROGRESS_BAR_HOOK to ensure progress updates
    from isolated nodes reach the host.
    """
    
    _instance: Optional['UtilsProxy'] = None

    def __new__(cls) -> 'UtilsProxy':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @property
    def instance(self) -> 'UtilsProxy':
        return self

    async def progress_bar_hook(self, value: int, total: int, preview: Optional[bytes] = None, node_id: Optional[str] = None) -> Any:
        """
        Host-side implementation: forwards the call to the real global hook.
        Child-side: this method call is intercepted by RPC and sent to host.
        """
        # Manual RPC dispatch for Child process
        if os.environ.get("PYISOLATE_CHILD") == "1":
            import logging
            logger = logging.getLogger(__name__)
            # logger.info(f"[TRACE:PBAR] D: UtilsProxy (Child) instance={id(self)} RPC={hasattr(self, '_rpc') and self._rpc is not None}")
            
            # Primary channel: injected _rpc
            if hasattr(self, "_rpc") and self._rpc:
                return await self._rpc.call_remote("progress_bar_hook", value, total, preview, node_id)
            
            # Fallback channel: current context (e.g. if we are in a callback but instance is fresh)
            try:
                from pyisolate._internal.shared import current_rpc_context
                rpc = current_rpc_context.get()
                if rpc:
                    # logger.info(f"[TRACE:PBAR] D: Using fallback RPC context")
                    return await rpc.call_remote("progress_bar_hook", value, total, preview, node_id)
            except (ImportError, LookupError):
                pass

            logger.error(f"[TRACE:PBAR] D: RPC MISSING IN CHILD UtilsProxy! instance={id(self)}")
            return None

        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"[TRACE:PBAR] G: UtilsProxy.progress_bar_hook (Host) called. value={value}/{total}")
        
        if comfy.utils.PROGRESS_BAR_HOOK is not None:
            logger.info(f"[TRACE:PBAR] H: Forwarding to global hook: {comfy.utils.PROGRESS_BAR_HOOK}")
            comfy.utils.PROGRESS_BAR_HOOK(value, total, preview, node_id)
        else:
            logger.warning("[TRACE:PBAR] H: GLOBAL HOOK IS NONE on Host!")

    def set_progress_bar_global_hook(self, hook: Any) -> None:
        """Forward hook registration (though usually not needed from child)."""
        comfy.utils.set_progress_bar_global_hook(hook)
