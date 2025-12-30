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
    
    # _instance and __new__ removed to rely on SingletonMetaclass
    _rpc: Optional[Any] = None

    @classmethod
    def set_rpc(cls, rpc: Any) -> None:
        # Create caller using class name as ID (standard for Singletons)
        cls._rpc = rpc.create_caller(cls, "UtilsProxy")



    async def progress_bar_hook(self, value: int, total: int, preview: Optional[bytes] = None, node_id: Optional[str] = None) -> Any:
        """
        Host-side implementation: forwards the call to the real global hook.
        Child-side: this method call is intercepted by RPC and sent to host.
        """
        if os.environ.get("PYISOLATE_CHILD") == "1":
            # Manual RPC dispatch for Child process
            # Use class-level RPC storage (Static Injection)
            if UtilsProxy._rpc:
                return await UtilsProxy._rpc.progress_bar_hook(value, total, preview, node_id)
            
            # Fallback channel: current context
            try:
                from pyisolate._internal.shared import current_rpc_context
                rpc = current_rpc_context.get()
                if rpc:
                    return await rpc.call_remote("progress_bar_hook", value, total, preview, node_id)
            except (ImportError, LookupError):
                pass
            
            return None

        # Host Execution
        if comfy.utils.PROGRESS_BAR_HOOK is not None:
            comfy.utils.PROGRESS_BAR_HOOK(value, total, preview, node_id)

    def set_progress_bar_global_hook(self, hook: Any) -> None:
        """Forward hook registration (though usually not needed from child)."""
        comfy.utils.set_progress_bar_global_hook(hook)
