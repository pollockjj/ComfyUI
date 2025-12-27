from __future__ import annotations

from typing import Optional, Any
import comfy.utils
from pyisolate import ProxiedSingleton

class UtilsProxy(ProxiedSingleton):
    """
    Proxy for comfy.utils.
    Primarily handles the PROGRESS_BAR_HOOK to ensure progress updates
    from isolated nodes reach the host.
    """
    
    def progress_bar_hook(self, value: int, total: int, preview: Optional[bytes] = None, node_id: Optional[str] = None) -> None:
        """
        Host-side implementation: forwards the call to the real global hook.
        Child-side: this method call is intercepted by RPC and sent to host.
        """
        if comfy.utils.PROGRESS_BAR_HOOK is not None:
            comfy.utils.PROGRESS_BAR_HOOK(value, total, preview, node_id)

    def set_progress_bar_global_hook(self, hook: Any) -> None:
        """Forward hook registration (though usually not needed from child)."""
        comfy.utils.set_progress_bar_global_hook(hook)
