from __future__ import annotations

import logging
from typing import Any, Optional

try:
    from pyisolate import ProxiedSingleton
except ImportError:  # pragma: no cover - fallback when pyisolate unavailable
    class ProxiedSingleton:  # type: ignore
        pass

from comfy_execution.progress import get_progress_state

LOG_PREFIX = "]["
logger = logging.getLogger(__name__)


class ProgressProxy(ProxiedSingleton):
    """Proxy to forward progress updates from isolated children to host UI."""

    def set_progress(
        self,
        value: float,
        max_value: float,
        node_id: Optional[str] = None,
        image: Any = None,
    ) -> None:
        try:
            get_progress_state().update_progress(
                node_id=node_id,
                value=value,
                max_value=max_value,
                image=image,
            )
        except Exception as exc:  # pragma: no cover
            logger.error("%s[ProgressProxy] Failed to update progress: %s", LOG_PREFIX, exc)
            raise


__all__ = ["ProgressProxy"]
