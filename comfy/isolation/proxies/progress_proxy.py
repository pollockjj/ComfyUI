from __future__ import annotations

import logging
from typing import Any, Optional

from pyisolate import ProxiedSingleton

from comfy_execution.progress import get_progress_state

logger = logging.getLogger(__name__)


class ProgressProxy(ProxiedSingleton):
    def set_progress(
        self,
        value: float,
        max_value: float,
        node_id: Optional[str] = None,
        image: Any = None,
    ) -> None:
        get_progress_state().update_progress(
            node_id=node_id,
            value=value,
            max_value=max_value,
            image=image,
        )


__all__ = ["ProgressProxy"]
