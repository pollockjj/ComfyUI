"""Minimal session logging stub for parallel attention workflows.

The original implementation tracked detailed workflow metadata for
Ring/Ulysses runs. Phase A purged those features, but the standard
`SaveImage` node still imports `SessionLogger`. To keep the import
stable without reintroducing the legacy system, we provide a lightweight
placeholder that can be extended later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class _SessionState:
    """Internal state container for the singleton logger."""

    active: bool = False
    output_dir: Optional[str] = None


class SessionLogger:
    """Singleton no-op session logger used by `nodes.SaveImage`.

    The previous ring-attention implementation recorded extensive run
    metadata. That system has been removed, but the core still expects a
    `SessionLogger` interface. This stub keeps the API intact so the
    remaining code path can call `is_active()` and `finalize_session()`
    safely.
    """

    _instance: Optional["SessionLogger"] = None

    def __init__(self) -> None:
        self._state = _SessionState()

    @classmethod
    def get_instance(cls) -> "SessionLogger":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # Legacy entry point retained for compatibility. Phase A does not
    # activate the logger, but downstream code may call it defensively.
    def start_session(self, output_dir: Optional[str] = None) -> None:
        self._state.active = True
        self._state.output_dir = output_dir

    def is_active(self) -> bool:
        return self._state.active

    def finalize_session(self, output_dir: Optional[str] = None) -> None:
        if not self._state.active:
            return
        # Prefer latest non-empty directory value.
        if output_dir:
            self._state.output_dir = output_dir
        # Placeholder: in the legacy system this is where JSON metadata was
        # written. We intentionally leave it empty while keeping the hook.
        self._state.active = False

    # Convenience helper mirroring the old API.
    def get_output_dir(self) -> Optional[str]:
        return self._state.output_dir
