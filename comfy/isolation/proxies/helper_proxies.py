from __future__ import annotations

from typing import Any, Dict, Optional


class AnyTypeProxy(str):
    """Replacement for custom AnyType objects used by some nodes."""

    def __new__(cls, value: str = "*"):
        return super().__new__(cls, value)

    def __ne__(self, other):  # type: ignore[override]
        return False


class FlexibleOptionalInputProxy(dict):
    """Replacement for FlexibleOptionalInputType to allow dynamic inputs."""

    def __init__(self, flex_type, data: Optional[Dict[str, object]] = None):
        super().__init__()
        self.type = flex_type
        if data:
            self.update(data)

    def __getitem__(self, key):  # type: ignore[override]
        return (self.type,)

    def __contains__(self, key):  # type: ignore[override]
        return True


class ByPassTypeTupleProxy(tuple):
    """Replacement for ByPassTypeTuple to mirror wildcard fallback behavior."""

    def __new__(cls, values):
        return super().__new__(cls, values)

    def __getitem__(self, index):  # type: ignore[override]
        if index >= len(self):
            return AnyTypeProxy("*")
        return super().__getitem__(index)


def _restore_special_value(value: Any) -> Any:
    if isinstance(value, dict):
        if value.get("__pyisolate_any_type__"):
            return AnyTypeProxy(value.get("value", "*"))
        if value.get("__pyisolate_flexible_optional__"):
            flex_type = _restore_special_value(value.get("type"))
            data_raw = value.get("data")
            data = (
                {k: _restore_special_value(v) for k, v in data_raw.items()}
                if isinstance(data_raw, dict)
                else {}
            )
            return FlexibleOptionalInputProxy(flex_type, data)
        if value.get("__pyisolate_tuple__") is not None:
            return tuple(_restore_special_value(v) for v in value["__pyisolate_tuple__"])
        if value.get("__pyisolate_bypass_tuple__") is not None:
            return ByPassTypeTupleProxy(
                tuple(_restore_special_value(v) for v in value["__pyisolate_bypass_tuple__"])
            )
        return {k: _restore_special_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_restore_special_value(v) for v in value]
    return value


def restore_input_types(raw: Dict[str, object]) -> Dict[str, object]:
    """Restore serialized INPUT_TYPES payload back into ComfyUI-compatible objects."""

    if not isinstance(raw, dict):
        return raw  # type: ignore[return-value]

    restored: Dict[str, object] = {}
    for section, entries in raw.items():
        if isinstance(entries, dict) and entries.get("__pyisolate_flexible_optional__"):
            restored[section] = _restore_special_value(entries)
        elif isinstance(entries, dict):
            restored[section] = {k: _restore_special_value(v) for k, v in entries.items()}
        else:
            restored[section] = _restore_special_value(entries)
    return restored


__all__ = [
    "AnyTypeProxy",
    "FlexibleOptionalInputProxy",
    "ByPassTypeTupleProxy",
    "restore_input_types",
]
