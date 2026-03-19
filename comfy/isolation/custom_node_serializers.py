"""Serializers for custom node data types.

These serializers exist because of specific custom node conversions and are
not required by core ComfyUI.  Types defined in comfy_api (File3D, VIDEO,
etc.) belong in adapter.py.  Only types invented by custom node packs that
don't exist in comfy_api belong here.

New custom node conversions that introduce types not covered here should
add their serializers to this file.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from pyisolate.interfaces import SerializerRegistryProtocol  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)

_announced: set[str] = set()


def _announce(name: str, desc: str) -> None:
    if name not in _announced:
        _announced.add(name)
        logger.info("][ Serializer: %s — %s", name, desc)


def register_custom_node_serializers(registry: SerializerRegistryProtocol) -> None:
    """Register all custom-node-originated serializers."""

    # -- PLY (comfy_api.latest._util.ply_types) --------------------------------
    # PLY point cloud container created for DA3 isolation conversion.
    # Origin: pollockjj for ComfyUI-DepthAnythingV3 isolation (commit 99d90b29)
    # Used by: ComfyUI-DepthAnythingV3, ComfyUI-GeometryPack

    def serialize_ply(obj: Any) -> Dict[str, Any]:
        _announce("PLY", "PLY (by pollockjj for DA3 isolation) serializer 1.0 (base64/tensors) for ComfyUI-DepthAnythingV3, ComfyUI-GeometryPack")
        import base64
        import torch
        if obj.raw_data is not None:
            return {
                "__type__": "PLY",
                "raw_data": base64.b64encode(obj.raw_data).decode("ascii"),
            }
        result: Dict[str, Any] = {"__type__": "PLY", "points": torch.from_numpy(obj.points)}
        if obj.colors is not None:
            result["colors"] = torch.from_numpy(obj.colors)
        if obj.confidence is not None:
            result["confidence"] = torch.from_numpy(obj.confidence)
        if obj.view_id is not None:
            result["view_id"] = torch.from_numpy(obj.view_id)
        return result

    def deserialize_ply(data: Any) -> Any:
        import base64
        from comfy_api.latest._util.ply_types import PLY
        if "raw_data" in data:
            return PLY(raw_data=base64.b64decode(data["raw_data"]))
        return PLY(
            points=data["points"],
            colors=data.get("colors"),
            confidence=data.get("confidence"),
            view_id=data.get("view_id"),
        )

    registry.register("PLY", serialize_ply, deserialize_ply, data_type=True)

    # -- NPZ (comfy_api.latest._util.npz_types) --------------------------------
    # NPZ depth map frame container created for DA3 isolation conversion.
    # Origin: pollockjj for ComfyUI-DepthAnythingV3 isolation (commit 99d90b29)
    # Used by: ComfyUI-DepthAnythingV3

    def serialize_npz(obj: Any) -> Dict[str, Any]:
        _announce("NPZ", "NPZ (by pollockjj for DA3 isolation) serializer 1.0 (base64 frames) for ComfyUI-DepthAnythingV3")
        import base64
        return {
            "__type__": "NPZ",
            "frames": [base64.b64encode(f).decode("ascii") for f in obj.frames],
        }

    def deserialize_npz(data: Any) -> Any:
        import base64
        from comfy_api.latest._util.npz_types import NPZ
        return NPZ(frames=[base64.b64decode(f) for f in data["frames"]])

    registry.register("NPZ", serialize_npz, deserialize_npz, data_type=True)
