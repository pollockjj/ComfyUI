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
import os
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

    # -- TRIMESH (trimesh.Trimesh by Michael Dawson-Haggerty) -------------------
    # In-memory triangular mesh object from the trimesh library.
    # Origin: trimesh library (https://github.com/mikedh/trimesh), MIT license
    # Used by: ComfyUI-GeometryPack (62 nodes)

    def serialize_trimesh(obj: Any) -> Dict[str, Any]:
        _announce("TRIMESH", "trimesh.Trimesh (by Michael Dawson-Haggerty) serializer 1.0 (tensors, dict) for ComfyUI-GeometryPack")
        import torch
        from comfy_api.latest._util.trimesh_types import TrimeshData

        # Handle both trimesh.Trimesh and TrimeshData (host round-trip)
        if isinstance(obj, TrimeshData):
            td = obj
        else:
            td = TrimeshData.from_trimesh(obj)

        result: Dict[str, Any] = {
            "__type__": "TRIMESH",
            "vertices": torch.from_numpy(td.vertices),
            "faces": torch.from_numpy(td.faces),
        }
        if td.vertex_normals is not None:
            result["vertex_normals"] = torch.from_numpy(td.vertex_normals)
        if td.face_normals is not None:
            result["face_normals"] = torch.from_numpy(td.face_normals)
        if td.vertex_colors is not None:
            result["vertex_colors"] = torch.from_numpy(td.vertex_colors)
        if td.uv is not None:
            result["uv"] = torch.from_numpy(td.uv)
        if td.material is not None:
            result["material"] = td.material
        if td.vertex_attributes:
            import numpy as np
            result["vertex_attributes"] = {
                k: torch.from_numpy(np.asarray(v)) if hasattr(v, "__array__") else v
                for k, v in td.vertex_attributes.items()
            }
        if td.face_attributes:
            import numpy as np
            result["face_attributes"] = {
                k: torch.from_numpy(np.asarray(v)) if hasattr(v, "__array__") else v
                for k, v in td.face_attributes.items()
            }
        if td.metadata:
            result["metadata"] = td.metadata

        return result

    def deserialize_trimesh(data: Any) -> Any:
        import os
        from comfy_api.latest._util.trimesh_types import TrimeshData

        def _to_np(v):
            return v.numpy() if hasattr(v, "numpy") else v

        va = None
        if "vertex_attributes" in data:
            va = {k: _to_np(v) for k, v in data["vertex_attributes"].items()}

        fa = None
        if "face_attributes" in data:
            fa = {k: _to_np(v) for k, v in data["face_attributes"].items()}

        td = TrimeshData(
            vertices=data["vertices"].numpy(),
            faces=data["faces"].numpy(),
            vertex_normals=data["vertex_normals"].numpy() if "vertex_normals" in data else None,
            face_normals=data["face_normals"].numpy() if "face_normals" in data else None,
            vertex_colors=data["vertex_colors"].numpy() if "vertex_colors" in data else None,
            uv=data["uv"].numpy() if "uv" in data else None,
            material=data.get("material"),
            vertex_attributes=va,
            face_attributes=fa,
            metadata=data.get("metadata"),
        )

        # Child process has trimesh installed — return real Trimesh object
        if os.environ.get("PYISOLATE_CHILD") == "1":
            return td.to_trimesh()
        return td

    registry.register("TRIMESH", serialize_trimesh, deserialize_trimesh, data_type=True)
    registry.register("Trimesh", serialize_trimesh, deserialize_trimesh, data_type=True)
    registry.register("TrimeshData", serialize_trimesh, deserialize_trimesh, data_type=True)
    if not os.environ.get("PYISOLATE_CHILD"):
        print("][ Serializer registered: TRIMESH")

    # -- SKELETON (GeometryPack skeleton dict) ----------------------------------
    # Custom dict type invented by GeometryPack for skeleton extraction.
    # Origin: ComfyUI-GeometryPack (PozzettiAndrea)
    # Used by: GeomPackExtractSkeleton, GeomPackMeshFromSkeleton

    def serialize_skeleton(obj: Any) -> Dict[str, Any]:
        _announce("SKELETON", "GeometryPack SKELETON (by PozzettiAndrea) serializer 1.0 (tensors, dict) for ComfyUI-GeometryPack")
        import numpy as np
        import torch

        result: Dict[str, Any] = {
            "__type__": "SKELETON",
            "vertices": torch.from_numpy(np.asarray(obj["vertices"], dtype=np.float64)),
            "edges": torch.from_numpy(np.asarray(obj["edges"], dtype=np.int64)),
            "scale": obj["scale"],
            "center": obj["center"],
            "normalized": obj["normalized"],
        }
        return result

    def deserialize_skeleton(data: Any) -> Any:
        return {
            "vertices": data["vertices"].numpy(),
            "edges": data["edges"].numpy(),
            "scale": data["scale"],
            "center": data["center"],
            "normalized": data["normalized"],
        }

    registry.register("SKELETON", serialize_skeleton, deserialize_skeleton, data_type=True)
    if not os.environ.get("PYISOLATE_CHILD"):
        print("][ Serializer registered: SKELETON")
