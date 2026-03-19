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

    # -- TRIMESH (trimesh.Trimesh by Michael Dawson-Haggerty) -------------------
    # In-memory triangular mesh object from the trimesh library.
    # Origin: trimesh library (https://github.com/mikedh/trimesh), MIT license
    # Used by: ComfyUI-GeometryPack (62 nodes)

    def serialize_trimesh(obj: Any) -> Dict[str, Any]:
        _announce("TRIMESH", "trimesh.Trimesh (by Michael Dawson-Haggerty) serializer 1.0 (tensors, dict) for ComfyUI-GeometryPack")
        import numpy as np
        import torch

        result: Dict[str, Any] = {
            "__type__": "TRIMESH",
            "vertices": torch.from_numpy(np.asarray(obj.vertices, dtype=np.float64)),
            "faces": torch.from_numpy(np.asarray(obj.faces, dtype=np.int64)),
        }

        # Vertex normals (computed lazily by trimesh — only serialize if cached)
        if obj._cache.cache.get("vertex_normals") is not None:
            result["vertex_normals"] = torch.from_numpy(np.asarray(obj.vertex_normals, dtype=np.float64))

        # Face normals
        if obj._cache.cache.get("face_normals") is not None:
            result["face_normals"] = torch.from_numpy(np.asarray(obj.face_normals, dtype=np.float64))

        # Vertex colors (RGBA uint8)
        try:
            vc = obj.visual.vertex_colors
            if vc is not None and len(vc) > 0:
                result["vertex_colors"] = torch.from_numpy(np.asarray(vc, dtype=np.uint8))
        except Exception:
            pass

        # Vertex attributes (dict of ndarray — scalar fields per vertex)
        if hasattr(obj, "vertex_attributes") and obj.vertex_attributes:
            va: Dict[str, Any] = {}
            for k, v in obj.vertex_attributes.items():
                if isinstance(v, np.ndarray):
                    va[k] = torch.from_numpy(v.copy())
                else:
                    va[k] = v
            result["vertex_attributes"] = va

        # Face attributes (dict of ndarray — scalar fields per face)
        if hasattr(obj, "face_attributes") and obj.face_attributes:
            fa: Dict[str, Any] = {}
            for k, v in obj.face_attributes.items():
                if isinstance(v, np.ndarray):
                    fa[k] = torch.from_numpy(v.copy())
                else:
                    fa[k] = v
            result["face_attributes"] = fa

        # Metadata (dict of JSON-compatible primitives)
        if obj.metadata:
            result["metadata"] = obj.metadata

        return result

    def deserialize_trimesh(data: Any) -> Any:
        import trimesh
        import numpy as np

        vertices = data["vertices"].numpy()
        faces = data["faces"].numpy()

        kwargs: Dict[str, Any] = {}

        if "vertex_normals" in data:
            kwargs["vertex_normals"] = data["vertex_normals"].numpy()
        if "face_normals" in data:
            kwargs["face_normals"] = data["face_normals"].numpy()
        if "metadata" in data:
            kwargs["metadata"] = data["metadata"]

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False, **kwargs)

        if "vertex_colors" in data:
            mesh.visual.vertex_colors = data["vertex_colors"].numpy()

        if "vertex_attributes" in data:
            for k, v in data["vertex_attributes"].items():
                mesh.vertex_attributes[k] = v.numpy() if hasattr(v, "numpy") else v

        if "face_attributes" in data:
            for k, v in data["face_attributes"].items():
                mesh.face_attributes[k] = v.numpy() if hasattr(v, "numpy") else v

        return mesh

    registry.register("TRIMESH", serialize_trimesh, deserialize_trimesh, data_type=True)
    registry.register("Trimesh", serialize_trimesh, deserialize_trimesh, data_type=True)
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
    print("][ Serializer registered: SKELETON")
