"""Serializers for custom node data types.

These serializers exist because of specific custom node conversions and are
not required by core ComfyUI.  Types defined in comfy_api (File3D, VIDEO,
etc.) belong in adapter.py.  Only types invented by custom node packs that
don't exist in comfy_api belong here.

New custom node conversions that introduce types not covered here should
add their serializers to this file.

IMPORTANT: Data serializers MUST NOT depend on torch. Use numpy .tolist()
and np.array() for serialization. The original comfy-env worker environments
do not have torch installed.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict

from pyisolate.interfaces import SerializerRegistryProtocol  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)

_IMPORT_TORCH = os.environ.get("PYISOLATE_IMPORT_TORCH", "1") == "1"

_announced: set[str] = set()


def _announce(name: str, desc: str) -> None:
    if name not in _announced:
        _announced.add(name)
        logger.info("][ Serializer: %s — %s", name, desc)


def _get_trimesh_data():
    if _IMPORT_TORCH:
        from comfy_api.latest._util.trimesh_types import TrimeshData
    else:
        from comfy_api_sealed_worker.trimesh_types import TrimeshData
    return TrimeshData


def _get_ply():
    if _IMPORT_TORCH:
        from comfy_api.latest._util.ply_types import PLY
    else:
        from comfy_api_sealed_worker.ply_types import PLY
    return PLY


def _get_npz():
    if _IMPORT_TORCH:
        from comfy_api.latest._util.npz_types import NPZ
    else:
        from comfy_api_sealed_worker.npz_types import NPZ
    return NPZ


def register_custom_node_serializers(registry: SerializerRegistryProtocol) -> None:
    """Register all custom-node-originated serializers."""

    # -- ndarray (numpy) -------------------------------------------------------
    # Torch-free ndarray serializer for sealed workers.
    # The adapter.py version uses torch.from_numpy(); this one uses .tolist().
    if not _IMPORT_TORCH:
        def serialize_ndarray(obj: Any) -> Any:
            return obj.tolist()

        registry.register("ndarray", serialize_ndarray, None)

    # -- PLY (comfy_api.latest._util.ply_types) --------------------------------
    # PLY point cloud container created for DA3 isolation conversion.
    # Origin: pollockjj for ComfyUI-DepthAnythingV3 isolation (commit 99d90b29)
    # Used by: ComfyUI-DepthAnythingV3, ComfyUI-GeometryPack

    def serialize_ply(obj: Any) -> Dict[str, Any]:
        _announce("PLY", "PLY (by pollockjj for DA3 isolation) serializer 1.0 (base64/lists) for ComfyUI-DepthAnythingV3, ComfyUI-GeometryPack")
        import base64
        import numpy as np
        if obj.raw_data is not None:
            return {
                "__type__": "PLY",
                "raw_data": base64.b64encode(obj.raw_data).decode("ascii"),
            }
        result: Dict[str, Any] = {"__type__": "PLY", "points": np.asarray(obj.points).tolist()}
        if obj.colors is not None:
            result["colors"] = np.asarray(obj.colors).tolist()
        if obj.confidence is not None:
            result["confidence"] = np.asarray(obj.confidence).tolist()
        if obj.view_id is not None:
            result["view_id"] = np.asarray(obj.view_id).tolist()
        return result

    def deserialize_ply(data: Any) -> Any:
        import base64
        import numpy as np
        PLY = _get_ply()
        if "raw_data" in data:
            return PLY(raw_data=base64.b64decode(data["raw_data"]))
        return PLY(
            points=np.array(data["points"]),
            colors=np.array(data["colors"]) if data.get("colors") is not None else None,
            confidence=np.array(data["confidence"]) if data.get("confidence") is not None else None,
            view_id=np.array(data["view_id"]) if data.get("view_id") is not None else None,
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
        NPZ = _get_npz()
        return NPZ(frames=[base64.b64decode(f) for f in data["frames"]])

    registry.register("NPZ", serialize_npz, deserialize_npz, data_type=True)

    # -- TRIMESH (trimesh.Trimesh by Michael Dawson-Haggerty) -------------------
    # In-memory triangular mesh object from the trimesh library.
    # Origin: trimesh library (https://github.com/mikedh/trimesh), MIT license
    # Used by: ComfyUI-GeometryPack (62 nodes)

    def serialize_trimesh(obj: Any) -> Dict[str, Any]:
        _announce("TRIMESH", "trimesh.Trimesh (by Michael Dawson-Haggerty) serializer 1.0 (lists, dict) for ComfyUI-GeometryPack")
        import numpy as np
        TrimeshData = _get_trimesh_data()

        # Handle both trimesh.Trimesh and TrimeshData (host round-trip)
        if isinstance(obj, TrimeshData):
            td = obj
        else:
            td = TrimeshData.from_trimesh(obj)

        result: Dict[str, Any] = {
            "__type__": "TRIMESH",
            "vertices": td.vertices.tolist(),
            "faces": td.faces.tolist(),
        }
        if td.vertex_normals is not None:
            result["vertex_normals"] = td.vertex_normals.tolist()
        if td.face_normals is not None:
            result["face_normals"] = td.face_normals.tolist()
        if td.vertex_colors is not None:
            result["vertex_colors"] = td.vertex_colors.tolist()
        if td.uv is not None:
            result["uv"] = td.uv.tolist()
        if td.material is not None:
            result["material"] = td.material
        if td.vertex_attributes:
            result["vertex_attributes"] = {
                k: np.asarray(v).tolist() if hasattr(v, "__array__") else v
                for k, v in td.vertex_attributes.items()
            }
        if td.face_attributes:
            result["face_attributes"] = {
                k: np.asarray(v).tolist() if hasattr(v, "__array__") else v
                for k, v in td.face_attributes.items()
            }
        if td.metadata:
            result["metadata"] = td.metadata

        return result

    def deserialize_trimesh(data: Any) -> Any:
        is_child = os.environ.get("PYISOLATE_CHILD") == "1"
        logger.warning("][ TRIMESH_DESERIALIZE_DIAG: entry, child=%s, type(data)=%s, keys=%s", is_child, type(data).__name__, list(data.keys()) if isinstance(data, dict) else "N/A")
        import numpy as np
        try:
            TrimeshData = _get_trimesh_data()
        except Exception as e:
            logger.error("][ TRIMESH_DESERIALIZE_DIAG: FAILED to get TrimeshData: %s", e)
            raise

        def _to_np(v):
            if isinstance(v, list):
                return np.array(v)
            return v.numpy() if hasattr(v, "numpy") else v

        va = None
        if "vertex_attributes" in data:
            va = {k: _to_np(v) for k, v in data["vertex_attributes"].items()}

        fa = None
        if "face_attributes" in data:
            fa = {k: _to_np(v) for k, v in data["face_attributes"].items()}

        td = TrimeshData(
            vertices=np.array(data["vertices"]),
            faces=np.array(data["faces"]),
            vertex_normals=np.array(data["vertex_normals"]) if "vertex_normals" in data else None,
            face_normals=np.array(data["face_normals"]) if "face_normals" in data else None,
            vertex_colors=np.array(data["vertex_colors"], dtype=np.uint8) if "vertex_colors" in data else None,
            uv=np.array(data["uv"]) if "uv" in data else None,
            material=data.get("material"),
            vertex_attributes=va,
            face_attributes=fa,
            metadata=data.get("metadata"),
        )

        # Child process has trimesh installed — return real Trimesh object
        if os.environ.get("PYISOLATE_CHILD") == "1":
            result = td.to_trimesh()
            logger.warning("][ TRIMESH_DESERIALIZE_DIAG: complete, child=True, returning type=%s", type(result).__name__)
            return result
        logger.warning("][ TRIMESH_DESERIALIZE_DIAG: complete, child=False, returning TrimeshData")
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
        _announce("SKELETON", "GeometryPack SKELETON (by PozzettiAndrea) serializer 1.0 (lists, dict) for ComfyUI-GeometryPack")
        import numpy as np

        result: Dict[str, Any] = {
            "__type__": "SKELETON",
            "vertices": np.asarray(obj["vertices"], dtype=np.float64).tolist(),
            "edges": np.asarray(obj["edges"], dtype=np.int64).tolist(),
            "scale": obj["scale"],
            "center": obj["center"],
            "normalized": obj["normalized"],
        }
        return result

    def deserialize_skeleton(data: Any) -> Any:
        import numpy as np
        return {
            "vertices": np.array(data["vertices"], dtype=np.float64),
            "edges": np.array(data["edges"], dtype=np.int64),
            "scale": data["scale"],
            "center": data["center"],
            "normalized": data["normalized"],
        }

    registry.register("SKELETON", serialize_skeleton, deserialize_skeleton, data_type=True)
    if not os.environ.get("PYISOLATE_CHILD"):
        print("][ Serializer registered: SKELETON")
