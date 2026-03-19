from __future__ import annotations

import numpy as np


class TrimeshData:
    """Triangular mesh payload for cross-process transfer.

    Lightweight carrier for mesh geometry that does not depend on the
    ``trimesh`` library.  Serializers create this on the host side;
    isolated child processes convert to/from ``trimesh.Trimesh`` as needed.
    """

    def __init__(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        vertex_normals: np.ndarray | None = None,
        face_normals: np.ndarray | None = None,
        vertex_colors: np.ndarray | None = None,
        vertex_attributes: dict | None = None,
        face_attributes: dict | None = None,
        metadata: dict | None = None,
    ) -> None:
        self.vertices = np.ascontiguousarray(vertices, dtype=np.float64)
        self.faces = np.ascontiguousarray(faces, dtype=np.int64)
        self.vertex_normals = (
            np.ascontiguousarray(vertex_normals, dtype=np.float64)
            if vertex_normals is not None
            else None
        )
        self.face_normals = (
            np.ascontiguousarray(face_normals, dtype=np.float64)
            if face_normals is not None
            else None
        )
        self.vertex_colors = (
            np.ascontiguousarray(vertex_colors, dtype=np.uint8)
            if vertex_colors is not None
            else None
        )
        self.vertex_attributes = vertex_attributes or {}
        self.face_attributes = face_attributes or {}
        self.metadata = self._detensorize_dict(metadata) if metadata else {}

    @staticmethod
    def _detensorize_dict(d):
        """Recursively convert any tensors in a dict back to numpy arrays."""
        if not isinstance(d, dict):
            return d
        result = {}
        for k, v in d.items():
            if hasattr(v, "numpy"):
                result[k] = v.cpu().numpy() if hasattr(v, "cpu") else v.numpy()
            elif isinstance(v, dict):
                result[k] = TrimeshData._detensorize_dict(v)
            elif isinstance(v, list):
                result[k] = [
                    item.cpu().numpy() if hasattr(item, "numpy") and hasattr(item, "cpu")
                    else item.numpy() if hasattr(item, "numpy")
                    else item
                    for item in v
                ]
            else:
                result[k] = v
        return result

    @staticmethod
    def _to_numpy(arr, dtype):
        if arr is None:
            return None
        if hasattr(arr, "numpy"):
            arr = arr.cpu().numpy() if hasattr(arr, "cpu") else arr.numpy()
        return np.ascontiguousarray(arr, dtype=dtype)

    @property
    def num_vertices(self) -> int:
        return self.vertices.shape[0]

    @property
    def num_faces(self) -> int:
        return self.faces.shape[0]

    def to_trimesh(self):
        """Convert to trimesh.Trimesh (requires trimesh in the environment)."""
        import trimesh

        kwargs = {}
        if self.vertex_normals is not None:
            kwargs["vertex_normals"] = self.vertex_normals
        if self.face_normals is not None:
            kwargs["face_normals"] = self.face_normals
        if self.metadata:
            kwargs["metadata"] = self.metadata

        mesh = trimesh.Trimesh(
            vertices=self.vertices, faces=self.faces, process=False, **kwargs
        )

        if self.vertex_colors is not None:
            mesh.visual.vertex_colors = self.vertex_colors

        for k, v in self.vertex_attributes.items():
            mesh.vertex_attributes[k] = v

        for k, v in self.face_attributes.items():
            mesh.face_attributes[k] = v

        return mesh

    @classmethod
    def from_trimesh(cls, mesh) -> TrimeshData:
        """Create from a trimesh.Trimesh object."""
        import numpy as np

        vertex_normals = None
        if mesh._cache.cache.get("vertex_normals") is not None:
            vertex_normals = np.asarray(mesh.vertex_normals)

        face_normals = None
        if mesh._cache.cache.get("face_normals") is not None:
            face_normals = np.asarray(mesh.face_normals)

        vertex_colors = None
        try:
            vc = mesh.visual.vertex_colors
            if vc is not None and len(vc) > 0:
                vertex_colors = np.asarray(vc, dtype=np.uint8)
        except Exception:
            pass

        va = {}
        if hasattr(mesh, "vertex_attributes") and mesh.vertex_attributes:
            for k, v in mesh.vertex_attributes.items():
                va[k] = np.asarray(v) if hasattr(v, "__array__") else v

        fa = {}
        if hasattr(mesh, "face_attributes") and mesh.face_attributes:
            for k, v in mesh.face_attributes.items():
                fa[k] = np.asarray(v) if hasattr(v, "__array__") else v

        return cls(
            vertices=np.asarray(mesh.vertices),
            faces=np.asarray(mesh.faces),
            vertex_normals=vertex_normals,
            face_normals=face_normals,
            vertex_colors=vertex_colors,
            vertex_attributes=va if va else None,
            face_attributes=fa if fa else None,
            metadata=mesh.metadata if mesh.metadata else None,
        )
