# Temporary stub for GeometryPackGPU standalone validation.
# Remove when GeometryPack main (#43) comes online.

import os

try:
    import folder_paths
    COMFYUI_OUTPUT_FOLDER = folder_paths.get_output_directory()
except (ImportError, AttributeError):
    COMFYUI_OUTPUT_FOLDER = None


class SaveMeshGPU:
    """Temporary save mesh stub for GeometryPackGPU standalone validation."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "file_path": ("STRING", {
                    "default": "output",
                    "multiline": False,
                }),
                "format": (["obj", "ply", "stl", "off", "glb", "gltf"], {
                    "default": "obj",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_path",)
    FUNCTION = "save_mesh"
    CATEGORY = "geompack_gpu/io"
    OUTPUT_NODE = True

    def save_mesh(self, trimesh, file_path, format="obj"):
        if trimesh is None:
            raise ValueError("Cannot save mesh: received None")

        expected_ext = f".{format}"
        if not file_path.lower().endswith(expected_ext):
            base_path = os.path.splitext(file_path)[0]
            file_path = base_path + expected_ext

        full_path = file_path
        if not os.path.isabs(file_path) and COMFYUI_OUTPUT_FOLDER is not None:
            full_path = os.path.join(COMFYUI_OUTPUT_FOLDER, file_path)

        os.makedirs(os.path.dirname(full_path) if os.path.dirname(full_path) else ".", exist_ok=True)
        trimesh.export(full_path)
        print(f"[SaveMesh_GPU] Saved: {full_path} ({len(trimesh.vertices)} vertices, {len(trimesh.faces)} faces)")
        return (full_path,)


NODE_CLASS_MAPPINGS = {
    "SaveMesh_GPU": SaveMeshGPU,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveMesh_GPU": "Save Mesh (GPU Stub)",
}
