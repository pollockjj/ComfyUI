# Temporary stub for GeometryPackGPU standalone validation.
# Remove when GeometryPack main (#43) comes online.

import os
import numpy as np
import trimesh

try:
    import folder_paths
    COMFYUI_INPUT_FOLDER = folder_paths.get_input_directory()
except (ImportError, AttributeError):
    COMFYUI_INPUT_FOLDER = None


class LoadMeshPathGPU:
    """Temporary load mesh stub for GeometryPackGPU standalone validation."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                }),
            },
        }

    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("mesh",)
    FUNCTION = "load_mesh"
    CATEGORY = "geompack_gpu/io"

    def load_mesh(self, file_path):
        file_path = file_path.strip()

        if not os.path.isabs(file_path) and COMFYUI_INPUT_FOLDER:
            candidate = os.path.join(COMFYUI_INPUT_FOLDER, "3d", file_path)
            if os.path.exists(candidate):
                file_path = candidate

        if not os.path.exists(file_path):
            raise ValueError(f"File not found: {file_path}")

        mesh = trimesh.load(file_path, force="mesh", process=False)
        print(f"[LoadMeshPath_GPU] Loaded: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        return (mesh,)


NODE_CLASS_MAPPINGS = {
    "LoadMeshPath_GPU": LoadMeshPathGPU,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadMeshPath_GPU": "Load Mesh Path (GPU Stub)",
}
