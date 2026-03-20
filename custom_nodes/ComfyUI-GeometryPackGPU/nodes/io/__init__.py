from .load_mesh_path_gpu import NODE_CLASS_MAPPINGS as LoadMeshPath_GPU_mappings
from .load_mesh_path_gpu import NODE_DISPLAY_NAME_MAPPINGS as LoadMeshPath_GPU_display
from .save_mesh_gpu import NODE_CLASS_MAPPINGS as SaveMesh_GPU_mappings
from .save_mesh_gpu import NODE_DISPLAY_NAME_MAPPINGS as SaveMesh_GPU_display

NODE_CLASS_MAPPINGS = {
    **LoadMeshPath_GPU_mappings,
    **SaveMesh_GPU_mappings,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **LoadMeshPath_GPU_display,
    **SaveMesh_GPU_display,
}
