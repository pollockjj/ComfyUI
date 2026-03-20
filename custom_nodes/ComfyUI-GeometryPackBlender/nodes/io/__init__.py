from .load_mesh_path_blender import NODE_CLASS_MAPPINGS as load_mappings
from .load_mesh_path_blender import NODE_DISPLAY_NAME_MAPPINGS as load_display
from .save_mesh_blender import NODE_CLASS_MAPPINGS as save_mappings
from .save_mesh_blender import NODE_DISPLAY_NAME_MAPPINGS as save_display

NODE_CLASS_MAPPINGS = {
    **load_mappings,
    **save_mappings,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **load_display,
    **save_display,
}
