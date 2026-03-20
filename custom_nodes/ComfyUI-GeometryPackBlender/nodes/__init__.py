from .blender.remeshing import NODE_CLASS_MAPPINGS as remesh_mappings
from .blender.remeshing import NODE_DISPLAY_NAME_MAPPINGS as remesh_display
from .io import NODE_CLASS_MAPPINGS as io_mappings
from .io import NODE_DISPLAY_NAME_MAPPINGS as io_display

NODE_CLASS_MAPPINGS = {
    **remesh_mappings,
    **io_mappings,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **remesh_display,
    **io_display,
}
