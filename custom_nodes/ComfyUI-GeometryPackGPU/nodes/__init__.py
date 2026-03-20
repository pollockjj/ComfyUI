from .gpu import NODE_CLASS_MAPPINGS as gpu_mappings
from .gpu import NODE_DISPLAY_NAME_MAPPINGS as gpu_display
from .io import NODE_CLASS_MAPPINGS as io_mappings
from .io import NODE_DISPLAY_NAME_MAPPINGS as io_display

NODE_CLASS_MAPPINGS = {
    **gpu_mappings,
    **io_mappings,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **gpu_display,
    **io_display,
}
