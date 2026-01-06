import comfy.model_management as mm
from pyisolate import ProxiedSingleton

class ModelManagementProxy(ProxiedSingleton):
    """
    Dynamic proxy for comfy.model_management.
    Uses __getattr__ to forward all calls to the underlying module,
    reducing maintenance burden.
    """

    # Explicitly expose Enums/Classes as properties
    @property
    def VRAMState(self):
        return mm.VRAMState

    @property
    def CPUState(self):
        return mm.CPUState

    @property
    def OOM_EXCEPTION(self):
        return mm.OOM_EXCEPTION

    def __getattr__(self, name):
        """Forward all other attribute access to the module."""
        return getattr(mm, name)
