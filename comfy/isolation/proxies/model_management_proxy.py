from pyisolate import ProxiedSingleton


def _mm():
    import comfy.model_management
    return comfy.model_management


class ModelManagementProxy(ProxiedSingleton):
    """
    Dynamic proxy for comfy.model_management.
    Uses __getattr__ to forward all calls to the underlying module,
    reducing maintenance burden.
    """

    # Explicitly expose Enums/Classes as properties
    @property
    def VRAMState(self):
        return _mm().VRAMState

    @property
    def CPUState(self):
        return _mm().CPUState

    @property
    def OOM_EXCEPTION(self):
        return _mm().OOM_EXCEPTION

    def __getattr__(self, name):
        """Forward all other attribute access to the module."""
        return getattr(_mm(), name)
