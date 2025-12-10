"""Host process initialization for PyIsolate."""
import logging
import os

logger = logging.getLogger(__name__)


def initialize_host_process() -> None:
    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)
    root.addHandler(logging.NullHandler())

    from .proxies.folder_paths_proxy import FolderPathsProxy
    from .proxies.model_management_proxy import ModelManagementProxy
    from .proxies.progress_proxy import ProgressProxy
    from .proxies.prompt_server_proxy import PromptServerProxy
    from .vae_proxy import VAERegistry
    from .model_sampling_proxy import ModelSamplingRegistry
    from .development.model_patcher_proxy import ModelPatcherRegistry

    FolderPathsProxy()
    ModelManagementProxy()
    ProgressProxy()
    PromptServerProxy()
    VAERegistry()
    ModelSamplingRegistry()
    ModelPatcherRegistry()

    if os.environ.get("PYISOLATE_DEV") == "1":
        from .development.proxies.nodes_proxy import NodesProxy
        from .development.proxies.utils_proxy import UtilsProxy
        from .clip_proxy import CLIPRegistry
        from .model_sampling_proxy import ModelSamplingRegistry as DevModelSamplingRegistry

        NodesProxy()
        UtilsProxy()
        CLIPRegistry()
        DevModelSamplingRegistry()
