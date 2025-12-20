"""Experimental proxy implementations (gated behind IS_DEV)."""
import os

IS_DEV = os.environ.get("PYISOLATE_DEV") == "1"

if IS_DEV:
    from .prompt_server_proxy import PromptServerProxy
    from .nodes_proxy import NodesProxy
    from .utils_proxy import UtilsProxy
    
    __all__ = ["PromptServerProxy", "NodesProxy", "UtilsProxy"]
else:
    __all__ = []
