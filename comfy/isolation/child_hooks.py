"""Child process initialization for PyIsolate."""
import logging
import os
import sys

logger = logging.getLogger(__name__)


def is_child_process() -> bool:
    return os.environ.get("PYISOLATE_CHILD") == "1"


def initialize_child_process() -> None:
    _setup_prompt_server_proxy()
    _setup_logging()
    sys.exit(0)


def _setup_prompt_server_proxy() -> None:
    if os.environ.get("PYISOLATE_DEV") != "1":
        return
    try:
        import server
        from .development.proxies.prompt_server_proxy import PromptServerProxy
        server.PromptServer.instance = PromptServerProxy()
    except Exception:
        pass


def _setup_logging() -> None:
    logging.getLogger().setLevel(logging.WARNING)
