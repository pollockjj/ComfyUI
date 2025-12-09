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


def _setup_prompt_server_proxy() -> None:
    try:
        from .proxies.prompt_server_proxy import PromptServerProxy
        # Proxy will be registered when PyIsolate extension loads
    except Exception as e:
        logger.error(f"Failed to setup PromptServer proxy: {e}")


def _setup_logging() -> None:
    logging.getLogger().setLevel(logging.WARNING)
