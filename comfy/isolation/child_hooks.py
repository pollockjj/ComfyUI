"""Child process initialization for PyIsolate."""
import logging
import os
import sys

logger = logging.getLogger(__name__)


def is_child_process() -> bool:
    return os.environ.get("PYISOLATE_CHILD") == "1"


def initialize_child_process() -> None:
    _setup_prompt_server_proxy()
    _setup_utils_proxy()
    _setup_logging()


def _setup_utils_proxy() -> None:
    try:
        from .proxies.utils_proxy import UtilsProxy
        import comfy.utils
        import asyncio
        
        # NOTE: We inject a wrapper that schedules the async RPC call on the event loop.
        # This is necessary because comfy.utils.ProgressBar expects a synchronous hook,
        # but our Proxy system (and UtilsProxy) operate via async RPC.

        def sync_hook_wrapper(value: int, total: int, preview: None = None, node_id: None = None) -> None:
            logging.getLogger(__name__).info(f"[TRACE:PBAR] C: Sync Wrapper Called. value={value}/{total}")
            # Try to get loop from PromptServerProxy which exposes the child loop
            from .proxies.prompt_server_proxy import PromptServerProxy
            try:
                loop = PromptServerProxy.instance.loop
            except Exception:
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    return

            if loop.is_running():
                # UtilsProxy() returns the singleton instance
                loop.create_task(UtilsProxy().progress_bar_hook(value, total, preview=preview, node_id=node_id))

        comfy.utils.PROGRESS_BAR_HOOK = sync_hook_wrapper
        logger.info(f"Injected isolated progress bar hook: {comfy.utils.PROGRESS_BAR_HOOK}")

    except Exception as e:
        logger.error(f"Failed to setup UtilsProxy hook: {e}")


def _setup_prompt_server_proxy() -> None:
    try:
        from .proxies.prompt_server_proxy import PromptServerProxy
        # Proxy will be registered when PyIsolate extension loads
    except Exception as e:
        logger.error(f"Failed to setup PromptServer proxy: {e}")


def _setup_logging() -> None:
    logging.getLogger().setLevel(logging.INFO)
