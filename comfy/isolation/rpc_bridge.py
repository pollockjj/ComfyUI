import asyncio
import logging
import threading

logger = logging.getLogger(__name__)


class RpcBridge:
    """Minimal helper to run coroutines synchronously inside isolated processes.

    If an event loop is already running, the coroutine is executed on a fresh
    thread with its own loop to avoid nested run_until_complete errors.
    """

    def run_sync(self, maybe_coro):
        if not asyncio.iscoroutine(maybe_coro):
            return maybe_coro

        try:
            loop = asyncio.get_running_loop()
            if loop.is_running() and not loop.is_closed():
                result_container = {}
                exc_container = {}

                def _runner():
                    try:
                        # Reuse running loop via call_soon_threadsafe instead of creating a throwaway loop
                        fut = asyncio.run_coroutine_threadsafe(maybe_coro, loop)
                        result_container["value"] = fut.result()
                    except Exception as exc:  # pragma: no cover
                        exc_container["error"] = exc

                t = threading.Thread(target=_runner, daemon=True)
                t.start()
                t.join()

                if "error" in exc_container:
                    raise exc_container["error"]
                return result_container.get("value")
        except RuntimeError:
            loop = None

        # No running loop: create one and keep it open for the duration of the call
        new_loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(new_loop)
            return new_loop.run_until_complete(maybe_coro)
        finally:
            try:
                new_loop.close()
            except Exception:
                pass
