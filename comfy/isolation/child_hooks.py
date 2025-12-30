"""Child process initialization for PyIsolate."""
import logging
import os
import sys

logger = logging.getLogger(__name__)


def is_child_process() -> bool:
    return os.environ.get("PYISOLATE_CHILD") == "1"


def initialize_child_process() -> None:
    # Manual RPC Injection Fallback
    try:
        from pyisolate._internal.shared import get_child_rpc_instance
        rpc = get_child_rpc_instance()
        if rpc:
             _setup_prompt_server_stub(rpc)
             _setup_utils_proxy(rpc)
        else:
             logger.warning("Could not get child RPC instance for manual injection")
             _setup_prompt_server_stub()
             _setup_utils_proxy()
    except Exception as e:
        logger.error(f"Manual RPC Injection failed: {e}")
        _setup_prompt_server_stub()
        _setup_utils_proxy()
    
    _setup_logging()

def _setup_prompt_server_stub(rpc=None) -> None:
    try:
        from .proxies.prompt_server_impl import PromptServerStub
        import sys
        import types

        # Mock 'server' module to avoid side-effects
        if "server" not in sys.modules:
            mock_server = types.ModuleType("server")
            sys.modules["server"] = mock_server
        
        server = sys.modules["server"]
        
        if not hasattr(server, "PromptServer"):
            class MockPromptServer: pass
            server.PromptServer = MockPromptServer
        
        stub = PromptServerStub()
        
        if rpc:
             PromptServerStub.set_rpc(rpc)
             if hasattr(stub, "set_rpc"):
                 stub.set_rpc(rpc)

        server.PromptServer.instance = stub
        logger.info(f"PromptServerStub manually setup with RPC={rpc is not None}")
        
    except Exception as e:
        logger.error(f"Failed to setup PromptServerStub: {e}")

def _setup_utils_proxy(rpc=None) -> None:
    try:
        from .proxies.utils_proxy import UtilsProxy
        import comfy.utils
        import asyncio
        
        from .proxies.utils_proxy import UtilsProxy
        import comfy.utils
        import asyncio
        
        # UtilsProxy instantiation removed to allow pyisolate injection
        # logic to handle it (SingletonMetaclass collision avoidance).
        # We also don't need manual set_rpc as injection handles it.

        def sync_hook_wrapper(value: int, total: int, preview: None = None, node_id: None = None) -> None:
            resolved = "arg"
            if node_id is None:
                try:
                    from comfy_execution.utils import get_executing_context
                    ctx = get_executing_context()
                    if ctx:
                        node_id = ctx.node_id
                        resolved = f"ctx({node_id})"
                    else:
                        resolved = "ctx(None)"
                except Exception as e:
                    resolved = f"err({e})"

            # Direct RPC Injection to bypass blocked Event Loop
            # The standard proxy method is async and requires the loop to run to schedule the send.
            # But the loop is blocked by KSampler (us!). So we must manually inject into the
            # thread-safe outbox which feeds the independent _send_thread.
            if rpc:
                try:
                    # We need to construct a pseudo-request. We don't care about the response.
                    # importing here to avoid circular deps if possible, or just using dict
                    
                    # We can't easily get the strict types, but it's just a dict at runtime.
                    # Structure matches pyisolate._internal.shared.RPCPendingRequest
                    
                    # object_id="UtilsProxy" because that's what UtilsProxy is registered as.
                    # method="progress_bar_hook"
                    
                    # We provide a dummy future/loop for the response handler, preventing crashes.
                    # The response will arrive and callback will be scheduled on the blocked loop,
                    # which is fine (we ignore the result anyway/it happens later).
                    
                    loop = asyncio.get_event_loop()
                    
                    # NOTE: We are NOT awaiting this future.
                    # We are forcing the "Call" packet into the outbox.
                    
                    rpc.outbox.put({
                        "kind": "call",
                        "object_id": "UtilsProxy",
                        "parent_call_id": None, # We are root here usually
                        "calling_loop": loop,
                        "future": loop.create_future(), # Dummy future
                        "method": "progress_bar_hook",
                        "args": (value, total, preview, node_id),
                        "kwargs": {}
                    })

                except Exception as e:
                     logging.getLogger(__name__).error(f"Manual Inject Failed: {e}")
            else:
                 logging.getLogger(__name__).warning("No RPC instance available for progress update")

        comfy.utils.PROGRESS_BAR_HOOK = sync_hook_wrapper
        logger.info(f"Injected isolated progress bar hook: {comfy.utils.PROGRESS_BAR_HOOK}")

    except Exception as e:
        logger.error(f"Failed to setup UtilsProxy hook: {e}")


def _setup_logging() -> None:
    logging.getLogger().setLevel(logging.INFO)
