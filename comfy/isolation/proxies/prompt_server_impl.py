# pylint: disable=import-outside-toplevel,logging-fstring-interpolation,redefined-outer-name,reimported,super-init-not-called
"""Stateless RPC Implementation for PromptServer.

Replaces the legacy PromptServerProxy (Singleton) with a clean Service/Stub architecture.
- Host: PromptServerService (RPC Handler)
- Child: PromptServerStub (Interface Implementation)
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Callable

import logging

# IMPORTS
from pyisolate import ProxiedSingleton
from .base import call_singleton_rpc

logger = logging.getLogger(__name__)
LOG_PREFIX = "[Isolation:C<->H]"

# ...

# =============================================================================
# CHILD SIDE: PromptServerStub
# =============================================================================


class PromptServerStub:
    """Stateless Stub for PromptServer."""

    # Masquerade as the real server module
    __module__ = "server"

    _instance: Optional["PromptServerStub"] = None
    _rpc: Optional[Any] = None  # This will be the Caller object
    _source_file: Optional[str] = None

    def __init__(self):
        self.routes = RouteStub(self)

    @classmethod
    def set_rpc(cls, rpc: Any) -> None:
        """Inject RPC client (called by adapter.py or manually)."""
        # Bind a caller to the host-side PromptServerService, registered under its
        # class name. The Service is defined in this module, importable on either side.
        target_id = "PromptServerService"
        cls._rpc = rpc.create_caller(PromptServerService, target_id)

    @classmethod
    def clear_rpc(cls) -> None:
        cls._rpc = None

    @property
    def instance(self) -> "PromptServerStub":
        return self

    # ... Compatibility ...
    @classmethod
    def _get_source_file(cls) -> str:
        if cls._source_file is None:
            import folder_paths

            cls._source_file = os.path.join(folder_paths.base_path, "server.py")
        return cls._source_file

    @property
    def __file__(self) -> str:
        return self._get_source_file()

    # --- Properties ---
    @property
    def client_id(self) -> Optional[str]:
        return "isolated_client"

    @property
    def supports(self) -> set:
        return {"custom_nodes_from_web"}

    @property
    def app(self):
        return _AppStub(self)

    @property
    def prompt_queue(self):
        raise RuntimeError(
            "PromptServer.prompt_queue is not accessible in isolated nodes."
        )

    # --- UI Communication (RPC Delegates) ---
    async def send_sync(
        self, event: str, data: Dict[str, Any], sid: Optional[str] = None
    ) -> None:
        if self._rpc:
            await self._rpc.ui_send_sync(event, data, sid)

    async def send(
        self, event: str, data: Dict[str, Any], sid: Optional[str] = None
    ) -> None:
        if self._rpc:
            await self._rpc.ui_send(event, data, sid)

    def send_progress_text(self, text: str, node_id: str, sid=None) -> None:
        if self._rpc:
            # Node code calls this synchronously; schedule the async host send on the
            # running loop when there is one, else relay it synchronously.
            import asyncio

            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._rpc.ui_send_progress_text(text, node_id, sid))
            except RuntimeError:
                call_singleton_rpc(self._rpc, "ui_send_progress_text", text, node_id, sid)

    # --- Route Registration Logic ---
    _pending_child_routes: list = []

    def register_route(self, method: str, path: str, handler: Callable):
        """Buffer route registration. Routes are flushed via flush_child_routes()."""
        PromptServerStub._pending_child_routes.append((method, path, handler))
        logger.info("%s Buffered isolated route %s %s", LOG_PREFIX, method, path)

    @classmethod
    async def flush_child_routes(cls):
        """Send all buffered route registrations to host via RPC. Call from on_module_loaded."""
        if not cls._rpc:
            return 0
        flushed = 0
        for method, path, handler in cls._pending_child_routes:
            try:
                await cls._rpc.register_route_rpc(method, path, handler)
                flushed += 1
            except Exception as e:
                logger.error("%s Child route flush failed %s %s: %s", LOG_PREFIX, method, path, e)
        cls._pending_child_routes = []
        return flushed


class RouteStub:
    """Simulates aiohttp.web.RouteTableDef."""

    def __init__(self, stub: PromptServerStub):
        self._stub = stub

    def get(self, path: str):
        def decorator(handler):
            self._stub.register_route("GET", path, handler)
            return handler

        return decorator

    def post(self, path: str):
        def decorator(handler):
            self._stub.register_route("POST", path, handler)
            return handler

        return decorator

    def patch(self, path: str):
        def decorator(handler):
            self._stub.register_route("PATCH", path, handler)
            return handler

        return decorator

    def put(self, path: str):
        def decorator(handler):
            self._stub.register_route("PUT", path, handler)
            return handler

        return decorator

    def delete(self, path: str):
        def decorator(handler):
            self._stub.register_route("DELETE", path, handler)
            return handler

        return decorator


# =============================================================================
# HOST SIDE: PromptServerService
# =============================================================================


class PromptServerService(ProxiedSingleton):
    """Host-side RPC Service for PromptServer."""

    def __init__(self):
        pass

    @property
    def server(self):
        from server import PromptServer

        return PromptServer.instance

    async def ui_send_sync(
        self, event: str, data: Dict[str, Any], sid: Optional[str] = None
    ):
        await self.server.send_sync(event, data, sid)

    async def ui_send(
        self, event: str, data: Dict[str, Any], sid: Optional[str] = None
    ):
        await self.server.send(event, data, sid)

    async def ui_send_progress_text(self, text: str, node_id: str, sid=None):
        # Made async to be awaitable by RPC layer
        self.server.send_progress_text(text, node_id, sid)

    async def register_route_rpc(self, method: str, path: str, child_handler_proxy):
        """RPC Target: Register a route that forwards to the Child."""
        from aiohttp import web
        logger.info("%s Registering isolated route %s %s", LOG_PREFIX, method, path)

        async def route_wrapper(request: web.Request) -> web.Response:
            # 1. Capture request data
            req_data = {
                "method": request.method,
                "path": request.path,
                "query": dict(request.query),
            }
            if request.can_read_body:
                req_data["text"] = await request.text()

            try:
                # 2. Call Child Handler via RPC (child_handler_proxy is async callable)
                result = await child_handler_proxy(req_data)

                # 3. Serialize Response
                return self._serialize_response(result)
            except Exception as e:
                logger.error(f"{LOG_PREFIX} Isolated Route Error: {e}")
                return web.Response(status=500, text=str(e))

        self.server.app.router.add_route(method, path, route_wrapper)
        logger.info("%s Registered isolated route %s %s", LOG_PREFIX, method, path)

    def _serialize_response(self, result: Any) -> Any:
        """Helper to convert Child result -> web.Response"""
        from aiohttp import web
        if isinstance(result, web.Response):
            return result
        # Handle dict (json)
        if isinstance(result, dict):
            return web.json_response(result)
        # Handle string
        if isinstance(result, str):
            return web.Response(text=result)
        # Fallback
        return web.Response(text=str(result))


class _RouterStub:
    """Captures router.add_route and router.add_static calls in isolation child."""

    def __init__(self, stub):
        self._stub = stub

    def add_route(self, method, path, handler, **kwargs):
        self._stub.register_route(method, path, handler)

    def add_static(self, prefix, path, **kwargs):
        # Static file serving not supported in isolation — silently skip
        pass


class _AppStub:
    """Captures PromptServer.app access patterns in isolation child."""

    def __init__(self, stub):
        self.router = _RouterStub(stub)
        self.frozen = False

    def add_routes(self, routes):
        # aiohttp route table — iterate and register each
        for route in routes:
            if hasattr(route, "method") and hasattr(route, "handler"):
                self.router.add_route(route.method, route.path, route.handler)
            # StaticDef and other non-method routes — silently skip
