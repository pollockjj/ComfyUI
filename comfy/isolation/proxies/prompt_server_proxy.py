"""ProxiedSingleton for PromptServer.

Rev 1.0 Implementation - UI communication and route registration.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
from typing import Any, Dict, Optional

from aiohttp import web

try:
    from pyisolate import ProxiedSingleton
    from pyisolate._internal.shared import current_rpc_context
except ImportError:
    # Fallback when pyisolate not installed
    class ProxiedSingleton:
        pass
    current_rpc_context = None

LOG_PREFIX = "📚 [PyIsolate]"
logger = logging.getLogger(__name__)


class PromptServerProxy(ProxiedSingleton):
    """Proxy for PromptServer.instance.
    
    Enables isolated nodes to:
    - Send messages to the UI (send_sync, send)
    - Register HTTP routes (via RouteInjector)
    - Access server configuration
    
    Rev 1.0: Per PYISOLATE_COMFY_INTEGRATION_ARCHITECTURE.md
    
    inspect.getfile Compatibility:
    -----------------------------
    Some nodes use inspect.getfile(PromptServer) to find the server.py location.
    This proxy sets __module__ = 'server' so inspect resolves correctly.
    """
    
    # === inspect.getfile compatibility (REVIEW FIX) ===
    # inspect.getfile() uses __module__ and sys.modules to find source.
    # We set __module__ to 'server' so it resolves to the real server.py
    __module__ = 'server'
    
    _instance: Optional['PromptServerProxy'] = None
    _source_file: Optional[str] = None
    
    def __new__(cls) -> 'PromptServerProxy':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def _get_source_file(cls) -> str:
        """Return path to real server.py for inspect.getfile compatibility."""
        if cls._source_file is None:
            import folder_paths
            cls._source_file = os.path.join(folder_paths.base_path, 'server.py')
        return cls._source_file
    
    # Make inspect.getfile work by providing __file__ at class level
    @property
    def __file__(self) -> str:
        return self._get_source_file()
    
    # =========================================================================
    # Instance Access (matches PromptServer.instance pattern)
    # =========================================================================
    
    @property
    def instance(self) -> 'PromptServerProxy':
        """Return self as the instance (matches PromptServer.instance pattern)."""
        return self
    
    # =========================================================================
    # UI Communication Methods
    # =========================================================================
    
    async def send_sync(self, event: str, data: Dict[str, Any], sid: Optional[str] = None) -> None:
        """Send synchronous message to UI.
        
        This is the primary method for nodes to communicate status/progress to the UI.
        """
        from server import PromptServer
        logger.debug(f"{LOG_PREFIX}[PromptServerProxy] send_sync({event}, sid={sid})")
        return await PromptServer.instance.send_sync(event, data, sid)
    
    async def send(self, event: str, data: Dict[str, Any], sid: Optional[str] = None) -> None:
        """Send async message to UI."""
        from server import PromptServer
        logger.debug(f"{LOG_PREFIX}[PromptServerProxy] send({event}, sid={sid})")
        return await PromptServer.instance.send(event, data, sid)
    
    def send_sync_blocking(self, event: str, data: Dict[str, Any], sid: Optional[str] = None) -> None:
        """Synchronous blocking version of send_sync for non-async contexts."""
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Schedule coroutine and wait
            import concurrent.futures
            future = asyncio.run_coroutine_threadsafe(
                self.send_sync(event, data, sid), 
                loop
            )
            future.result(timeout=5.0)
        else:
            loop.run_until_complete(self.send_sync(event, data, sid))
    
    # =========================================================================
    # Server Properties
    # =========================================================================
    
    @property
    def client_id(self) -> Optional[str]:
        """Current client session ID."""
        from server import PromptServer
        result = PromptServer.instance.client_id
        logger.debug(f"{LOG_PREFIX}[PromptServerProxy] client_id → {result}")
        return result
    
    def supports(self, feature: str) -> bool:
        """Check if server supports a feature."""
        from server import PromptServer
        result = PromptServer.instance.supports(feature)
        logger.debug(f"{LOG_PREFIX}[PromptServerProxy] supports({feature}) → {result}")
        return result
    
    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        """Event loop - return child's loop, not host's.
        
        Note: In isolated context, this returns the child process's event loop,
        not the host's. This is intentional for async coordination within the child.
        """
        return asyncio.get_event_loop()
    
    # =========================================================================
    # Route Registration (for RouteInjector)
    # =========================================================================
    
    def register_route(
        self, 
        method: str, 
        path: str, 
        handler: Any, 
        source: str = "local", 
        is_callback: bool = False
    ) -> None:
        """Register a route on the host PromptServer.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            path: URL path
            handler: Handler function or callback ID
            source: Source identifier for logging
            is_callback: If True, handler is a callback ID for RPC
        """
        from server import PromptServer
        
        logger.info(f"{LOG_PREFIX}[PromptServerProxy] register_route({method}, {path}) from {source} (callback={is_callback})")
        
        if is_callback:
            callback_id = handler
            rpc = current_rpc_context.get() if current_rpc_context else None
            if rpc is None:
                logger.error(f"{LOG_PREFIX}[PromptServerProxy] register_route called with is_callback=True but no RPC context found")
                return

            async def route_wrapper(request: web.Request) -> web.Response:
                # Convert request to serializable dict
                req_data = {
                    "method": request.method,
                    "path": request.path,
                    "query": dict(request.query),
                    "text": await request.text(),
                }
                
                try:
                    # Call the isolated handler via RPC callback
                    result = await rpc.call_callback(callback_id, req_data)
                    return self._serialize_response(result)
                except Exception as e:
                    logger.error(f"{LOG_PREFIX}[PromptServerProxy] Error in route handler: {e}")
                    return web.Response(status=500, text=f"Isolation Error: {e}")

            PromptServer.instance.register_route(method, path, route_wrapper)
        else:
            async def route_wrapper(request: web.Request) -> web.Response:
                req_data = {
                    "method": request.method,
                    "path": request.path,
                    "query": dict(request.query),
                }
                
                try:
                    result = await handler(req_data)
                    return self._serialize_response(result)
                except Exception as e:
                    logger.error(f"{LOG_PREFIX}[PromptServerProxy] Error in route handler: {e}")
                    return web.Response(status=500, text=f"Isolation Error: {e}")

            PromptServer.instance.register_route(method, path, route_wrapper)
    
    def _serialize_response(self, result: Any) -> web.Response:
        """Convert handler result to aiohttp Response."""
        if isinstance(result, dict):
            return web.json_response(result)
        elif isinstance(result, str):
            return web.Response(text=result)
        elif hasattr(result, "body") and hasattr(result, "status"):
            return web.Response(
                body=result.body, 
                status=result.status, 
                headers=getattr(result, 'headers', None)
            )
        else:
            return web.Response(text=str(result))

