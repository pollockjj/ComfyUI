"""ProxiedSingleton for PromptServer."""

import logging
from typing import Any
from pyisolate import ProxiedSingleton
from pyisolate._internal.shared import current_rpc_context
from server import PromptServer
from comfy.isolation import LOG_PREFIX
from aiohttp import web

logger = logging.getLogger(__name__)

class PromptServerProxy(ProxiedSingleton):
    """Proxy for PromptServer singleton."""
    
    def register_route(self, method: str, path: str, handler: Any, source: str = "local", is_callback: bool = False) -> None:
        """Register a route on the host PromptServer."""
        logger.info(f"{LOG_PREFIX}[PromptServerProxy] register_route({method}, {path}) from {source} (callback={is_callback})")
        
        if is_callback:
            callback_id = handler
            rpc = current_rpc_context.get()
            if rpc is None:
                logger.error(f"{LOG_PREFIX}[PromptServerProxy] register_route called with is_callback=True but no RPC context found")
                return

            async def route_wrapper(request):
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
                    
                    # Handle response
                    if isinstance(result, dict):
                        return web.json_response(result)
                    elif isinstance(result, str):
                        return web.Response(text=result)
                    elif hasattr(result, "body") and hasattr(result, "status"):
                        return web.Response(body=result.body, status=result.status, headers=result.headers)
                    else:
                        return web.Response(text=str(result))
                except Exception as e:
                    logger.error(f"{LOG_PREFIX}[PromptServerProxy] Error in route handler: {e}")
                    return web.Response(status=500, text=f"Isolation Error: {e}")

            # Use the real PromptServer instance to register the route
            PromptServer.instance.register_route(method, path, route_wrapper)
        else:
            # Legacy/Direct handler support
            async def route_wrapper(request):
                # Convert request to serializable dict
                req_data = {
                    "method": request.method,
                    "path": request.path,
                    "query": dict(request.query),
                }
                
                try:
                    # Call the isolated handler via RPC
                    # handler is a proxy to the function in the isolated process
                    result = await handler(req_data)
                    
                    # Handle response
                    if isinstance(result, dict):
                        return web.json_response(result)
                    elif isinstance(result, str):
                        return web.Response(text=result)
                    # Check for serialized Response object (if passed as dict/object)
                    elif hasattr(result, "body") and hasattr(result, "status"):
                        return web.Response(body=result.body, status=result.status, headers=result.headers)
                    else:
                        return web.Response(text=str(result))
                except Exception as e:
                    logger.error(f"{LOG_PREFIX}[PromptServerProxy] Error in route handler: {e}")
                    return web.Response(status=500, text=f"Isolation Error: {e}")

            # Use the real PromptServer instance to register the route
            PromptServer.instance.register_route(method, path, route_wrapper)

