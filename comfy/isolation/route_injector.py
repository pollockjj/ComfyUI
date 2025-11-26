"""Host-Side Route Injector for PyIsolate.

Rev 1.0 Implementation - Registers route shims with PromptServer that forward
requests to isolated processes via RPC.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Callable

from aiohttp import web

if TYPE_CHECKING:
    from pyisolate import Extension

LOG_PREFIX = "📚 [PyIsolate]"
logger = logging.getLogger(__name__)


class RouteInjector:
    """Manages route injection for isolated custom nodes.
    
    Reads route_manifest.json and registers shim handlers with the host
    PromptServer. These shims forward requests to the isolated process
    via RPC and return the response.
    
    Rev 1.0: Per HOST_SIDE_ROUTE_INJECTION_PLAN.md
    """
    
    def __init__(self):
        self._registered_routes: Dict[str, Dict[str, Any]] = {}
    
    def inject_routes(
        self, 
        prompt_server: Any, 
        extension: 'Extension', 
        manifest_path: Path
    ) -> int:
        """Register routes from manifest with PromptServer.
        
        Args:
            prompt_server: The PromptServer instance
            extension: PyIsolate extension handle for RPC
            manifest_path: Path to route_manifest.json
        
        Returns:
            Number of routes registered
        """
        if not manifest_path.exists():
            logger.debug(f"{LOG_PREFIX}[RouteInjector] No manifest at {manifest_path}")
            return 0
        
        try:
            manifest = json.loads(manifest_path.read_text())
        except json.JSONDecodeError as e:
            logger.error(f"{LOG_PREFIX}[RouteInjector] Invalid manifest JSON: {e}")
            return 0
        
        node_name = manifest.get("node_name", "unknown")
        routes = manifest.get("routes", [])
        
        if not routes:
            logger.debug(f"{LOG_PREFIX}[RouteInjector] No routes in manifest for {node_name}")
            return 0
        
        registered = 0
        for route_info in routes:
            try:
                self._register_route(prompt_server, extension, node_name, route_info)
                registered += 1
            except Exception as e:
                logger.error(
                    f"{LOG_PREFIX}[RouteInjector] Failed to register route "
                    f"{route_info.get('method', '?')} {route_info.get('path', '?')}: {e}"
                )
        
        logger.info(
            f"{LOG_PREFIX}[RouteInjector] Registered {registered}/{len(routes)} "
            f"routes for {node_name}"
        )
        return registered
    
    def _register_route(
        self,
        prompt_server: Any,
        extension: 'Extension',
        node_name: str,
        route_info: Dict[str, Any]
    ) -> None:
        """Register a single route with PromptServer."""
        path = route_info["path"]
        method = route_info["method"].lower()
        handler_module = route_info["handler_module"]
        handler_func = route_info["handler_func"]
        is_async = route_info.get("is_async", True)
        
        # Create the shim handler
        shim = self._create_route_shim(
            extension=extension,
            handler_module=handler_module,
            handler_func=handler_func,
            node_name=node_name,
            path=path,
            is_async=is_async,
        )
        
        # Get the route method (get, post, etc.)
        routes_obj = prompt_server.routes
        route_method = getattr(routes_obj, method, None)
        
        if route_method is None:
            raise ValueError(f"Unknown HTTP method: {method}")
        
        # Register with aiohttp
        route_method(path)(shim)
        
        # Track registration
        self._registered_routes[f"{method.upper()} {path}"] = {
            "node_name": node_name,
            "handler": f"{handler_module}.{handler_func}",
            "extension": extension,
        }
        
        logger.info(
            f"{LOG_PREFIX}[RouteInjector] Registered {method.upper()} {path} "
            f"→ {handler_module}.{handler_func} ({node_name})"
        )
    
    def _create_route_shim(
        self,
        extension: 'Extension',
        handler_module: str,
        handler_func: str,
        node_name: str,
        path: str,
        is_async: bool,
    ) -> Callable:
        """Create an async handler shim that forwards requests to isolated process.
        
        The shim:
        1. Serializes the incoming request to a dict
        2. Sends RPC call to isolated process
        3. Deserializes the response
        4. Returns appropriate aiohttp Response
        """
        
        async def shim_handler(request: web.Request) -> web.Response:
            """Forward request to isolated process via RPC."""
            try:
                # Serialize request to dict
                request_data = await self._serialize_request(request)
                
                # Call isolated handler via RPC
                result = await extension.call_route_handler(
                    handler_module=handler_module,
                    handler_func=handler_func,
                    request_data=request_data,
                )
                
                # Convert result to aiohttp Response
                return self._deserialize_response(result)
                
            except asyncio.TimeoutError:
                logger.error(
                    f"{LOG_PREFIX}[RouteInjector] Timeout calling {path} in {node_name}"
                )
                return web.Response(
                    status=504,
                    text=f"Gateway Timeout: Isolated handler for {path} timed out"
                )
            except Exception as e:
                logger.error(
                    f"{LOG_PREFIX}[RouteInjector] Error calling {path} in {node_name}: {e}"
                )
                return web.Response(
                    status=500,
                    text=f"Internal Server Error: {e}"
                )
        
        return shim_handler
    
    async def _serialize_request(self, request: web.Request) -> Dict[str, Any]:
        """Serialize aiohttp Request to a dict for RPC transport."""
        # Read body based on content type
        content_type = request.content_type or ""
        
        if "json" in content_type:
            try:
                body = await request.json()
            except json.JSONDecodeError:
                body = await request.text()
        elif "form" in content_type or "urlencoded" in content_type:
            body = dict(await request.post())
        elif "multipart" in content_type:
            # Handle multipart form data
            body = {}
            reader = await request.multipart()
            async for part in reader:
                if part.filename:
                    # File upload - read bytes
                    body[part.name] = {
                        "filename": part.filename,
                        "content_type": part.headers.get("Content-Type", "application/octet-stream"),
                        "data": (await part.read()).decode("utf-8", errors="replace")
                    }
                else:
                    body[part.name] = (await part.read()).decode("utf-8")
        else:
            body = await request.text()
        
        return {
            "method": request.method,
            "path": str(request.path),
            "query": dict(request.query),
            "headers": dict(request.headers),
            "content_type": content_type,
            "body": body,
            "match_info": dict(request.match_info),
        }
    
    def _deserialize_response(self, result: Any) -> web.Response:
        """Convert RPC result to aiohttp Response."""
        if result is None:
            return web.Response(status=204)
        
        # Handle dict responses (most common)
        if isinstance(result, dict):
            # Check if it's a serialized Response object
            if "type" in result:
                resp_type = result["type"]
                body = result.get("body", "")
                status = result.get("status", 200)
                headers = result.get("headers", {})
                
                if resp_type == "json":
                    return web.json_response(body, status=status, headers=headers)
                elif resp_type == "text":
                    return web.Response(text=body, status=status, headers=headers)
                elif resp_type == "binary":
                    return web.Response(
                        body=body.encode() if isinstance(body, str) else body,
                        status=status,
                        headers=headers
                    )
            else:
                # Plain dict - return as JSON
                return web.json_response(result)
        
        # Handle string responses
        if isinstance(result, str):
            return web.Response(text=result)
        
        # Handle bytes
        if isinstance(result, bytes):
            return web.Response(body=result)
        
        # Fallback - convert to string
        return web.Response(text=str(result))
    
    def get_registered_routes(self) -> Dict[str, Dict[str, Any]]:
        """Get dict of all registered routes."""
        return dict(self._registered_routes)


# Global injector instance
_injector: Optional[RouteInjector] = None


def get_route_injector() -> RouteInjector:
    """Get or create the global RouteInjector instance."""
    global _injector
    if _injector is None:
        _injector = RouteInjector()
    return _injector


def inject_routes(
    prompt_server: Any,
    extension: 'Extension',
    manifest_path: Path
) -> int:
    """Convenience function to inject routes using global injector.
    
    Args:
        prompt_server: The PromptServer instance
        extension: PyIsolate extension handle
        manifest_path: Path to route_manifest.json
    
    Returns:
        Number of routes registered
    """
    return get_route_injector().inject_routes(prompt_server, extension, manifest_path)
