"""ProxiedSingleton for PromptServer."""

import logging
from typing import Any
from pyisolate import ProxiedSingleton
from server import PromptServer
from comfy.isolation import LOG_PREFIX
from aiohttp import web

logger = logging.getLogger(__name__)

class PromptServerProxy(ProxiedSingleton):
    """Proxy for PromptServer singleton."""
    
    def register_route(self, method: str, path: str, handler: Any, source: str = "local") -> None:
        """Register a route on the host PromptServer."""
        logger.info(f"{LOG_PREFIX}[PromptServerProxy] register_route({method}, {path}) from {source}")
        
        async def route_wrapper(request):
            # Convert request to serializable dict
            # TODO: Implement full Request proxying
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
        # We must access the class attribute directly to avoid recursion if 'instance' is patched
        real_instance = PromptServer.instance
        # If we are in the host process, PromptServer.instance might be this proxy if patched incorrectly
        # But wait, this code runs on the HOST.
        # The issue is that in extension_wrapper.py we patched server.PromptServer.instance = proxy
        # So when we call PromptServer.instance.register_route, we are calling self.register_route!
        
        # We need to find the REAL PromptServer instance.
        # Since we are running on the host, we can find it by looking at the garbage collector or 
        # by assuming the original instance is stored somewhere.
        # Better yet, we should NOT patch PromptServer.instance on the HOST.
        # We only patch it in the ISOLATED process.
        
        # Wait, this file is imported by BOTH host and isolated process?
        # No, ProxiedSingletons are instantiated on the host.
        # The isolated process gets a proxy object via RPC.
        
        # The recursion happens because PromptServer.instance IS this proxy object on the host?
        # No, on the host PromptServer.instance should be the real server.
        
        # Let's check where we patched it.
        # extension_wrapper.py: before_module_loaded -> patches PromptServer.instance
        # This runs in the ISOLATED process.
        
        # So why does the HOST code recurse?
        # Ah, the traceback shows the recursion happening in `register_route`.
        # If this code is running on the HOST, then `PromptServer.instance` should be the real server.
        
        # UNLESS... we are running this code in the ISOLATED process?
        # If this is a ProxiedSingleton, the methods run on the HOST via RPC.
        # BUT, if we call `PromptServer.instance.register_route` in the isolated process,
        # and `PromptServer.instance` is the proxy, then it calls `register_route` on the proxy.
        # The proxy (client side) sends RPC to host.
        # Host executes `PromptServerProxy.register_route`.
        # Host calls `PromptServer.instance.register_route`.
        
        # If `PromptServer.instance` on the HOST is ALSO the proxy, then we have infinite recursion.
        # Did we patch it on the host?
        # comfy/isolation/__init__.py imports PromptServerProxy and instantiates it.
        # It does NOT patch PromptServer.instance.
        
        # However, `server.py` sets `PromptServer.instance = self` in `__init__`.
        
        # Let's look at the traceback again.
        # File "/home/johnj/ComfyUI/comfy/isolation/proxies/prompt_server_proxy.py", line 47, in register_route
        # PromptServer.instance.register_route(method, path, route_wrapper, source)
        
        # This implies that `PromptServer.instance` IS `PromptServerProxy` (self).
        # How did that happen on the host?
        
        # Maybe `PromptServer.instance` is NOT the proxy, but `register_route` is calling ITSELF?
        # No, `PromptServer` has `register_route` method now (we added it).
        
        # Wait, if `PromptServer.instance` is the real server, then `PromptServer.instance.register_route`
        # calls the method in `server.py`.
        
        # Is it possible that `PromptServerProxy` inherits from `PromptServer`?
        # No, it inherits from `ProxiedSingleton`.
        
        # Is it possible that `PromptServer` IS `PromptServerProxy`?
        # No.
        
        # Let's debug by printing the type of PromptServer.instance
        logger.info(f"{LOG_PREFIX}[PromptServerProxy] PromptServer.instance type: {type(PromptServer.instance)}")
        
        PromptServer.instance.register_route(method, path, route_wrapper, source)
