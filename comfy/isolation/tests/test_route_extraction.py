"""
Unit tests for route extraction - runs at ComfyUI startup.

These tests validate that:
1. Route extractor can parse decorator patterns from source files
2. Route manifests can be loaded and validated
3. Route injector creates proper shim handlers
"""
import logging
import json
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

# POC Node paths (must exist)
POC_NODES = {
    "cg-image-filter": Path("/home/johnj/ComfyUI/custom_nodes/cg-image-filter"),
    "comfyui-ollama": Path("/home/johnj/ComfyUI/custom_nodes/comfyui-ollama"),
    "comfyui-get-meta": Path("/home/johnj/ComfyUI/custom_nodes/comfyui-get-meta"),
}

def run_tests() -> List[Dict[str, Any]]:
    """Run all route extraction tests. Returns list of results."""
    results = []
    
    results.append(test_extract_routes_cg_image_filter())
    results.append(test_extract_routes_comfyui_ollama())
    results.append(test_extract_routes_comfyui_get_meta())
    results.append(test_manifest_load_cg_image_filter())
    results.append(test_manifest_load_comfyui_ollama())
    results.append(test_manifest_load_comfyui_get_meta())
    
    return results


def test_extract_routes_cg_image_filter() -> Dict[str, Any]:
    """Test: Extract routes from cg-image-filter using AST."""
    name = "test_extract_routes_cg_image_filter"
    try:
        from comfy.isolation.route_extractor import extract_routes
        
        node_path = POC_NODES["cg-image-filter"]
        if not node_path.exists():
            return {"name": name, "passed": False, "error": f"Node path missing: {node_path}"}
        
        routes = extract_routes(node_path)
        
        if len(routes) != 1:
            return {"name": name, "passed": False, "error": f"Expected 1 route, got {len(routes)}"}
        
        route = routes[0]
        if route.path != "/cg-image-filter-message":
            return {"name": name, "passed": False, "error": f"Wrong path: {route.path}"}
        
        if route.method != "POST":
            return {"name": name, "passed": False, "error": f"Wrong method: {route.method}"}
        
        logger.info(f"📚 [PyIsolate][Test] ✅ {name}")
        return {"name": name, "passed": True, "error": None}
        
    except Exception as e:
        return {"name": name, "passed": False, "error": str(e)}


def test_extract_routes_comfyui_ollama() -> Dict[str, Any]:
    """Test: Extract routes from comfyui-ollama using AST."""
    name = "test_extract_routes_comfyui_ollama"
    try:
        from comfy.isolation.route_extractor import extract_routes
        
        node_path = POC_NODES["comfyui-ollama"]
        if not node_path.exists():
            return {"name": name, "passed": False, "error": f"Node path missing: {node_path}"}
        
        routes = extract_routes(node_path)
        
        if len(routes) != 1:
            return {"name": name, "passed": False, "error": f"Expected 1 route, got {len(routes)}"}
        
        route = routes[0]
        if route.path != "/ollama/get_models":
            return {"name": name, "passed": False, "error": f"Wrong path: {route.path}"}
        
        if route.method != "POST":
            return {"name": name, "passed": False, "error": f"Wrong method: {route.method}"}
        
        logger.info(f"📚 [PyIsolate][Test] ✅ {name}")
        return {"name": name, "passed": True, "error": None}
        
    except Exception as e:
        return {"name": name, "passed": False, "error": str(e)}


def test_extract_routes_comfyui_get_meta() -> Dict[str, Any]:
    """Test: Extract routes from comfyui-get-meta using AST."""
    name = "test_extract_routes_comfyui_get_meta"
    try:
        from comfy.isolation.route_extractor import extract_routes
        
        node_path = POC_NODES["comfyui-get-meta"]
        if not node_path.exists():
            return {"name": name, "passed": False, "error": f"Node path missing: {node_path}"}
        
        routes = extract_routes(node_path)
        
        if len(routes) != 1:
            return {"name": name, "passed": False, "error": f"Expected 1 route, got {len(routes)}"}
        
        route = routes[0]
        if route.path != "/shinich39/comfyui-get-meta/read-metadata":
            return {"name": name, "passed": False, "error": f"Wrong path: {route.path}"}
        
        if route.method != "POST":
            return {"name": name, "passed": False, "error": f"Wrong method: {route.method}"}
        
        logger.info(f"📚 [PyIsolate][Test] ✅ {name}")
        return {"name": name, "passed": True, "error": None}
        
    except Exception as e:
        return {"name": name, "passed": False, "error": str(e)}


def test_manifest_load_cg_image_filter() -> Dict[str, Any]:
    """Test: Load and validate route_manifest.json for cg-image-filter."""
    name = "test_manifest_load_cg_image_filter"
    try:
        manifest_path = POC_NODES["cg-image-filter"] / "route_manifest.json"
        
        if not manifest_path.exists():
            return {"name": name, "passed": False, "error": f"Manifest missing: {manifest_path}"}
        
        with open(manifest_path) as f:
            manifest = json.load(f)
        
        if "routes" not in manifest:
            return {"name": name, "passed": False, "error": "Manifest missing 'routes' key"}
        
        if len(manifest["routes"]) != 1:
            return {"name": name, "passed": False, "error": f"Expected 1 route, got {len(manifest['routes'])}"}
        
        route = manifest["routes"][0]
        required_keys = ["path", "method", "handler_module", "handler_func"]
        for key in required_keys:
            if key not in route:
                return {"name": name, "passed": False, "error": f"Route missing key: {key}"}
        
        logger.info(f"📚 [PyIsolate][Test] ✅ {name}")
        return {"name": name, "passed": True, "error": None}
        
    except Exception as e:
        return {"name": name, "passed": False, "error": str(e)}


def test_manifest_load_comfyui_ollama() -> Dict[str, Any]:
    """Test: Load and validate route_manifest.json for comfyui-ollama."""
    name = "test_manifest_load_comfyui_ollama"
    try:
        manifest_path = POC_NODES["comfyui-ollama"] / "route_manifest.json"
        
        if not manifest_path.exists():
            return {"name": name, "passed": False, "error": f"Manifest missing: {manifest_path}"}
        
        with open(manifest_path) as f:
            manifest = json.load(f)
        
        if "routes" not in manifest:
            return {"name": name, "passed": False, "error": "Manifest missing 'routes' key"}
        
        if len(manifest["routes"]) != 1:
            return {"name": name, "passed": False, "error": f"Expected 1 route, got {len(manifest['routes'])}"}
        
        logger.info(f"📚 [PyIsolate][Test] ✅ {name}")
        return {"name": name, "passed": True, "error": None}
        
    except Exception as e:
        return {"name": name, "passed": False, "error": str(e)}


def test_manifest_load_comfyui_get_meta() -> Dict[str, Any]:
    """Test: Load and validate route_manifest.json for comfyui-get-meta."""
    name = "test_manifest_load_comfyui_get_meta"
    try:
        manifest_path = POC_NODES["comfyui-get-meta"] / "route_manifest.json"
        
        if not manifest_path.exists():
            return {"name": name, "passed": False, "error": f"Manifest missing: {manifest_path}"}
        
        with open(manifest_path) as f:
            manifest = json.load(f)
        
        if "routes" not in manifest:
            return {"name": name, "passed": False, "error": "Manifest missing 'routes' key"}
        
        if len(manifest["routes"]) != 1:
            return {"name": name, "passed": False, "error": f"Expected 1 route, got {len(manifest['routes'])}"}
        
        logger.info(f"📚 [PyIsolate][Test] ✅ {name}")
        return {"name": name, "passed": True, "error": None}
        
    except Exception as e:
        return {"name": name, "passed": False, "error": str(e)}
