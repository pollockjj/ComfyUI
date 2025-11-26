"""AST-based route extraction for PyIsolate.

Rev 1.0 Implementation - Extracts @PromptServer.instance.routes.* decorators
from custom node source files and generates route_manifest.json.
"""

from __future__ import annotations

import ast
import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

LOG_PREFIX = "📚 [PyIsolate]"
logger = logging.getLogger(__name__)


@dataclass
class ExtractedRoute:
    """Represents an extracted route from AST analysis."""
    path: str                    # e.g., "/cg-image-filter-message"
    method: str                  # e.g., "POST", "GET"
    handler_module: str          # e.g., "image_filter_messaging"
    handler_func: str            # e.g., "cg_image_filter_message"
    source_file: str             # Relative path to source file
    line_number: int             # Line number of decorator
    is_async: bool = True        # Whether handler is async
    
    def to_dict(self) -> dict:
        return asdict(self)


class RouteExtractor(ast.NodeVisitor):
    """AST visitor that extracts PromptServer route decorators.
    
    Detects patterns like:
        @PromptServer.instance.routes.post('/path')
        @PromptServer.instance.routes.get('/path')
        @server.PromptServer.instance.routes.post('/path')
    
    And generates route manifest entries for host-side registration.
    """
    
    def __init__(self, source_file: str, module_name: str):
        self.source_file = source_file
        self.module_name = module_name
        self.routes: List[ExtractedRoute] = []
        self._current_decorators: List[ast.expr] = []
    
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit function definition and check for route decorators."""
        self._check_function_for_routes(node, is_async=False)
        self.generic_visit(node)
    
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Visit async function definition and check for route decorators."""
        self._check_function_for_routes(node, is_async=True)
        self.generic_visit(node)
    
    def _check_function_for_routes(
        self, 
        node: ast.FunctionDef | ast.AsyncFunctionDef, 
        is_async: bool
    ) -> None:
        """Check if function has route decorators and extract them."""
        for decorator in node.decorator_list:
            route_info = self._parse_route_decorator(decorator)
            if route_info:
                path, method = route_info
                route = ExtractedRoute(
                    path=path,
                    method=method.upper(),
                    handler_module=self.module_name,
                    handler_func=node.name,
                    source_file=self.source_file,
                    line_number=decorator.lineno,
                    is_async=is_async,
                )
                self.routes.append(route)
                logger.debug(
                    f"{LOG_PREFIX}[RouteExtractor] Found {method.upper()} {path} "
                    f"→ {self.module_name}.{node.name} (line {decorator.lineno})"
                )
    
    def _parse_route_decorator(self, decorator: ast.expr) -> Optional[tuple[str, str]]:
        """Parse a decorator to extract route path and method.
        
        Returns:
            Tuple of (path, method) if this is a route decorator, None otherwise.
        """
        # Handle @decorator(args) form
        if isinstance(decorator, ast.Call):
            # Check if it's a routes decorator
            method = self._get_route_method(decorator.func)
            if method and decorator.args:
                # Get the path argument
                path_arg = decorator.args[0]
                if isinstance(path_arg, ast.Constant) and isinstance(path_arg.value, str):
                    return (path_arg.value, method)
        
        return None
    
    def _get_route_method(self, node: ast.expr) -> Optional[str]:
        """Check if node is a PromptServer.instance.routes.<method> access.
        
        Handles patterns:
            - PromptServer.instance.routes.post
            - server.PromptServer.instance.routes.get
            - routes.post (if routes is aliased)
        
        Returns:
            HTTP method name if this is a route decorator, None otherwise.
        """
        # Pattern: X.Y.Z.method where Z is 'routes'
        if not isinstance(node, ast.Attribute):
            return None
        
        method_name = node.attr  # e.g., 'post', 'get'
        
        # Check if it's a known HTTP method
        known_methods = {'get', 'post', 'put', 'patch', 'delete', 'head', 'options'}
        if method_name.lower() not in known_methods:
            return None
        
        # Now check if the value chain contains 'routes' and 'PromptServer'
        value = node.value
        
        # Walk up the attribute chain looking for 'routes'
        if isinstance(value, ast.Attribute) and value.attr == 'routes':
            # Check for 'instance' or 'PromptServer'
            inner = value.value
            if self._is_prompt_server_chain(inner):
                return method_name
        
        # Also handle aliased patterns like: routes = PromptServer.instance.routes
        # In this case, we see routes.post directly
        # For now, we don't support this - would need data flow analysis
        
        return None
    
    def _is_prompt_server_chain(self, node: ast.expr) -> bool:
        """Check if node represents PromptServer.instance or server.PromptServer.instance."""
        # Pattern 1: PromptServer.instance
        if isinstance(node, ast.Attribute) and node.attr == 'instance':
            inner = node.value
            if isinstance(inner, ast.Name) and inner.id == 'PromptServer':
                return True
            # Pattern 2: server.PromptServer.instance
            if isinstance(inner, ast.Attribute) and inner.attr == 'PromptServer':
                return True
        
        return False


def extract_routes_from_file(file_path: Path, module_name: str) -> List[ExtractedRoute]:
    """Extract routes from a single Python file.
    
    Args:
        file_path: Path to the Python file
        module_name: Module name for the extracted routes
    
    Returns:
        List of extracted routes
    """
    try:
        source = file_path.read_text(encoding='utf-8')
        tree = ast.parse(source, filename=str(file_path))
        
        extractor = RouteExtractor(
            source_file=str(file_path.name),
            module_name=module_name
        )
        extractor.visit(tree)
        
        return extractor.routes
    except SyntaxError as e:
        logger.warning(f"{LOG_PREFIX}[RouteExtractor] Syntax error in {file_path}: {e}")
        return []
    except Exception as e:
        logger.warning(f"{LOG_PREFIX}[RouteExtractor] Error parsing {file_path}: {e}")
        return []


def extract_routes(node_path: Path) -> List[ExtractedRoute]:
    """Extract all routes from a custom node directory.
    
    Scans all Python files in the node directory for route decorators.
    
    Args:
        node_path: Path to the custom node directory
    
    Returns:
        List of all extracted routes
    """
    all_routes: List[ExtractedRoute] = []
    
    if not node_path.is_dir():
        logger.warning(f"{LOG_PREFIX}[RouteExtractor] Not a directory: {node_path}")
        return all_routes
    
    # Scan all Python files
    for py_file in node_path.rglob("*.py"):
        # Skip __pycache__ and hidden directories
        if '__pycache__' in str(py_file) or py_file.name.startswith('.'):
            continue
        
        # Compute module name relative to node path
        rel_path = py_file.relative_to(node_path)
        module_name = str(rel_path.with_suffix('')).replace('/', '.').replace('\\', '.')
        
        routes = extract_routes_from_file(py_file, module_name)
        all_routes.extend(routes)
    
    logger.info(f"{LOG_PREFIX}[RouteExtractor] Found {len(all_routes)} routes in {node_path.name}")
    return all_routes


def generate_route_manifest(node_path: Path, output_path: Optional[Path] = None) -> dict:
    """Generate route_manifest.json for a custom node.
    
    Args:
        node_path: Path to the custom node directory
        output_path: Optional path for output file. If None, uses node_path/route_manifest.json
    
    Returns:
        The manifest dict
    """
    routes = extract_routes(node_path)
    
    manifest = {
        "node_name": node_path.name,
        "version": "1.0",
        "routes": [route.to_dict() for route in routes]
    }
    
    if output_path is None:
        output_path = node_path / "route_manifest.json"
    
    output_path.write_text(json.dumps(manifest, indent=2))
    logger.info(f"{LOG_PREFIX}[RouteExtractor] Generated manifest: {output_path}")
    
    return manifest


def validate_route_manifest(manifest_path: Path) -> tuple[bool, List[str]]:
    """Validate a route manifest file.
    
    Returns:
        Tuple of (is_valid, list of error messages)
    """
    errors: List[str] = []
    
    if not manifest_path.exists():
        return False, ["Manifest file does not exist"]
    
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as e:
        return False, [f"Invalid JSON: {e}"]
    
    if "routes" not in manifest:
        errors.append("Missing 'routes' key")
    
    if "node_name" not in manifest:
        errors.append("Missing 'node_name' key")
    
    for i, route in enumerate(manifest.get("routes", [])):
        if "path" not in route:
            errors.append(f"Route {i}: missing 'path'")
        if "method" not in route:
            errors.append(f"Route {i}: missing 'method'")
        if "handler_func" not in route:
            errors.append(f"Route {i}: missing 'handler_func'")
    
    return len(errors) == 0, errors
