from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, TypedDict

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore[no-redef]

logger = logging.getLogger(__name__)

class HostSecurityPolicy(TypedDict):
    allow_network: bool
    writable_paths: List[str]
    readonly_paths: List[str]
    whitelist: Dict[str, str]

DEFAULT_POLICY: HostSecurityPolicy = {
    "allow_network": False,
    "writable_paths": ["/dev/shm", "/tmp"],
    "readonly_paths": [],
    "whitelist": {},
}

def load_host_policy(comfy_root: Path) -> HostSecurityPolicy:
    config_path = comfy_root / "pyproject.toml"
    
    with config_path.open("rb") as f:
        data = tomllib.load(f)
    
    tool_config = data["tool"]["comfy"]["host"]
    policy = DEFAULT_POLICY.copy()
    
    if "allow_network" in tool_config:
        policy["allow_network"] = bool(tool_config["allow_network"])
    
    if "writable_paths" in tool_config:
        policy["writable_paths"] = [str(p) for p in tool_config["writable_paths"]]
    
    if "readonly_paths" in tool_config:
        policy["readonly_paths"] = [str(p) for p in tool_config["readonly_paths"]]
    
    if "whitelist" in tool_config:
        policy["whitelist"] = {str(k): str(v) for k, v in tool_config["whitelist"].items()}
    
    logger.debug(f"Loaded Host Policy: {len(policy['whitelist'])} whitelisted nodes, Network={policy['allow_network']}")
    return policy

__all__ = ["HostSecurityPolicy", "load_host_policy", "DEFAULT_POLICY"]
