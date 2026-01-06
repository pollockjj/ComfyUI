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
    "allow_network": False,  # Secure by default
    "writable_paths": ["/dev/shm", "/tmp"],
    "readonly_paths": [],
    "whitelist": {},
}

def load_host_policy(comfy_root: Path) -> HostSecurityPolicy:
    """
    Load the Host Security Policy from ComfyUI/pyproject.toml.

    Schema:
    [tool.comfy.host]
    allow_network = bool (default: false)
    writable_paths = [str] (default: /dev/shm, /tmp)
    readonly_paths = [str] (default: [])

    [tool.comfy.host.whitelist]
    "NodeName" = "version_spec" (or "*")
    """
    config_path = comfy_root / "pyproject.toml"

    if not config_path.exists():
        logger.info("No host policy found (pyproject.toml missing). Using secure defaults.")
        # Return a COPY of defaults
        return DEFAULT_POLICY.copy()

    try:
        with config_path.open("rb") as f:
            data = tomllib.load(f)

        tool_config = data.get("tool", {}).get("comfy", {}).get("host", {})

        policy = DEFAULT_POLICY.copy()

        # Override defaults if present
        if "allow_network" in tool_config:
            policy["allow_network"] = bool(tool_config["allow_network"])

        if "writable_paths" in tool_config:
            # Merge or Replace?
            # Security wisdom: Replace allows tighter control, but merging is friendlier.
            # Decision: Replace. User must explicitly list defaults if they want to keep them AND add more,
            # OR we ensure criticals (/dev/shm) are always added by the consumer (extension_loader).
            # Let's trust the user config but ensure we don't break execution.
            # Actually, let's keep it simple: Replace.
            paths = tool_config["writable_paths"]
            if isinstance(paths, list):
                policy["writable_paths"] = [str(p) for p in paths]

        if "readonly_paths" in tool_config:
            paths = tool_config["readonly_paths"]
            if isinstance(paths, list):
                policy["readonly_paths"] = [str(p) for p in paths]

        whitelist = tool_config.get("whitelist", {})
        if isinstance(whitelist, dict):
            policy["whitelist"] = {str(k): str(v) for k, v in whitelist.items()}

        logger.debug(f"Loaded Host Policy: {len(policy['whitelist'])} whitelisted nodes, Network={policy['allow_network']}")
        return policy

    except Exception as e:
        logger.warning(f"Failed to parse host policy from {config_path}: {e}. Using defaults.")
        return DEFAULT_POLICY.copy()

__all__ = ["HostSecurityPolicy", "load_host_policy", "DEFAULT_POLICY"]
