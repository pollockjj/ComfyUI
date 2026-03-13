# pylint: disable=logging-fstring-interpolation
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, TypedDict

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore[no-redef]

logger = logging.getLogger(__name__)

HOST_POLICY_PATH_ENV = "COMFY_HOST_POLICY_PATH"
VALID_SANDBOX_MODES = frozenset({"required", "disabled"})


class HostSecurityPolicy(TypedDict):
    sandbox_mode: str
    allow_network: bool
    writable_paths: List[str]
    readonly_paths: List[str]
    whitelist: Dict[str, str]


DEFAULT_POLICY: HostSecurityPolicy = {
    "sandbox_mode": "required",
    "allow_network": False,
    "writable_paths": ["/dev/shm", "/tmp"],
    "readonly_paths": [],
    "whitelist": {},
}


def _default_policy() -> HostSecurityPolicy:
    return {
        "sandbox_mode": DEFAULT_POLICY["sandbox_mode"],
        "allow_network": DEFAULT_POLICY["allow_network"],
        "writable_paths": list(DEFAULT_POLICY["writable_paths"]),
        "readonly_paths": list(DEFAULT_POLICY["readonly_paths"]),
        "whitelist": dict(DEFAULT_POLICY["whitelist"]),
    }


def load_host_policy(comfy_root: Path) -> HostSecurityPolicy:
    config_override = os.environ.get(HOST_POLICY_PATH_ENV)
    config_path = Path(config_override) if config_override else comfy_root / "pyproject.toml"
    policy = _default_policy()

    if not config_path.exists():
        logger.debug("Host policy file missing at %s, using defaults.", config_path)
        return policy

    try:
        with config_path.open("rb") as f:
            data = tomllib.load(f)
    except Exception:
        logger.warning(
            "Failed to parse host policy from %s, using defaults.",
            config_path,
            exc_info=True,
        )
        return policy

    tool_config = data.get("tool", {}).get("comfy", {}).get("host", {})
    if not isinstance(tool_config, dict):
        logger.debug("No [tool.comfy.host] section found, using defaults.")
        return policy

    sandbox_mode = tool_config.get("sandbox_mode")
    if isinstance(sandbox_mode, str):
        normalized_sandbox_mode = sandbox_mode.strip().lower()
        if normalized_sandbox_mode in VALID_SANDBOX_MODES:
            policy["sandbox_mode"] = normalized_sandbox_mode
        else:
            logger.warning(
                "Invalid host sandbox_mode %r in %s, using default %r.",
                sandbox_mode,
                config_path,
                DEFAULT_POLICY["sandbox_mode"],
            )

    if "allow_network" in tool_config:
        policy["allow_network"] = bool(tool_config["allow_network"])

    if "writable_paths" in tool_config:
        policy["writable_paths"] = [str(p) for p in tool_config["writable_paths"]]

    if "readonly_paths" in tool_config:
        policy["readonly_paths"] = [str(p) for p in tool_config["readonly_paths"]]

    whitelist_raw = tool_config.get("whitelist")
    if isinstance(whitelist_raw, dict):
        policy["whitelist"] = {str(k): str(v) for k, v in whitelist_raw.items()}

    logger.debug(
        "Loaded Host Policy: %d whitelisted nodes, Sandbox=%s, Network=%s",
        len(policy["whitelist"]),
        policy["sandbox_mode"],
        policy["allow_network"],
    )
    return policy


__all__ = ["HostSecurityPolicy", "load_host_policy", "DEFAULT_POLICY"]
