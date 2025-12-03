from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Set, TYPE_CHECKING

from .proxies.helper_proxies import restore_input_types

if TYPE_CHECKING:
    from .extension_wrapper import ComfyNodeExtension

LOG_PREFIX = "[I]"


def build_stub_class(
    node_name: str,
    info: Dict[str, object],
    extension: "ComfyNodeExtension",
    running_extensions: Dict[str, "ComfyNodeExtension"],
    logger: logging.Logger,
) -> type:
    function_name = "_pyisolate_execute"
    restored_input_types = restore_input_types(info.get("input_types", {}))

    async def _execute(self, **inputs):
        extension.ensure_process_started()
        running_extensions[extension.name] = extension

        try:
            from pyisolate._internal.model_serialization import (
                serialize_for_isolation,
                deserialize_from_isolation,
            )

            serialized = serialize_for_isolation(inputs)
            result = await extension.execute_node(node_name, **serialized)
            return deserialize_from_isolation(result)
        except ImportError as exc:  # pragma: no cover - optional dependency
            logger.warning("%s[Serialization] Serialization not available: %s", LOG_PREFIX, exc)
            return await extension.execute_node(node_name, **inputs)

    def _input_types(cls):
        return restored_input_types

    attributes: Dict[str, object] = {
        "FUNCTION": function_name,
        "CATEGORY": info.get("category", ""),
        "OUTPUT_NODE": info.get("output_node", False),
        "RETURN_TYPES": tuple(info.get("return_types", ()) or ()),
        "RETURN_NAMES": info.get("return_names"),
        function_name: _execute,
        "_pyisolate_extension": extension,
        "_pyisolate_node_name": node_name,
        "INPUT_TYPES": classmethod(_input_types),
    }

    output_is_list = info.get("output_is_list")
    if output_is_list is not None:
        attributes["OUTPUT_IS_LIST"] = tuple(output_is_list)

    display_name = info.get("display_name") or node_name
    class_name = f"PyIsolate_{node_name}".replace(" ", "_")
    stub_cls = type(class_name, (), attributes)
    stub_cls.__doc__ = f"PyIsolate proxy node for {display_name}"
    return stub_cls


def get_class_types_for_extension(
    extension_name: str,
    running_extensions: Dict[str, "ComfyNodeExtension"],
    specs: List["IsolatedNodeSpec"],
) -> Set[str]:
    extension = running_extensions.get(extension_name)
    if not extension:
        return set()

    ext_path = Path(extension.module_path)
    class_types = set()
    for spec in specs:
        if spec.module_path.resolve() == ext_path.resolve():
            class_types.add(spec.node_name)
    return class_types


__all__ = ["build_stub_class", "get_class_types_for_extension"]
