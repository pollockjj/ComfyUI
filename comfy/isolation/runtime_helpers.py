from __future__ import annotations

import logging
import copy
from pathlib import Path
from typing import Dict, List, Set, TYPE_CHECKING, Any

from .proxies.helper_proxies import restore_input_types
from comfy_api.internal import _ComfyNodeInternal
from comfy_api.latest import _io as latest_io

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
    is_v3 = bool(info.get("is_v3", False))
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

    def _input_types(cls, include_hidden: bool = True, return_schema: bool = False, live_inputs: Any = None):
        if not is_v3:
            return restored_input_types

        # V3 path: mimic _ComfyNodeBaseInternal.INPUT_TYPES
        inputs_copy = copy.deepcopy(restored_input_types)

        # remove hidden if requested
        if not include_hidden:
            inputs_copy.pop("hidden", None)

        v3_data: Dict[str, Any] = {"hidden_inputs": {}}
        dynamic = inputs_copy.pop("dynamic_paths", None)
        if dynamic is not None:
            v3_data["dynamic_paths"] = dynamic

        if return_schema:
            hidden_vals = info.get("hidden", []) or []
            hidden_enums = []
            for h in hidden_vals:
                try:
                    hidden_enums.append(latest_io.Hidden(h))
                except Exception:
                    hidden_enums.append(h)

            class SchemaProxy:
                hidden = hidden_enums

            return inputs_copy, SchemaProxy, v3_data

        return inputs_copy

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
    bases = (_ComfyNodeInternal,) if is_v3 else ()
    stub_cls = type(class_name, bases, attributes)
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
