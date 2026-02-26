from __future__ import annotations

import copy
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Set, TYPE_CHECKING

from .proxies.helper_proxies import restore_input_types
from comfy_api.internal import _ComfyNodeInternal
from comfy_api.latest import _io as latest_io

if TYPE_CHECKING:
    from .extension_wrapper import ComfyNodeExtension

LOG_PREFIX = "]["


def _flush_tensor_transport_state(marker: str, logger: logging.Logger) -> None:
    try:
        from pyisolate import flush_tensor_keeper  # type: ignore[attr-defined]
    except Exception:
        return
    if not callable(flush_tensor_keeper):
        return
    flushed = flush_tensor_keeper()
    if flushed > 0:
        logger.debug("%s %s flush_tensor_keeper released=%d", LOG_PREFIX, marker, flushed)


def _relieve_host_vram_pressure(marker: str, logger: logging.Logger) -> None:
    import comfy.model_management as model_management

    model_management.cleanup_models_gc()
    model_management.cleanup_models()

    device = model_management.get_torch_device()
    if not hasattr(device, "type") or device.type == "cpu":
        return

    required = model_management.minimum_inference_memory()
    if model_management.get_free_memory(device) < required:
        model_management.free_memory(required, device, for_dynamic=True)
        model_management.cleanup_models()
        model_management.soft_empty_cache()
        logger.debug("%s %s free_memory target=%d", LOG_PREFIX, marker, required)


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
        from comfy.isolation import _RUNNING_EXTENSIONS
        # Update BOTH the local dict AND the module-level dict
        running_extensions[extension.name] = extension
        _RUNNING_EXTENSIONS[extension.name] = extension
        prev_child = None
        try:
            if os.environ.get("PYISOLATE_ISOLATION_ACTIVE") == "1":
                _relieve_host_vram_pressure("RUNTIME:pre_execute", logger)
                _flush_tensor_transport_state("RUNTIME:pre_execute", logger)
            from pyisolate._internal.model_serialization import (
                serialize_for_isolation,
                deserialize_from_isolation,
            )
            prev_child = os.environ.pop("PYISOLATE_CHILD", None)
            serialized = serialize_for_isolation(inputs)
            result = await extension.execute_node(node_name, **serialized)
            deserialized = await deserialize_from_isolation(result, extension)
            return deserialized
        except ImportError:
            return await extension.execute_node(node_name, **inputs)
        except Exception:
            raise
        finally:
            if prev_child is not None:
                os.environ["PYISOLATE_CHILD"] = prev_child
            if os.environ.get("PYISOLATE_ISOLATION_ACTIVE") == "1":
                _flush_tensor_transport_state("RUNTIME:post_execute", logger)
    def _input_types(cls, include_hidden: bool = True, return_schema: bool = False, live_inputs: Any = None):
        if not is_v3:
            return restored_input_types

        inputs_copy = copy.deepcopy(restored_input_types)
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

    def _validate_class(cls):
        return True

    def _get_node_info_v1(cls):
        return info.get("schema_v1", {})

    def _get_base_class(cls):
        return latest_io.ComfyNode

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

    if is_v3:
        attributes["VALIDATE_CLASS"] = classmethod(_validate_class)
        attributes["GET_NODE_INFO_V1"] = classmethod(_get_node_info_v1)
        attributes["GET_BASE_CLASS"] = classmethod(_get_base_class)
        attributes["DESCRIPTION"] = info.get("description", "")
        attributes["EXPERIMENTAL"] = info.get("experimental", False)
        attributes["DEPRECATED"] = info.get("deprecated", False)
        attributes["API_NODE"] = info.get("api_node", False)
        attributes["NOT_IDEMPOTENT"] = info.get("not_idempotent", False)
        attributes["INPUT_IS_LIST"] = info.get("input_is_list", False)


    class_name = f"PyIsolate_{node_name}".replace(" ", "_")
    bases = (_ComfyNodeInternal,) if is_v3 else ()
    stub_cls = type(class_name, bases, attributes)

    if is_v3:
        try:
            stub_cls.VALIDATE_CLASS()
        except Exception as e:
            logger.error("%s VALIDATE_CLASS failed: %s - %s", LOG_PREFIX, node_name, e)

    return stub_cls


def get_class_types_for_extension(
    extension_name: str,
    running_extensions: Dict[str, "ComfyNodeExtension"],
    specs: List[Any],
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
