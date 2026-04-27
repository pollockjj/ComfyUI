"""Regression test for SeedVR2InputProcessing schema/signature drift."""

import importlib
import inspect
import sys
from unittest.mock import MagicMock, patch


def test_seedvr_node_signature_matches_schema():
    mock_model_management = MagicMock()
    nodes_seedvr = None
    with patch.dict(sys.modules, {"comfy.model_management": mock_model_management}):
        sys.modules.pop("comfy_extras.nodes_seedvr", None)
        try:
            nodes_seedvr = importlib.import_module("comfy_extras.nodes_seedvr")
            schema_ids = [
                i.id for i in nodes_seedvr.SeedVR2InputProcessing.define_schema().inputs
            ]
            exec_params = [
                p
                for p in inspect.signature(
                    nodes_seedvr.SeedVR2InputProcessing.execute
                ).parameters.keys()
                if p != "cls"
            ]
            assert schema_ids == exec_params, (
                f"SeedVR2InputProcessing schema input ids do not match execute() "
                f"parameter order: schema_ids={schema_ids}, exec_params={exec_params}"
            )
        finally:
            sys.modules.pop("comfy_extras.nodes_seedvr", None)
            comfy_extras_module = sys.modules.get("comfy_extras")
            if (
                comfy_extras_module is not None
                and getattr(comfy_extras_module, "nodes_seedvr", None) is nodes_seedvr
            ):
                delattr(comfy_extras_module, "nodes_seedvr")
            comfy_module = sys.modules.get("comfy")
            if (
                comfy_module is not None
                and getattr(comfy_module, "model_management", None) is mock_model_management
            ):
                delattr(comfy_module, "model_management")

    assert "comfy_extras.nodes_seedvr" not in sys.modules
    comfy_extras_module = sys.modules.get("comfy_extras")
    assert comfy_extras_module is None or getattr(
        comfy_extras_module, "nodes_seedvr", None
    ) is not nodes_seedvr
