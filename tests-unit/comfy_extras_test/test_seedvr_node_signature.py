"""Regression test: SeedVR2InputProcessing schema input ids must match
execute() positional parameter order. Drift between the two would silently
swap arguments at runtime; this test fails loudly on any future drift.

The schema input attribute is `.id` (verified live via Python introspection
on the upstream class -- there is no `.name`).

`comfy.model_management` is stubbed via `patch.dict(sys.modules, ...)` for
the import performed inside this test, so importing
`comfy_extras.nodes_seedvr` here does not call
`torch.cuda.is_available()` or trigger other GPU/server-side
initialization through that dependency. Live introspection indicated that
`comfy_extras.nodes_seedvr` pulls in `comfy.model_management`
transitively here (not `nodes`, not `server`).

`comfy_extras.nodes_seedvr` is unconditionally evicted from `sys.modules`
in a `finally` block, and the mocked `model_management` attribute is
removed from the `comfy` package object if it points at the test's mock,
so the stub does not leak into later tests that may import the real
`comfy_extras.nodes_seedvr`."""

import importlib
import inspect
import sys
from unittest.mock import MagicMock, patch


def test_seedvr_node_signature_matches_schema():
    mock_model_management = MagicMock()
    with patch.dict(sys.modules, {"comfy.model_management": mock_model_management}):
        sys.modules.pop("comfy_extras.nodes_seedvr", None)
        try:
            nodes_seedvr = importlib.import_module("comfy_extras.nodes_seedvr")
            schema_ids = [i.id for i in nodes_seedvr.SeedVR2InputProcessing.define_schema().inputs]
            exec_params = [
                p
                for p in inspect.signature(nodes_seedvr.SeedVR2InputProcessing.execute).parameters.keys()
                if p != "cls"
            ]
            assert schema_ids == exec_params, (
                f"SeedVR2InputProcessing schema input ids do not match execute() "
                f"parameter order: schema_ids={schema_ids}, exec_params={exec_params}"
            )
        finally:
            sys.modules.pop("comfy_extras.nodes_seedvr", None)
            comfy_module = sys.modules.get("comfy")
            if comfy_module is not None and getattr(comfy_module, "model_management", None) is mock_model_management:
                delattr(comfy_module, "model_management")
