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
in a `finally` block. The `nodes_seedvr` attribute on the `comfy_extras`
package object and the `model_management` attribute on the `comfy`
package object are restored to their pre-test values via a sentinel:
the original value is captured before the `patch.dict` block, and in
`finally` the attribute is set back to the original if it had been set,
or deleted if it had not. This prevents the test from clobbering a
real `comfy.model_management` (or `comfy_extras.nodes_seedvr`)
attribute that another test may have legitimately imported earlier in
the same pytest process, while still preventing the test's mock from
leaking into later tests that import the real
`comfy_extras.nodes_seedvr`."""

import importlib
import inspect
import sys
from unittest.mock import MagicMock, patch


def test_seedvr_node_signature_matches_schema():
    mock_model_management = MagicMock()
    sentinel = object()

    comfy_module_pre = sys.modules.get("comfy")
    comfy_extras_module_pre = sys.modules.get("comfy_extras")
    prior_comfy_mm_attr = (
        getattr(comfy_module_pre, "model_management", sentinel)
        if comfy_module_pre is not None
        else sentinel
    )
    prior_comfy_extras_seedvr_attr = (
        getattr(comfy_extras_module_pre, "nodes_seedvr", sentinel)
        if comfy_extras_module_pre is not None
        else sentinel
    )

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
            comfy_extras_module = sys.modules.get("comfy_extras")
            if comfy_extras_module is not None:
                if prior_comfy_extras_seedvr_attr is sentinel:
                    if hasattr(comfy_extras_module, "nodes_seedvr"):
                        delattr(comfy_extras_module, "nodes_seedvr")
                else:
                    setattr(comfy_extras_module, "nodes_seedvr", prior_comfy_extras_seedvr_attr)
            comfy_module = sys.modules.get("comfy")
            if comfy_module is not None:
                if prior_comfy_mm_attr is sentinel:
                    if hasattr(comfy_module, "model_management"):
                        delattr(comfy_module, "model_management")
                else:
                    setattr(comfy_module, "model_management", prior_comfy_mm_attr)
