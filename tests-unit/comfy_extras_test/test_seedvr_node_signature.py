"""Regression test: SeedVR2InputProcessing schema input ids must match
execute() positional parameter order. Drift between the two would silently
swap arguments at runtime; this test fails loudly on any future drift.

The schema input attribute is `.id` (verified live via Python introspection
on the upstream class -- there is no `.name`).

`comfy.model_management` is stubbed via `patch.dict(sys.modules, ...)` so
test collection does not transitively trigger `torch.cuda.is_available()`
or any other GPU/server-side initialization at module-import time. Live
introspection confirmed only `comfy.model_management` is pulled in
transitively by `comfy_extras.nodes_seedvr` (not `nodes`, not `server`)."""

import importlib
import inspect
import sys
from unittest.mock import MagicMock, patch


def test_seedvr_node_signature_matches_schema():
    with patch.dict(sys.modules, {"comfy.model_management": MagicMock()}):
        sys.modules.pop("comfy_extras.nodes_seedvr", None)
        nodes_seedvr = importlib.import_module("comfy_extras.nodes_seedvr")
        schema_ids = [i.id for i in nodes_seedvr.SeedVR2InputProcessing.define_schema().inputs]
        exec_params = [
            p
            for p in inspect.signature(nodes_seedvr.SeedVR2InputProcessing.execute).parameters
            if p != "cls"
        ]
        assert schema_ids == exec_params, (
            f"SeedVR2InputProcessing schema input ids do not match execute() "
            f"parameter order: schema_ids={schema_ids}, exec_params={exec_params}"
        )
