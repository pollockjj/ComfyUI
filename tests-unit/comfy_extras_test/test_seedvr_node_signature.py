"""Regression test: SeedVR2InputProcessing schema input ids must match
execute() positional parameter order. Drift between the two would silently
swap arguments at runtime; this test fails loudly on any future drift.

The schema input attribute is `.id` (verified live via Python introspection
on the upstream class — there is no `.name`)."""

import inspect

from comfy_extras.nodes_seedvr import SeedVR2InputProcessing


def test_seedvr_node_signature_matches_schema():
    schema_ids = [i.id for i in SeedVR2InputProcessing.define_schema().inputs]
    exec_params = [p for p in inspect.signature(SeedVR2InputProcessing.execute).parameters if p != "cls"]
    print("schema_ids:", schema_ids)
    print("exec_params:", exec_params)
    assert schema_ids == exec_params
