import ast
import inspect
import textwrap

import torch

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

from comfy_extras import nodes_seedvr  # noqa: E402


def _schema_ids(items):
    return [item.id for item in items]


def test_input_processing_schema_is_preprocess_only():
    schema = nodes_seedvr.SeedVR2InputProcessing.define_schema()

    assert _schema_ids(schema.inputs) == ["images", "shorter_edge"]
    assert _schema_ids(schema.outputs) == ["input_pixels"]
    assert schema.outputs[0].get_io_type() == "IMAGE"


def test_input_processing_does_not_call_encode_decode_or_color_transfer():
    source = inspect.getsource(nodes_seedvr.SeedVR2InputProcessing.execute)
    tree = ast.parse(textwrap.dedent(source))
    forbidden_names = {
        "encode",
        "encode_tiled",
        "decode",
        "decode_tiled",
        "tiled_vae",
        "lab_color_transfer",
    }

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr
            else:
                continue
            assert name not in forbidden_names
