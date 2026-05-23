import json
from pathlib import Path

import pytest
import torch

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

from comfy_extras import nodes_seedvr  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
GRAPHS = [
    ROOT / "tests/inference/graphs/seedvr2_simple_refactor_api.json",
    ROOT / "tests/inference/graphs/seedvr2_advanced_refactor_api.json",
]
ALLOWED = {
    "CheckpointLoaderSimple",
    "CreateVideo",
    "GetVideoComponents",
    "LoadImage",
    "LoadVideo",
    "SaveAnimatedWEBP",
    "SaveImage",
    "SaveVideo",
    "SeedVR2Conditioning",
    "SeedVR2ResizeAndPad",
    "SeedVR2PostProcessing",
    "KSampler",
    "SeedVR2ProgressiveSampler",
    "UNETLoader",
    "VAEDecode",
    "VAEDecodeTiled",
    "VAEEncode",
    "VAEEncodeTiled",
    "VAELoader",
}
REQUIRED = {"SeedVR2ResizeAndPad", "SeedVR2PostProcessing"}
SEEDVR2_SCHEMAS = {
    "SeedVR2Conditioning": nodes_seedvr.SeedVR2Conditioning,
    "SeedVR2ResizeAndPad": nodes_seedvr.SeedVR2ResizeAndPad,
    "SeedVR2PostProcessing": nodes_seedvr.SeedVR2PostProcessing,
    "SeedVR2ProgressiveSampler": nodes_seedvr.SeedVR2ProgressiveSampler,
}


def _schema_input_ids(node_cls):
    return {item.id for item in node_cls.define_schema().inputs}


def test_seedvr2_workflow_graphs_use_native_boundary_nodes():
    for graph in GRAPHS:
        data = json.loads(graph.read_text())
        classes = {node["class_type"] for node in data.values()}
        unexpected = classes - ALLOWED
        missing = REQUIRED - classes
        if unexpected:
            pytest.fail(f"{graph}: unexpected class types {sorted(unexpected)}")
        if missing:
            pytest.fail(f"{graph}: missing required class types {sorted(missing)}")
        if not {"VAEEncode", "VAEEncodeTiled"}.intersection(classes):
            pytest.fail(f"{graph}: missing VAE encode boundary node")
        if not {"VAEDecode", "VAEDecodeTiled"}.intersection(classes):
            pytest.fail(f"{graph}: missing VAE decode boundary node")


def test_seedvr2_workflow_graphs_match_seedvr2_node_input_schemas():
    for graph in GRAPHS:
        data = json.loads(graph.read_text())
        for node_id, node in data.items():
            node_cls = SEEDVR2_SCHEMAS.get(node["class_type"])
            if node_cls is None:
                continue
            allowed_inputs = _schema_input_ids(node_cls)
            actual_inputs = set(node["inputs"])
            extra_inputs = actual_inputs - allowed_inputs
            if extra_inputs:
                pytest.fail(
                    f"{graph} node {node_id} {node['class_type']}: "
                    f"inputs absent from schema {sorted(extra_inputs)}"
                )


def test_seedvr2_workflow_graphs_route_model_through_conditioning():
    for graph in GRAPHS:
        data = json.loads(graph.read_text())
        conditioning_ids = {
            node_id
            for node_id, node in data.items()
            if node["class_type"] == "SeedVR2Conditioning"
        }
        sampler_nodes = [
            (node_id, node)
            for node_id, node in data.items()
            if node["class_type"] == "SeedVR2ProgressiveSampler"
        ]
        for node_id, node in sampler_nodes:
            model_input = node["inputs"]["model"]
            assert model_input[0] in conditioning_ids
            assert model_input[1] == 0, (
                f"{graph} node {node_id}: sampler model input must use the "
                f"SeedVR2Conditioning model passthrough output"
            )
            assert node["inputs"]["positive"] == [model_input[0], 1]
            assert node["inputs"]["negative"] == [model_input[0], 2]
            assert node["inputs"]["latent_image"] == [model_input[0], 3]


def test_seedvr2_workflow_graphs_route_original_image_to_post_processing():
    for graph in GRAPHS:
        data = json.loads(graph.read_text())
        resize_and_pad_nodes = [
            (node_id, node)
            for node_id, node in data.items()
            if node["class_type"] == "SeedVR2ResizeAndPad"
        ]
        post_processing_nodes = [
            (node_id, node)
            for node_id, node in data.items()
            if node["class_type"] == "SeedVR2PostProcessing"
        ]
        assert len(resize_and_pad_nodes) == 1
        _, resize_and_pad = resize_and_pad_nodes[0]
        for node_id, node in post_processing_nodes:
            assert node["inputs"]["original_image"] == [resize_and_pad_nodes[0][0], 1], (
                f"{graph} node {node_id}: post-processing must receive the "
                "SeedVR2ResizeAndPad original_image output"
            )
            assert node["inputs"]["upscaled_shorter_edge"] == [resize_and_pad_nodes[0][0], 2], (
                f"{graph} node {node_id}: post-processing must use the same "
                "SeedVR2ResizeAndPad upscaled_shorter_edge output"
            )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
