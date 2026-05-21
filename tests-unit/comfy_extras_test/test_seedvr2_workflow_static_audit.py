import json
from pathlib import Path

import pytest


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
    "SeedVR2InputProcessing",
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
REQUIRED = {"SeedVR2InputProcessing", "SeedVR2PostProcessing"}


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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
