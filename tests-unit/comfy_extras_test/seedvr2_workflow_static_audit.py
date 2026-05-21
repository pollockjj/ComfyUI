import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
GRAPHS = [
    ROOT / "tests/inference/graphs/seedvr2_simple_refactor_api.json",
    ROOT / "tests/inference/graphs/seedvr2_advanced_refactor_api.json",
]
ALLOWED = {
    "CheckpointLoaderSimple",
    "UNETLoader",
    "VAELoader",
    "LoadImage",
    "SeedVR2InputProcessing",
    "VAEEncode",
    "VAEEncodeTiled",
    "SeedVR2Conditioning",
    "KSampler",
    "SeedVR2ProgressiveSampler",
    "VAEDecode",
    "VAEDecodeTiled",
    "SeedVR2PostProcessing",
    "SaveImage",
    "SaveAnimatedWEBP",
}
REQUIRED = {"SeedVR2InputProcessing", "SeedVR2PostProcessing"}


def main():
    for graph in GRAPHS:
        data = json.loads(graph.read_text())
        classes = {node["class_type"] for node in data.values()}
        unexpected = classes - ALLOWED
        missing = REQUIRED - classes
        if unexpected:
            raise SystemExit(f"{graph}: unexpected class types {sorted(unexpected)}")
        if missing:
            raise SystemExit(f"{graph}: missing required class types {sorted(missing)}")
        if not {"VAEEncode", "VAEEncodeTiled"}.intersection(classes):
            raise SystemExit(f"{graph}: missing VAE encode boundary node")
        if not {"VAEDecode", "VAEDecodeTiled"}.intersection(classes):
            raise SystemExit(f"{graph}: missing VAE decode boundary node")
    print("PASS seedvr2 workflow static audit")


if __name__ == "__main__":
    main()
