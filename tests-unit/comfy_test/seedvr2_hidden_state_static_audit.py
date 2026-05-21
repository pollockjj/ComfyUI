import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FILES = [
    ROOT / "comfy/ldm/seedvr/vae.py",
    ROOT / "comfy/sd.py",
    ROOT / "comfy_extras/nodes_seedvr.py",
]
FORBIDDEN_ATTRS = {"original_image_video", "img_dims"}
FORBIDDEN_KEYS = {
    "sampler_metadata",
    "latent_sidecar_metadata",
    "saved_latent_metadata",
    "workflow_hidden_state",
}


def main():
    for path in FILES:
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_ATTRS:
                raise SystemExit(f"{path}: forbidden VAE object state attr {node.attr}")
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if node.value in FORBIDDEN_ATTRS or node.value in FORBIDDEN_KEYS:
                    raise SystemExit(f"{path}: forbidden hidden-state string {node.value}")
    print("PASS seedvr2 hidden-state static audit")


if __name__ == "__main__":
    main()
