from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _read(relative):
    return (ROOT / relative).read_text()


def main():
    nodes = _read("comfy_extras/nodes_seedvr.py")
    sd = _read("comfy/sd.py")
    vae = _read("comfy/ldm/seedvr/vae.py")

    required = [
        "SeedVR2PostProcessing",
        'io.Image.Input("decoded")',
        'io.Image.Input("reference")',
        'io.Combo.Input("method", options=["lab", "none"], default="lab")',
        "def _format_seedvr2_encoded_samples",
        "def decode(self, z, tiled_args=None)",
    ]
    for needle in required:
        if needle not in nodes + sd + vae:
            raise SystemExit(f"missing required static token: {needle}")

    forbidden = ["original_image_video", "img_dims"]
    for needle in forbidden:
        if needle in nodes + sd + vae:
            raise SystemExit(f"forbidden hidden-state token remains: {needle}")

    print("PASS seedvr2 windows static verify")


if __name__ == "__main__":
    main()
