from pathlib import Path


def test_seedvr_vae_tiled_args_uses_get_not_pop():
    path = Path(__file__).resolve().parents[2] / "comfy" / "ldm" / "seedvr" / "vae.py"
    src = path.read_text(encoding="utf-8")
    assert ".tiled_args.pop(" not in src, f"VideoAutoencoderKLWrapper.decode contains tiled_args.pop(...) which mutates self.tiled_args across calls; expected tiled_args.get(...) per the upstream fix in Comfy-Org/ComfyUI#11294 commit 3b418da. Source path: {path}"
    assert ".tiled_args.get(" in src, f"VideoAutoencoderKLWrapper.decode does not read tiled_args via .get(); expected exactly one self.tiled_args.get(\"enable_tiling\", False) call per Slice 1 baseline. Source path: {path}"
