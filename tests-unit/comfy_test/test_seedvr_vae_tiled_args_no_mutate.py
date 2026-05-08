import re
from pathlib import Path


def test_seedvr_vae_tiled_args_uses_get_not_pop():
    path = Path(__file__).resolve().parents[2] / "comfy" / "ldm" / "seedvr" / "vae.py"
    src = path.read_text(encoding="utf-8")
    assert not re.search(r"(?:self\.)?tiled_args\.pop\s*\(", src), (
        f"VideoAutoencoderKLWrapper.decode contains tiled_args.pop(...) which mutates tiled_args across calls; "
        f"expected reads via .get(...) only. "
        f"Source path: {path}"
    )
    enable_tiling_get_calls = re.findall(
        r"self\.tiled_args\.get\s*\(\s*[\"']enable_tiling[\"']\s*,\s*False\s*\)",
        src,
    )
    assert len(enable_tiling_get_calls) == 1, (
        f"VideoAutoencoderKLWrapper.decode should contain exactly one "
        f"self.tiled_args.get('enable_tiling', False) call per Slice 1 baseline; "
        f"found {len(enable_tiling_get_calls)}. Source path: {path}"
    )
