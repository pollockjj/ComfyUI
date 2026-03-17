from __future__ import annotations

import shutil
from pathlib import Path


COMFYUI_ROOT = Path(__file__).resolve().parents[2]
PROBE_SOURCE_ROOT = COMFYUI_ROOT / "tests" / "isolation" / "internal_probe_node"
TARGET_ROOT = COMFYUI_ROOT / "custom_nodes" / "InternalIsolationProbeNode"

PYPROJECT_CONTENT = """[project]
name = "InternalIsolationProbeNode"
version = "0.0.1"

[tool.comfy.isolation]
can_isolate = true
share_torch = true
"""


def stage_probe_node() -> Path:
    if not PROBE_SOURCE_ROOT.is_dir():
        raise RuntimeError(f"Missing probe source directory: {PROBE_SOURCE_ROOT}")

    TARGET_ROOT.mkdir(parents=True, exist_ok=True)
    for source_path in PROBE_SOURCE_ROOT.iterdir():
        destination_path = TARGET_ROOT / source_path.name
        if source_path.is_dir():
            shutil.copytree(source_path, destination_path, dirs_exist_ok=True)
        else:
            shutil.copy2(source_path, destination_path)

    (TARGET_ROOT / "pyproject.toml").write_text(PYPROJECT_CONTENT, encoding="utf-8")
    return TARGET_ROOT


if __name__ == "__main__":
    staged = stage_probe_node()
    print(staged)
