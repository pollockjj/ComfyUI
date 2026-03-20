from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from comfy.isolation.adapter import ComfyUIAdapter


COMFYUI_ROOT = Path(__file__).resolve().parents[2]
TARGET_FILES = [
    COMFYUI_ROOT / "tests" / "isolation" / "test_model_management_proxy.py",
    COMFYUI_ROOT / "tests" / "isolation" / "test_folder_paths_proxy.py",
    COMFYUI_ROOT / "tests" / "isolation" / "test_init.py",
]


def _collect_torch_share_subset() -> list[str]:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            *[str(path) for path in TARGET_FILES],
            "-k",
            "torch_share_subset",
            "--collect-only",
            "-q",
        ],
        cwd=COMFYUI_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return [
        line.strip()
        for line in result.stdout.splitlines()
        if "::" in line and "torch_share_subset" in line
    ]


def test_torch_share_mode_scope() -> None:
    collected = _collect_torch_share_subset()

    assert any("test_model_management_proxy.py::" in line for line in collected)
    assert any("test_folder_paths_proxy.py::" in line for line in collected)
    assert any("test_init.py::" in line for line in collected)


def test_host_runtime_allowed_only_when_enabled() -> None:
    adapter = ComfyUIAdapter()

    minimal_names = {
        service.__name__
        for service in adapter.provide_rpc_services_for_config(
            {"execution_model": "sealed_worker", "share_torch": False},
            host_side=True,
        )
    }
    torch_share_names = {
        service.__name__
        for service in adapter.provide_rpc_services_for_config(
            {"execution_model": "sealed_worker", "share_torch": True},
            host_side=True,
        )
    }

    assert "ModelManagementProxy" not in minimal_names
    assert "PromptServerService" not in minimal_names
    assert "ModelManagementProxy" in torch_share_names
    assert "PromptServerService" in torch_share_names
