"""Tests for execution_model parsing and sealed-worker loader selection."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

_mock_wrapper = MagicMock()
_mock_wrapper.ComfyNodeExtension = type("ComfyNodeExtension", (), {})

if "comfy.isolation" not in sys.modules:
    _iso_mod = types.ModuleType("comfy.isolation")
    _iso_mod.__path__ = [  # type: ignore[attr-defined]
        str(Path(__file__).resolve().parent.parent.parent / "comfy" / "isolation")
    ]
    _iso_mod.__package__ = "comfy.isolation"
    sys.modules["comfy.isolation"] = _iso_mod

sys.modules["comfy.isolation.extension_wrapper"] = _mock_wrapper
sys.modules.setdefault("comfy.isolation.runtime_helpers", MagicMock())
sys.modules.setdefault("comfy.isolation.manifest_loader", MagicMock())
sys.modules.setdefault("comfy.isolation.host_policy", MagicMock())

_mock_fp = MagicMock()
_mock_fp.base_path = "/fake/comfyui"
sys.modules.setdefault("folder_paths", _mock_fp)

from comfy.isolation.extension_loader import load_isolated_node  # noqa: E402


def _make_manifest(
    *,
    package_manager: str = "uv",
    execution_model: str | None = None,
    can_isolate: bool = True,
    dependencies: list[str] | None = None,
) -> dict:
    isolation: dict = {"can_isolate": can_isolate}
    if package_manager != "uv":
        isolation["package_manager"] = package_manager
    if execution_model is not None:
        isolation["execution_model"] = execution_model

    return {
        "project": {
            "name": "test-extension",
            "dependencies": dependencies or ["numpy"],
        },
        "tool": {"comfy": {"isolation": isolation}},
    }


@pytest.fixture
def manifest_file(tmp_path):
    path = tmp_path / "pyproject.toml"
    path.write_bytes(b"")
    return path


@pytest.fixture
def mock_pyisolate():
    mock_ext = AsyncMock()
    mock_ext.list_nodes = AsyncMock(return_value={})

    mock_manager = MagicMock()
    mock_manager.load_extension = MagicMock(return_value=mock_ext)
    sealed_type = type("SealedNodeExtension", (), {})

    with patch("comfy.isolation.extension_loader.pyisolate") as mock_pi:
        mock_pi.ExtensionManager = MagicMock(return_value=mock_manager)
        mock_pi.SealedNodeExtension = sealed_type
        yield mock_pi, mock_manager, mock_ext, sealed_type


@pytest.mark.asyncio
async def test_uv_sealed_worker_selects_sealed_extension_type(
    mock_pyisolate, manifest_file, tmp_path
):
    manifest = _make_manifest(execution_model="sealed_worker")

    _, mock_manager, _, sealed_type = mock_pyisolate

    with patch("comfy.isolation.extension_loader.tomllib") as mock_tomllib:
        mock_tomllib.load.return_value = manifest
        await load_isolated_node(
            node_dir=tmp_path,
            manifest_path=manifest_file,
            logger=MagicMock(),
            build_stub_class=MagicMock(),
            venv_root=tmp_path / "venvs",
            extension_managers=[],
        )

    extension_type = sys.modules["comfy.isolation.extension_loader"].pyisolate.ExtensionManager.call_args[0][0]
    config = mock_manager.load_extension.call_args[0][0]
    assert extension_type is sealed_type
    assert config["execution_model"] == "sealed_worker"
    assert config["apis"] == []


@pytest.mark.asyncio
async def test_default_uv_keeps_host_coupled_extension_type(
    mock_pyisolate, manifest_file, tmp_path
):
    manifest = _make_manifest()

    _, mock_manager, _, sealed_type = mock_pyisolate

    with patch("comfy.isolation.extension_loader.tomllib") as mock_tomllib:
        mock_tomllib.load.return_value = manifest
        await load_isolated_node(
            node_dir=tmp_path,
            manifest_path=manifest_file,
            logger=MagicMock(),
            build_stub_class=MagicMock(),
            venv_root=tmp_path / "venvs",
            extension_managers=[],
        )

    extension_type = sys.modules["comfy.isolation.extension_loader"].pyisolate.ExtensionManager.call_args[0][0]
    config = mock_manager.load_extension.call_args[0][0]
    assert extension_type is not sealed_type
    assert "execution_model" not in config


@pytest.mark.asyncio
async def test_conda_without_execution_model_remains_sealed_worker(
    mock_pyisolate, manifest_file, tmp_path
):
    manifest = _make_manifest(package_manager="conda")
    manifest["tool"]["comfy"]["isolation"]["conda_channels"] = ["conda-forge"]
    manifest["tool"]["comfy"]["isolation"]["conda_dependencies"] = ["eccodes"]

    _, mock_manager, _, sealed_type = mock_pyisolate

    with patch("comfy.isolation.extension_loader.tomllib") as mock_tomllib:
        mock_tomllib.load.return_value = manifest
        await load_isolated_node(
            node_dir=tmp_path,
            manifest_path=manifest_file,
            logger=MagicMock(),
            build_stub_class=MagicMock(),
            venv_root=tmp_path / "venvs",
            extension_managers=[],
        )

    extension_type = sys.modules["comfy.isolation.extension_loader"].pyisolate.ExtensionManager.call_args[0][0]
    config = mock_manager.load_extension.call_args[0][0]
    assert extension_type is sealed_type
    assert config["execution_model"] == "sealed_worker"
