"""Tests for conda config parsing in extension_loader.py (Slice 5).

These tests verify that extension_loader.py correctly parses conda-related
fields from pyproject.toml manifests and passes them into the extension config
dict given to pyisolate. The torch import chain is broken by pre-mocking
extension_wrapper before importing extension_loader.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ── Break the import chain ────────────────────────────────────────────
# comfy.isolation.__init__ imports runtime_helpers → comfy_api → av (missing),
# and extension_wrapper → torch (docstring collision). We create a bare
# namespace module for comfy.isolation so __init__.py never executes, then
# mock the problematic submodules before importing extension_loader.
import types

_mock_wrapper = MagicMock()
_mock_wrapper.ComfyNodeExtension = type("ComfyNodeExtension", (), {})

# Create comfy.isolation as bare namespace (skip __init__.py)
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

# Mock folder_paths (ComfyUI internal)
_mock_fp = MagicMock()
_mock_fp.base_path = "/fake/comfyui"
sys.modules.setdefault("folder_paths", _mock_fp)

# Now safe to import extension_loader
from comfy.isolation.extension_loader import load_isolated_node  # noqa: E402


def _make_manifest(
    *,
    package_manager: str = "uv",
    conda_channels: list[str] | None = None,
    conda_dependencies: list[str] | None = None,
    conda_platforms: list[str] | None = None,
    share_torch: bool = False,
    can_isolate: bool = True,
    dependencies: list[str] | None = None,
    cuda_wheels: list[str] | None = None,
) -> dict:
    """Build a manifest dict matching tomllib.load() output."""
    isolation: dict = {"can_isolate": can_isolate}
    if package_manager != "uv":
        isolation["package_manager"] = package_manager
    if conda_channels is not None:
        isolation["conda_channels"] = conda_channels
    if conda_dependencies is not None:
        isolation["conda_dependencies"] = conda_dependencies
    if conda_platforms is not None:
        isolation["conda_platforms"] = conda_platforms
    if share_torch:
        isolation["share_torch"] = True
    if cuda_wheels is not None:
        isolation["cuda_wheels"] = cuda_wheels

    return {
        "project": {
            "name": "test-extension",
            "dependencies": dependencies or ["numpy"],
        },
        "tool": {"comfy": {"isolation": isolation}},
    }


@pytest.fixture
def manifest_file(tmp_path):
    """Create a dummy pyproject.toml so manifest_path.open('rb') succeeds."""
    path = tmp_path / "pyproject.toml"
    path.write_bytes(b"")  # content is overridden by tomllib mock
    return path


@pytest.fixture
def mock_pyisolate():
    """Mock pyisolate to avoid real venv creation."""
    mock_ext = AsyncMock()
    mock_ext.list_nodes = AsyncMock(return_value={})

    mock_manager = MagicMock()
    mock_manager.load_extension = MagicMock(return_value=mock_ext)

    with patch("comfy.isolation.extension_loader.pyisolate") as mock_pi:
        mock_pi.ExtensionManager = MagicMock(return_value=mock_manager)
        yield mock_pi, mock_manager, mock_ext


class TestCondaPackageManagerParsing:
    """Verify extension_loader.py parses conda config from pyproject.toml."""

    @pytest.mark.asyncio
    async def test_conda_package_manager_in_config(
        self, mock_pyisolate, manifest_file, tmp_path
    ):
        """package_manager='conda' must appear in extension_config."""

        manifest = _make_manifest(
            package_manager="conda",
            conda_channels=["conda-forge"],
            conda_dependencies=["eccodes"],
        )

        _, mock_manager, _ = mock_pyisolate

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

        config = mock_manager.load_extension.call_args[0][0]
        assert config["package_manager"] == "conda"

    @pytest.mark.asyncio
    async def test_conda_channels_in_config(
        self, mock_pyisolate, manifest_file, tmp_path
    ):
        """conda_channels must be passed through to extension_config."""

        manifest = _make_manifest(
            package_manager="conda",
            conda_channels=["conda-forge", "nvidia"],
            conda_dependencies=["eccodes"],
        )

        _, mock_manager, _ = mock_pyisolate

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

        config = mock_manager.load_extension.call_args[0][0]
        assert config["conda_channels"] == ["conda-forge", "nvidia"]

    @pytest.mark.asyncio
    async def test_conda_dependencies_in_config(
        self, mock_pyisolate, manifest_file, tmp_path
    ):
        """conda_dependencies must be passed through to extension_config."""

        manifest = _make_manifest(
            package_manager="conda",
            conda_channels=["conda-forge"],
            conda_dependencies=["eccodes", "cfgrib"],
        )

        _, mock_manager, _ = mock_pyisolate

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

        config = mock_manager.load_extension.call_args[0][0]
        assert config["conda_dependencies"] == ["eccodes", "cfgrib"]

    @pytest.mark.asyncio
    async def test_conda_platforms_in_config(
        self, mock_pyisolate, manifest_file, tmp_path
    ):
        """conda_platforms must be passed through to extension_config."""

        manifest = _make_manifest(
            package_manager="conda",
            conda_channels=["conda-forge"],
            conda_dependencies=["eccodes"],
            conda_platforms=["linux-64"],
        )

        _, mock_manager, _ = mock_pyisolate

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

        config = mock_manager.load_extension.call_args[0][0]
        assert config["conda_platforms"] == ["linux-64"]


class TestCondaForcedOverrides:
    """Verify conda forces share_torch=False, share_cuda_ipc=False."""

    @pytest.mark.asyncio
    async def test_conda_forces_share_torch_false(
        self, mock_pyisolate, manifest_file, tmp_path
    ):
        """share_torch must be forced False for conda, even if manifest says True."""

        manifest = _make_manifest(
            package_manager="conda",
            conda_channels=["conda-forge"],
            conda_dependencies=["eccodes"],
            share_torch=True,  # manifest requests True — must be overridden
        )

        _, mock_manager, _ = mock_pyisolate

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

        config = mock_manager.load_extension.call_args[0][0]
        assert config["share_torch"] is False

    @pytest.mark.asyncio
    async def test_conda_forces_share_cuda_ipc_false(
        self, mock_pyisolate, manifest_file, tmp_path
    ):
        """share_cuda_ipc must be forced False for conda."""

        manifest = _make_manifest(
            package_manager="conda",
            conda_channels=["conda-forge"],
            conda_dependencies=["eccodes"],
            share_torch=True,
        )

        _, mock_manager, _ = mock_pyisolate

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

        config = mock_manager.load_extension.call_args[0][0]
        assert config["share_cuda_ipc"] is False

    @pytest.mark.asyncio
    async def test_conda_skips_sandbox_config(
        self, mock_pyisolate, manifest_file, tmp_path
    ):
        """sandbox config must be empty for conda (no bwrap)."""

        manifest = _make_manifest(
            package_manager="conda",
            conda_channels=["conda-forge"],
            conda_dependencies=["eccodes"],
        )

        _, mock_manager, _ = mock_pyisolate

        with (
            patch("comfy.isolation.extension_loader.tomllib") as mock_tomllib,
            patch(
                "comfy.isolation.extension_loader.platform.system",
                return_value="Linux",
            ),
        ):
            mock_tomllib.load.return_value = manifest
            await load_isolated_node(
                node_dir=tmp_path,
                manifest_path=manifest_file,
                logger=MagicMock(),
                build_stub_class=MagicMock(),
                venv_root=tmp_path / "venvs",
                extension_managers=[],
            )

        config = mock_manager.load_extension.call_args[0][0]
        assert config.get("sandbox") == {} or "sandbox" not in config


class TestUvUnchanged:
    """Verify uv extensions are NOT affected by conda changes."""

    @pytest.mark.asyncio
    async def test_uv_default_no_conda_keys(
        self, mock_pyisolate, manifest_file, tmp_path
    ):
        """Default uv extension must NOT have package_manager or conda keys."""

        manifest = _make_manifest()  # defaults: uv, no conda fields

        _, mock_manager, _ = mock_pyisolate

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

        config = mock_manager.load_extension.call_args[0][0]
        # uv extensions should not have conda-specific keys
        assert config.get("package_manager", "uv") == "uv"
        assert "conda_channels" not in config
        assert "conda_dependencies" not in config
