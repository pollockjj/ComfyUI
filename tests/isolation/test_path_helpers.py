from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from pyisolate.path_helpers import build_child_sys_path, serialize_host_snapshot


def test_serialize_host_snapshot_includes_expected_keys(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "snapshot.json"
    monkeypatch.setenv("EXTRA_FLAG", "1")
    snapshot = serialize_host_snapshot(output_path=output, extra_env_keys=["EXTRA_FLAG"])

    assert "sys_path" in snapshot
    assert "sys_executable" in snapshot
    assert "sys_prefix" in snapshot
    assert "environment" in snapshot
    assert output.exists()
    assert snapshot["environment"].get("EXTRA_FLAG") == "1"

    persisted = json.loads(output.read_text(encoding="utf-8"))
    assert persisted["sys_path"] == snapshot["sys_path"]


def test_child_import_succeeds_after_path_unification(tmp_path: Path, monkeypatch) -> None:
    host_root = tmp_path / "host"
    utils_pkg = host_root / "utils"
    app_pkg = host_root / "app"
    utils_pkg.mkdir(parents=True)
    app_pkg.mkdir(parents=True)

    (utils_pkg / "__init__.py").write_text("from . import install_util\n", encoding="utf-8")
    (utils_pkg / "install_util.py").write_text("VALUE = 'hello'\n", encoding="utf-8")
    (app_pkg / "__init__.py").write_text("", encoding="utf-8")
    (app_pkg / "frontend_management.py").write_text(
        "from utils import install_util\nVALUE = install_util.VALUE\n",
        encoding="utf-8",
    )

    child_only = tmp_path / "child_only"
    child_only.mkdir()

    target_module = "app.frontend_management"
    for name in [n for n in list(sys.modules) if n.startswith("app") or n.startswith("utils")]:
        sys.modules.pop(name)

    monkeypatch.setattr(sys, "path", [str(child_only)])
    with pytest.raises(ModuleNotFoundError):
        __import__(target_module)

    for name in [n for n in list(sys.modules) if n.startswith("app") or n.startswith("utils")]:
        sys.modules.pop(name)

    unified = build_child_sys_path([], [], preferred_root=str(host_root))
    monkeypatch.setattr(sys, "path", unified)
    module = __import__(target_module, fromlist=["VALUE"])
    assert module.VALUE == "hello"
