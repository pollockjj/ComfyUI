from pathlib import Path

import pytest


def _write_pyproject(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def test_load_host_policy_reads_values(tmp_path):
    from comfy.isolation.host_policy import load_host_policy

    _write_pyproject(
        tmp_path / "pyproject.toml",
        """
[tool.comfy.host]
sandbox_mode = "disabled"
allow_network = true
writable_paths = ["/tmp/a", "/tmp/b"]
readonly_paths = ["/opt/readonly"]

[tool.comfy.host.whitelist]
ExampleNode = "*"
""".strip(),
    )

    policy = load_host_policy(tmp_path)
    assert policy["sandbox_mode"] == "disabled"
    assert policy["allow_network"] is True
    assert policy["writable_paths"] == ["/tmp/a", "/tmp/b"]
    assert policy["readonly_paths"] == ["/opt/readonly"]
    assert policy["whitelist"] == {"ExampleNode": "*"}


def test_load_host_policy_ignores_invalid_whitelist_type(tmp_path):
    from comfy.isolation.host_policy import DEFAULT_POLICY, load_host_policy

    _write_pyproject(
        tmp_path / "pyproject.toml",
        """
[tool.comfy.host]
allow_network = true
whitelist = ["bad"]
""".strip(),
    )

    policy = load_host_policy(tmp_path)
    assert policy["allow_network"] is True
    assert policy["whitelist"] == DEFAULT_POLICY["whitelist"]


def test_load_host_policy_ignores_invalid_sandbox_mode(tmp_path):
    from comfy.isolation.host_policy import DEFAULT_POLICY, load_host_policy

    _write_pyproject(
        tmp_path / "pyproject.toml",
        """
[tool.comfy.host]
sandbox_mode = "surprise"
""".strip(),
    )

    policy = load_host_policy(tmp_path)

    assert policy["sandbox_mode"] == DEFAULT_POLICY["sandbox_mode"]


def test_disallows_host_tmp_default_or_override_config(tmp_path):
    from comfy.isolation.host_policy import load_host_policy

    _write_pyproject(
        tmp_path / "pyproject.toml",
        """
[tool.comfy.host]
writable_paths = ["/dev/shm", "/tmp", "/tmp/", "/work/cache"]
""".strip(),
    )

    policy = load_host_policy(tmp_path)

    assert policy["writable_paths"] == ["/dev/shm", "/work/cache"]


def test_sealed_worker_ro_import_paths_rejects_non_list_or_relative(tmp_path):
    from comfy.isolation.host_policy import load_host_policy

    _write_pyproject(
        tmp_path / "pyproject.toml",
        """
[tool.comfy.host]
sealed_worker_ro_import_paths = "/opt/comfyui"
""".strip(),
    )
    with pytest.raises(ValueError, match="must be a list of absolute paths"):
        load_host_policy(tmp_path)

    _write_pyproject(
        tmp_path / "pyproject.toml",
        """
[tool.comfy.host]
sealed_worker_ro_import_paths = ["relative/path"]
""".strip(),
    )
    with pytest.raises(ValueError, match="entries must be absolute paths"):
        load_host_policy(tmp_path)


def test_host_policy_path_override_controls_ro_import_paths(tmp_path, monkeypatch):
    from comfy.isolation.host_policy import load_host_policy

    _write_pyproject(
        tmp_path / "pyproject.toml",
        """
[tool.comfy.host]
sealed_worker_ro_import_paths = ["/ignored/base/path"]
""".strip(),
    )
    override_path = tmp_path / "host_policy_override.toml"
    _write_pyproject(
        override_path,
        """
[tool.comfy.host]
sealed_worker_ro_import_paths = ["/override/ro/path"]
""".strip(),
    )
    monkeypatch.setenv("COMFY_HOST_POLICY_PATH", str(override_path))

    policy = load_host_policy(tmp_path)
    assert policy["sealed_worker_ro_import_paths"] == ["/override/ro/path"]
