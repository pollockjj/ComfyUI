from pathlib import Path


def _write_pyproject(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def test_load_host_policy_defaults_when_pyproject_missing(tmp_path):
    from comfy.isolation.host_policy import DEFAULT_POLICY, load_host_policy

    policy = load_host_policy(tmp_path)

    assert policy["sandbox_mode"] == DEFAULT_POLICY["sandbox_mode"]
    assert policy["allow_network"] == DEFAULT_POLICY["allow_network"]
    assert policy["writable_paths"] == DEFAULT_POLICY["writable_paths"]
    assert policy["readonly_paths"] == DEFAULT_POLICY["readonly_paths"]
    assert policy["whitelist"] == DEFAULT_POLICY["whitelist"]


def test_load_host_policy_defaults_when_section_missing(tmp_path):
    from comfy.isolation.host_policy import DEFAULT_POLICY, load_host_policy

    _write_pyproject(
        tmp_path / "pyproject.toml",
        """
[project]
name = "ComfyUI"
""".strip(),
    )

    policy = load_host_policy(tmp_path)
    assert policy["sandbox_mode"] == DEFAULT_POLICY["sandbox_mode"]
    assert policy["allow_network"] == DEFAULT_POLICY["allow_network"]
    assert policy["whitelist"] == {}


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


def test_load_host_policy_uses_env_override_path(tmp_path, monkeypatch):
    from comfy.isolation.host_policy import load_host_policy

    override_path = tmp_path / "host_policy_override.toml"
    _write_pyproject(
        override_path,
        """
[tool.comfy.host]
sandbox_mode = "disabled"
allow_network = true
""".strip(),
    )

    monkeypatch.setenv("COMFY_HOST_POLICY_PATH", str(override_path))

    policy = load_host_policy(tmp_path / "missing-root")

    assert policy["sandbox_mode"] == "disabled"
    assert policy["allow_network"] is True
