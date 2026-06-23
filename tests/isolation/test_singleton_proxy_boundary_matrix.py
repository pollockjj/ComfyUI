from __future__ import annotations


from tests.isolation.singleton_boundary_helpers import (
    capture_minimal_sealed_worker_imports,
    capture_sealed_singleton_imports,
)


def test_minimal_sealed_worker_forbidden_imports() -> None:
    payload = capture_minimal_sealed_worker_imports()

    assert payload["mode"] == "minimal_sealed_worker"
    assert payload["runtime_probe_function"] == "inspect"
    assert payload["forbidden_matches"] == []


def test_folder_paths_child_safe() -> None:
    payload = capture_sealed_singleton_imports()

    assert payload["mode"] == "sealed_singletons"
    assert payload["folder_path"] == "/sandbox/input/demo.png"
    assert payload["temp_dir"] == "/sandbox/temp"
    assert payload["models_dir"] == "/sandbox/models"
    assert payload["forbidden_matches"] == []

