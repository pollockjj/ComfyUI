from __future__ import annotations

from tests.isolation.singleton_boundary_helpers import capture_exact_small_proxy_relay


def _transcripts_for(payload: dict[str, object], object_id: str, method: str) -> list[dict[str, object]]:
    return [
        entry
        for entry in payload["transcripts"]
        if entry["object_id"] == object_id and entry["method"] == method
    ]


def test_folder_paths_exact_relay() -> None:
    payload = capture_exact_small_proxy_relay()

    assert payload["forbidden_matches"] == []
    assert payload["models_dir"] == "/sandbox/models"
    assert payload["folder_path"] == "/sandbox/input/demo.png"

    models_dir_calls = _transcripts_for(payload, "FolderPathsProxy", "rpc_get_models_dir")
    annotated_calls = _transcripts_for(payload, "FolderPathsProxy", "rpc_get_annotated_filepath")

    assert models_dir_calls
    assert annotated_calls
    assert all(entry["phase"] != "child_call" or entry["method"] != "rpc_snapshot" for entry in payload["transcripts"])


def test_progress_exact_relay() -> None:
    payload = capture_exact_small_proxy_relay()

    progress_calls = _transcripts_for(payload, "ProgressProxy", "rpc_set_progress")

    assert progress_calls
    host_targets = [entry["target"] for entry in progress_calls if entry["phase"] == "host_invocation"]
    assert host_targets == ["comfy_execution.progress.get_progress_state().update_progress"]
    result_entries = [entry for entry in progress_calls if entry["phase"] == "result"]
    assert result_entries == [{"phase": "result", "object_id": "ProgressProxy", "method": "rpc_set_progress", "result": None}]


def test_utils_exact_relay() -> None:
    payload = capture_exact_small_proxy_relay()

    utils_calls = _transcripts_for(payload, "UtilsProxy", "progress_bar_hook")

    assert utils_calls
    host_targets = [entry["target"] for entry in utils_calls if entry["phase"] == "host_invocation"]
    assert host_targets == ["comfy.utils.PROGRESS_BAR_HOOK"]
    result_entries = [entry for entry in utils_calls if entry["phase"] == "result"]
    assert result_entries
    assert result_entries[0]["result"]["value"] == 2
    assert result_entries[0]["result"]["total"] == 5


def test_helper_proxy_exact_relay() -> None:
    payload = capture_exact_small_proxy_relay()

    helper_calls = _transcripts_for(payload, "HelperProxiesService", "rpc_restore_input_types")

    assert helper_calls
    host_targets = [entry["target"] for entry in helper_calls if entry["phase"] == "host_invocation"]
    assert host_targets == ["comfy.isolation.proxies.helper_proxies.restore_input_types"]
    assert payload["restored_any_type"] == "*"
