from __future__ import annotations

from tests.isolation.singleton_boundary_helpers import capture_exact_proxy_mode_parity


def test_torch_share_exact_parity() -> None:
    payload = capture_exact_proxy_mode_parity()

    torch_share = payload["rows"]["sealed_worker share_torch=true"]
    assert torch_share["config_valid"] is True
    assert payload["missing_or_reduced"]["sealed_worker share_torch=true"] == []
    assert set(torch_share["entries"].keys()) == set(payload["inventory"])


def test_host_coupled_exact_parity() -> None:
    payload = capture_exact_proxy_mode_parity()

    host_coupled = payload["rows"]["host_coupled"]
    sealed = payload["rows"]["sealed_worker share_torch=false"]

    assert host_coupled["config_valid"] is True
    assert payload["missing_or_reduced"]["host_coupled"] == []
    assert host_coupled["entries"] == sealed["entries"]
