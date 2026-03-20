from __future__ import annotations

from comfy.isolation.adapter import ComfyUIAdapter


def _service_names(config: dict[str, object], *, host_side: bool = True) -> set[str]:
    adapter = ComfyUIAdapter()
    services = adapter.provide_rpc_services_for_config(config, host_side=host_side)
    return {service.__name__ for service in services}


def test_model_management_host_only() -> None:
    minimal_names = _service_names(
        {"execution_model": "sealed_worker", "share_torch": False}
    )
    assert "ModelManagementProxy" not in minimal_names
    assert "UtilsProxy" in minimal_names
    assert "ProgressProxy" in minimal_names

    torch_share_names = _service_names(
        {"execution_model": "sealed_worker", "share_torch": True}
    )
    assert "ModelManagementProxy" in torch_share_names

    host_coupled_names = _service_names(
        {"execution_model": "host-coupled", "share_torch": False}
    )
    assert "ModelManagementProxy" in host_coupled_names


def test_prompt_server_host_only() -> None:
    minimal_names = _service_names(
        {"execution_model": "sealed_worker", "share_torch": False}
    )
    assert "PromptServerService" not in minimal_names

    torch_share_names = _service_names(
        {"execution_model": "sealed_worker", "share_torch": True}
    )
    assert "PromptServerService" in torch_share_names

    host_coupled_names = _service_names(
        {"execution_model": "host-coupled", "share_torch": False}
    )
    assert "PromptServerService" in host_coupled_names
