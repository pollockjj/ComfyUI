from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from comfy.isolation.proxies.folder_paths_proxy import (
    FOLDER_PATHS_SINGLETON_CONTRACT,
    FolderPathsProxy,
)
from comfy.isolation.proxies.helper_proxies import HELPER_PROXIES_SINGLETON_CONTRACT
from comfy.isolation.proxies.model_management_proxy import (
    MODEL_MANAGEMENT_SINGLETON_CONTRACT,
    ModelManagementProxy,
)
from comfy.isolation.proxies.progress_proxy import PROGRESS_SINGLETON_CONTRACT
from comfy.isolation.proxies.prompt_server_impl import (
    PROMPT_SERVER_SINGLETON_CONTRACT,
    PromptServerStub,
)
from comfy.isolation.proxies.singleton_contract import (
    SingletonProxyContract,
    SingletonProxyContractError,
)
from comfy.isolation.proxies.utils_proxy import UTILS_SINGLETON_CONTRACT
from comfy.isolation.proxies.web_directory_proxy import WEB_DIRECTORY_SINGLETON_CONTRACT


REPO_ROOT = Path(__file__).resolve().parents[2]


class RecordingCaller:
    def __init__(self, result="ok"):
        self.calls = []
        self.result = result

    async def rpc_call(self, method_name, args, kwargs):
        self.calls.append(("rpc_call", method_name, args, kwargs))
        return self.result

    async def rpc_get_temp_directory(self):
        self.calls.append(("rpc_get_temp_directory",))
        return "/isolated/temp"


def public_module_functions(relative_path: str) -> tuple[str, ...]:
    tree = ast.parse((REPO_ROOT / relative_path).read_text(encoding="utf-8"))
    return tuple(
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and not node.name.startswith("_")
    )


def public_class_methods(relative_path: str, class_name: str) -> tuple[str, ...]:
    tree = ast.parse((REPO_ROOT / relative_path).read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return tuple(
                child.name
                for child in node.body
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                and not child.name.startswith("_")
            )
    raise AssertionError(f"{class_name} not found in {relative_path}")


@pytest.mark.parametrize(
    "contract",
    [
        FOLDER_PATHS_SINGLETON_CONTRACT,
        MODEL_MANAGEMENT_SINGLETON_CONTRACT,
        PROMPT_SERVER_SINGLETON_CONTRACT,
        PROGRESS_SINGLETON_CONTRACT,
        UTILS_SINGLETON_CONTRACT,
        HELPER_PROXIES_SINGLETON_CONTRACT,
        WEB_DIRECTORY_SINGLETON_CONTRACT,
    ],
)
def test_singleton_proxy_contracts_are_internally_complete(contract):
    contract.validate()


def test_folder_paths_contract_matches_current_comfy_source():
    assert FOLDER_PATHS_SINGLETON_CONTRACT.target_public_symbols == public_module_functions(
        "folder_paths.py"
    )


def test_model_management_contract_matches_current_comfy_source():
    assert (
        MODEL_MANAGEMENT_SINGLETON_CONTRACT.target_public_symbols
        == public_module_functions("comfy/model_management.py")
    )


def test_prompt_server_contract_matches_current_comfy_source():
    assert PROMPT_SERVER_SINGLETON_CONTRACT.target_public_symbols == public_class_methods(
        "server.py",
        "PromptServer",
    )


def test_drift_error_names_proxy_target_symbol_and_required_action():
    contract = SingletonProxyContract(
        proxy_name="ExampleProxy",
        target_name="example.module",
        target_public_symbols=("covered", "new_symbol"),
        relay_symbols=("covered",),
    )

    with pytest.raises(SingletonProxyContractError) as exc_info:
        contract.validate()

    message = str(exc_info.value)
    assert "ExampleProxy" in message
    assert "example.module" in message
    assert "new_symbol" in message
    assert "relay, custom serialization, or unsupported classification" in message


def test_model_management_install_materializes_concrete_relay_wrappers():
    caller = RecordingCaller(result={"__pyisolate_torch_device__": "cpu"})
    ModelManagementProxy._rpc = caller
    target = SimpleNamespace()

    try:
        ModelManagementProxy().install_into(target)
        result = target.get_torch_device("arg", flag=True)
    finally:
        ModelManagementProxy.clear_rpc()

    assert str(result) == "cpu"
    assert "get_torch_device" in vars(target)
    assert vars(target)["get_torch_device"].__name__ == "get_torch_device"
    assert caller.calls == [
        (
            "rpc_call",
            "get_torch_device",
            {"__pyisolate_tuple__": ["arg"]},
            {"flag": True},
        )
    ]


def test_model_management_device_relay_deserializes_to_child_torch_device():
    import torch

    caller = RecordingCaller(result={"__pyisolate_torch_device__": "cpu"})
    ModelManagementProxy._rpc = caller
    target = SimpleNamespace()

    try:
        ModelManagementProxy().install_into(target)
        result = target.get_torch_device()
    finally:
        ModelManagementProxy.clear_rpc()

    assert result == torch.device("cpu")
    assert torch.empty((1,), device=result).device == torch.device("cpu")


def test_model_management_archive_model_dtypes_stays_child_local():
    class FakeTensor:
        dtype = "fake-dtype"

    class FakeModule:
        def __init__(self):
            self.weight = FakeTensor()
            self.running = FakeTensor()

        def named_parameters(self, recurse=False):
            assert recurse is False
            return (("weight", self.weight),)

        def named_buffers(self, recurse=False):
            assert recurse is False
            return (("running", self.running),)

    class FakeModel:
        def __init__(self):
            self.module = FakeModule()

        def named_modules(self):
            return (("module", self.module),)

    caller = RecordingCaller(result="should-not-be-used")
    ModelManagementProxy._rpc = caller
    target = SimpleNamespace()
    model = FakeModel()

    try:
        ModelManagementProxy().install_into(target)
        result = target.archive_model_dtypes(model)
    finally:
        ModelManagementProxy.clear_rpc()

    assert result is None
    assert model.module.weight_comfy_model_dtype == "fake-dtype"
    assert model.module.running_comfy_model_dtype == "fake-dtype"
    assert caller.calls == []


def test_model_management_module_size_stays_child_local():
    class FakeTensor:
        def __init__(self, nbytes):
            self.nbytes = nbytes

    class FakeModule:
        def state_dict(self):
            return {"weight": FakeTensor(24), "bias": FakeTensor(8)}

    caller = RecordingCaller(result="should-not-be-used")
    ModelManagementProxy._rpc = caller
    target = SimpleNamespace()

    try:
        ModelManagementProxy().install_into(target)
        result = target.module_size(FakeModule())
    finally:
        ModelManagementProxy.clear_rpc()

    assert result == 32
    assert caller.calls == []


def test_folder_paths_install_materializes_custom_and_relay_wrappers(monkeypatch):
    caller = RecordingCaller(result="mapped")
    FolderPathsProxy._rpc = caller
    target = SimpleNamespace()
    monkeypatch.setenv("PYISOLATE_CHILD", "1")

    try:
        FolderPathsProxy().install_into(target)
        relayed = target.map_legacy("loras")
        custom = target.get_temp_directory()
    finally:
        FolderPathsProxy.clear_rpc()

    assert relayed == "mapped"
    assert custom == "/isolated/temp"
    assert "map_legacy" in vars(target)
    assert "get_temp_directory" in vars(target)
    assert vars(target)["map_legacy"].__name__ == "map_legacy"
    assert caller.calls == [
        ("rpc_call", "map_legacy", ("loras",), {}),
        ("rpc_get_temp_directory",),
    ]


def test_prompt_server_unsupported_methods_fail_loudly():
    stub = PromptServerStub()

    with pytest.raises(RuntimeError) as exc_info:
        stub.setup()

    message = str(exc_info.value)
    assert "PromptServerStub.setup" in message
    assert "intentionally unsupported" in message
    assert "relay, custom serialization, or unsupported classification" in message
