import os
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
pyisolate_root = repo_root.parent / "pyisolate"
if pyisolate_root.exists():
    sys.path.insert(0, str(pyisolate_root))

from comfy.isolation.adapter import ComfyUIAdapter
from pyisolate._internal.serialization_registry import SerializerRegistry


def test_identifier():
    adapter = ComfyUIAdapter()
    assert adapter.identifier == "comfyui"


def test_get_path_config_valid():
    adapter = ComfyUIAdapter()
    path = os.path.join("/opt", "ComfyUI", "custom_nodes", "demo")
    cfg = adapter.get_path_config(path)
    assert cfg is not None
    assert cfg["preferred_root"].endswith("ComfyUI")
    assert "custom_nodes" in cfg["additional_paths"][0]


def test_get_path_config_invalid():
    adapter = ComfyUIAdapter()
    assert adapter.get_path_config("/random/path") is None


def test_provide_rpc_services():
    adapter = ComfyUIAdapter()
    services = adapter.provide_rpc_services()
    names = {s.__name__ for s in services}
    assert "PromptServerProxy" in names
    assert "FolderPathsProxy" in names


def test_register_serializers():
    adapter = ComfyUIAdapter()
    registry = SerializerRegistry.get_instance()
    registry.clear()

    adapter.register_serializers(registry)
    assert registry.has_handler("ModelPatcher")
    assert registry.has_handler("CLIP")
    assert registry.has_handler("VAE")

    registry.clear()
