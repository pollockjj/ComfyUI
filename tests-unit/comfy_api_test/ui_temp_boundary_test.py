import os
from pathlib import Path

import torch

from comfy_api.latest import _ui
from comfy_api.latest._io import FolderType
from pyisolate._internal.sandbox import build_bwrap_command
from pyisolate._internal.sandbox_detect import RestrictionModel


def _assert_no_host_tmp_bind():
    cmd = build_bwrap_command(
        python_exe="/home/johnj/ComfyUI/.venv/bin/python",
        module_path="/home/johnj/ComfyUI/custom_nodes/ComfyUI-IsolationToolkit",
        venv_path="/home/johnj/ComfyUI/.venv",
        uds_address="/dev/shm/ui-temp-boundary.sock",
        allow_gpu=False,
        restriction_model=RestrictionModel.NONE,
        sandbox_config={"writable_paths": ["/dev/shm"], "readonly_paths": [], "network": False},
        adapter=None,
    )
    assert ["--bind", "/tmp", "/tmp"] not in [cmd[i : i + 3] for i in range(len(cmd) - 2)]


def test_preview_image_preview_audio_host_contract(monkeypatch):
    _assert_no_host_tmp_bind()
    monkeypatch.setenv("PYISOLATE_CHILD", "1")
    captured_folder_types = []

    def fake_save_images(images, filename_prefix, folder_type, cls, compress_level=4):
        captured_folder_types.append(folder_type)
        return [_ui.SavedResult("preview.png", "", folder_type)]

    def fake_save_audio(audio, filename_prefix, folder_type, cls, format="flac", quality="128k"):
        captured_folder_types.append(folder_type)
        return [_ui.SavedResult("preview.flac", "", folder_type)]

    monkeypatch.setattr(_ui.ImageSaveHelper, "save_images", staticmethod(fake_save_images))
    monkeypatch.setattr(_ui.AudioSaveHelper, "save_audio", staticmethod(fake_save_audio))

    preview_image = _ui.PreviewImage(torch.zeros((1, 2, 2, 3)))
    preview_audio = _ui.PreviewAudio({"waveform": torch.zeros((1, 8)), "sample_rate": 44100})

    assert captured_folder_types == [FolderType.output, FolderType.output]
    assert all(result["type"] == FolderType.output.value for result in preview_image.values)
    assert all(result["type"] == FolderType.output.value for result in preview_audio.values)


def test_preview_ui3d_background_transfer_without_host_tmp_bind(monkeypatch, tmp_path):
    _assert_no_host_tmp_bind()
    monkeypatch.setenv("PYISOLATE_CHILD", "1")

    host_tmp_sentinel = Path("/tmp/ui3d_host_sentinel.txt")
    host_tmp_sentinel.write_text("host-only", encoding="utf-8")
    monkeypatch.setattr(_ui.folder_paths, "get_output_directory", lambda: str(tmp_path))
    monkeypatch.setattr(_ui.folder_paths, "get_temp_directory", lambda: "/tmp/comfyui_temp")

    preview = _ui.PreviewUI3D(
        "model.glb",
        {"camera": "orbit"},
        bg_image=torch.zeros((1, 2, 2, 3)),
    )

    assert preview.bg_image_path is not None
    assert preview.bg_image_path.startswith("output/")
    output_name = preview.bg_image_path.split("/", 1)[1]
    assert (tmp_path / output_name).exists()
    assert host_tmp_sentinel.read_text(encoding="utf-8") == "host-only"
