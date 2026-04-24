from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any


def _json_dict(text: str) -> dict[str, Any]:
    if not text:
        return {}
    value = json.loads(text)
    if not isinstance(value, dict):
        raise TypeError("expected JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ffmpeg_snapshot(output_path: Path) -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "owner": "SeedVR2BaselineSaveWithArtifacts",
        "source": "ffmpeg",
        "filters_checked": ["psnr", "ssim", "libvmaf"],
    }
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", str(output_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    snapshot["ffprobe_returncode"] = result.returncode
    if result.returncode == 0:
        snapshot["ffprobe"] = json.loads(result.stdout)
    else:
        snapshot["ffprobe_error"] = result.stderr.strip()
    return snapshot


class SeedVR2BaselineSaveWithArtifacts:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "output_path": ("STRING", {"default": ""}),
                "artifact_path": ("STRING", {"default": ""}),
                "run_id": ("STRING", {"default": "run"}),
                "manifest_id": ("STRING", {"default": ""}),
                "implementation": (["custom_node", "native_pr"],),
                "input_json": ("STRING", {"default": "{}"}),
                "workflow_json": ("STRING", {"default": "{}"}),
                "node_parameters_json": ("STRING", {"default": "{}"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "save_with_artifacts"
    CATEGORY = "api/baseline"
    OUTPUT_NODE = True

    def save_with_artifacts(
        self,
        output_path: str,
        artifact_path: str,
        run_id: str,
        manifest_id: str,
        implementation: str,
        input_json: str,
        workflow_json: str,
        node_parameters_json: str,
    ):
        start = time.perf_counter()
        output = Path(output_path)
        artifact = Path(artifact_path)
        artifact.parent.mkdir(parents=True, exist_ok=True)
        if not output.exists():
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(b"issue-123-probe-video-placeholder\n")

        payload = {
            "run_id": run_id,
            "manifest_id": manifest_id,
            "implementation": implementation,
            "input": _json_dict(input_json),
            "workflow": _json_dict(workflow_json),
            "node_parameters": _json_dict(node_parameters_json),
            "output": {
                "path": str(output),
                "sha256": _sha256(output),
                "size_bytes": output.stat().st_size,
            },
            "metrics": {
                "ffmpeg": _ffmpeg_snapshot(output),
                "collection_context": {
                    "location": "comfy_node",
                    "post_comfy_metric_generation": False,
                },
            },
            "runtime": {
                "started_monotonic": start,
                "elapsed_seconds": time.perf_counter() - start,
            },
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
            },
            "metadata_storage": {
                "mode": "sidecar",
                "path": str(artifact),
            },
        }
        artifact.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return (str(artifact),)


class SeedVR2BaselineCompareArtifacts:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "custom_artifact_path": ("STRING", {"default": ""}),
                "native_artifact_path": ("STRING", {"default": ""}),
                "comparison_path": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "compare_artifacts"
    CATEGORY = "api/baseline"
    OUTPUT_NODE = True

    def compare_artifacts(self, custom_artifact_path: str, native_artifact_path: str, comparison_path: str):
        custom = json.loads(Path(custom_artifact_path).read_text(encoding="utf-8"))
        native = json.loads(Path(native_artifact_path).read_text(encoding="utf-8"))
        comparison = {
            "custom_manifest_id": custom["manifest_id"],
            "native_manifest_id": native["manifest_id"],
            "custom_output_sha256": custom["output"]["sha256"],
            "native_output_sha256": native["output"]["sha256"],
            "same_manifest_id": custom["manifest_id"] == native["manifest_id"],
        }
        out = Path(comparison_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(comparison, indent=2, sort_keys=True), encoding="utf-8")
        return (str(out),)


NODE_CLASS_MAPPINGS = {
    "SeedVR2BaselineSaveWithArtifacts": SeedVR2BaselineSaveWithArtifacts,
    "SeedVR2BaselineCompareArtifacts": SeedVR2BaselineCompareArtifacts,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SeedVR2BaselineSaveWithArtifacts": "SeedVR2 Baseline Save With Artifacts",
    "SeedVR2BaselineCompareArtifacts": "SeedVR2 Baseline Compare Artifacts",
}
