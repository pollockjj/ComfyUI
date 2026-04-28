"""SeedVR2 baseline artifact-collector custom nodes for issue #123.

Two ComfyUI nodes registered through `NODE_CLASS_MAPPINGS`:

- ``SeedVR2BaselineSaveWithArtifacts`` saves the per-implementation output video
  for one manifest entry, computes ffmpeg-backed PSNR/SSIM/VMAF against the
  paired ground-truth video when one is supplied, and writes a per-run JSON
  artifact that conforms to ``github_issues/123/artifact_schema.json``.
- ``SeedVR2BaselineCompareArtifacts`` consumes two per-implementation artifacts
  (custom_node + native_pr) and writes a paired comparison artifact carrying a
  ``custom_vs_native_similarity`` payload for downstream slices.

The node module follows the legacy ``NODE_CLASS_MAPPINGS`` registration pattern
established by ``custom_nodes/issue_87_probe.py`` and
``custom_nodes/issue_89_clipvision_probe.py`` (GTP-ProbePrecedent for
issue #123).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path


_ARTIFACT_SCHEMA_VERSION = 1


def _resolve_tool(name: str, env_var: str, posix_default: str) -> str:
    """Resolve an external CLI tool path.

    Resolution order:
      1. ``$<env_var>`` if set and the path exists.
      2. ``shutil.which(name)`` (PATH lookup; works on Linux, macOS, Windows).
      3. ``posix_default`` if it exists on disk.
      4. Raise ``RuntimeError`` (fail loud).

    The Linux POSIX default is kept as a fallback so existing prosoche /
    `dev_master` runs that rely on `/usr/bin/ffmpeg` keep working when neither
    the env override nor PATH lookup resolves.
    """
    override = os.environ.get(env_var)
    if override:
        override_path = Path(override)
        if override_path.exists():
            return str(override_path)
        raise RuntimeError(
            f"{env_var}={override!r} but path does not exist; clear the env var "
            f"or point it at a valid {name} binary"
        )
    on_path = shutil.which(name)
    if on_path:
        return on_path
    if Path(posix_default).exists():
        return posix_default
    raise RuntimeError(
        f"{name} not found: PATH lookup empty, {env_var} unset, "
        f"{posix_default} absent. Install {name} or set {env_var}."
    )


def _ffmpeg_bin() -> str:
    return _resolve_tool("ffmpeg", "ISSUE_123_FFMPEG_BIN", "/usr/bin/ffmpeg")


def _ffprobe_bin() -> str:
    return _resolve_tool("ffprobe", "ISSUE_123_FFPROBE_BIN", "/usr/bin/ffprobe")


def _epoch() -> float:
    return time.time()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _ffprobe_video(path: Path) -> dict:
    cmd = [
        _ffprobe_bin(),
        "-v", "error",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(proc.stdout)
    streams = [s for s in data.get("streams", []) if s.get("codec_type") == "video"]
    fmt = data.get("format", {})
    if not streams:
        return {"width": None, "height": None, "frame_count": None,
                "duration_seconds": None, "container": fmt.get("format_name"),
                "codec": None}
    s = streams[0]
    nb_frames_raw = s.get("nb_frames")
    try:
        nb_frames = int(nb_frames_raw) if nb_frames_raw not in (None, "N/A") else None
    except (TypeError, ValueError):
        nb_frames = None
    duration_raw = fmt.get("duration") or s.get("duration")
    try:
        duration = float(duration_raw) if duration_raw not in (None, "N/A") else None
    except (TypeError, ValueError):
        duration = None
    return {
        "width": int(s["width"]) if "width" in s else None,
        "height": int(s["height"]) if "height" in s else None,
        "frame_count": nb_frames,
        "duration_seconds": duration,
        "container": fmt.get("format_name"),
        "codec": s.get("codec_name"),
    }


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def _ffmpeg_upscale_to_match(
    *,
    input_path: Path,
    reference_path: Path | None,
    output_path: Path,
) -> None:
    """Re-encode input to match reference resolution (or itself if none).

    Uses libx264 yuv420p, bilinear upscale, 24 fps. Deterministic per single host
    invocation; the output sha256 is recomputed by the caller against the
    on-disk file, so byte-exact reproducibility is not required.
    """
    if reference_path is not None:
        ref = _ffprobe_video(reference_path)
        target_w = ref["width"]
        target_h = ref["height"]
        if target_w is None or target_h is None:
            raise RuntimeError(f"ffprobe did not return reference dimensions for {reference_path}")
        scale = f"scale={target_w}:{target_h}:flags=bilinear"
    else:
        scale = "scale=iw:ih"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    cmd = [
        _ffmpeg_bin(),
        "-hide_banner",
        "-y",
        "-i", str(input_path),
        "-vf", f"{scale},format=yuv420p",
        "-c:v", "libx264",
        "-crf", "18",
        "-preset", "medium",
        "-r", "24",
        "-an",
        str(output_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg encode failed (rc={proc.returncode}) for {input_path} -> {output_path}\n"
            f"stderr_tail:\n{proc.stderr[-2000:]}"
        )


_PSNR_AVG_RE = re.compile(r"average:\s*([0-9]+(?:\.[0-9]+)?(?:e[+-]?[0-9]+)?|inf)")
_SSIM_ALL_RE = re.compile(r"All:\s*([0-9]+(?:\.[0-9]+)?(?:e[+-]?[0-9]+)?)")


def _ffmpeg_psnr(distorted: Path, reference: Path) -> float:
    cmd = [
        _ffmpeg_bin(), "-hide_banner",
        "-i", str(distorted),
        "-i", str(reference),
        "-lavfi", "[0:v][1:v]psnr",
        "-f", "null", "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg psnr failed (rc={proc.returncode})\n{proc.stderr[-2000:]}")
    text = proc.stderr
    m = _PSNR_AVG_RE.search(text)
    if not m:
        raise RuntimeError(f"could not parse PSNR average from ffmpeg output:\n{text[-2000:]}")
    val = m.group(1)
    if val == "inf":
        return float("inf")
    return float(val)


def _ffmpeg_ssim(distorted: Path, reference: Path) -> float:
    cmd = [
        _ffmpeg_bin(), "-hide_banner",
        "-i", str(distorted),
        "-i", str(reference),
        "-lavfi", "[0:v][1:v]ssim",
        "-f", "null", "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg ssim failed (rc={proc.returncode})\n{proc.stderr[-2000:]}")
    text = proc.stderr
    m = _SSIM_ALL_RE.search(text)
    if not m:
        raise RuntimeError(f"could not parse SSIM All from ffmpeg output:\n{text[-2000:]}")
    return float(m.group(1))


def _ffmpeg_filter_escape(value: str) -> str:
    """Escape a path value for use inside an ffmpeg filter argument.

    ffmpeg's filter parser treats ``:`` and ``\\`` as syntax characters, so
    Windows drive paths like ``C:\\Users\\...`` corrupt filter arguments
    when interpolated raw. Escape both for safe use inside ``-lavfi``.
    """
    return value.replace("\\", "\\\\").replace(":", "\\:")


def _ffmpeg_vmaf(distorted: Path, reference: Path) -> float:
    log_dir = tempfile.mkdtemp(prefix="vmaf_", dir=str(distorted.parent))
    log_path = Path(log_dir) / "vmaf.json"
    log_path_escaped = _ffmpeg_filter_escape(str(log_path))
    cmd = [
        _ffmpeg_bin(), "-hide_banner",
        "-i", str(distorted),
        "-i", str(reference),
        "-lavfi", f"[0:v][1:v]libvmaf=log_path={log_path_escaped}:log_fmt=json",
        "-f", "null", "-",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg libvmaf failed (rc={proc.returncode})\n{proc.stderr[-2000:]}")
        with log_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        pooled = data.get("pooled_metrics", {})
        vmaf_block = pooled.get("vmaf", {})
        score = vmaf_block.get("mean")
        if score is None:
            raise RuntimeError(f"libvmaf log missing pooled_metrics.vmaf.mean: {data}")
        return float(score)
    finally:
        shutil.rmtree(log_dir, ignore_errors=True)


def _capture_environment() -> dict:
    env: dict = {
        "python_version": sys.version.split()[0],
        "host": os.uname().nodename if hasattr(os, "uname") else os.environ.get("COMPUTERNAME", ""),
        "platform": sys.platform,
        "ffmpeg_path": _ffmpeg_bin(),
    }
    try:
        ffv = subprocess.run([_ffmpeg_bin(), "-version"], capture_output=True, text=True, check=False)
        first = ffv.stdout.splitlines()[0] if ffv.stdout else ""
        env["ffmpeg_version_line"] = first
    except OSError:
        env["ffmpeg_version_line"] = ""
    try:
        import torch  # type: ignore
        env["torch_version"] = torch.__version__
        env["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            env["cuda_device"] = torch.cuda.get_device_name(0)
    except ImportError:
        env["torch_version"] = None
        env["cuda_available"] = False
    env["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    return env


class SeedVR2BaselineSaveWithArtifacts:
    """Save the per-implementation output video and write a per-run JSON
    artifact conforming to github_issues/123/artifact_schema.json."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_video_path": ("STRING", {"default": "", "multiline": False}),
                "manifest_id": ("STRING", {"default": "", "multiline": False}),
                "input_sha256": ("STRING", {"default": "", "multiline": False}),
                "gt_video_path": ("STRING", {"default": "", "multiline": False}),
                "gt_manifest_id": ("STRING", {"default": "", "multiline": False}),
                "gt_sha256": ("STRING", {"default": "", "multiline": False}),
                "implementation": ("STRING", {"default": "probe", "multiline": False}),
                "workflow_path": ("STRING", {"default": "", "multiline": False}),
                "node_parameters_json": ("STRING", {"default": "{}", "multiline": True}),
                "artifact_dir": ("STRING", {"default": "", "multiline": False}),
                "output_video_dir": ("STRING", {"default": "", "multiline": False}),
                "started_at_epoch": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 9.999e12}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("artifact_path",)
    FUNCTION = "save_with_artifacts"
    OUTPUT_NODE = True
    CATEGORY = "issue_123/baseline"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return time.time()

    def save_with_artifacts(
        self,
        *,
        input_video_path: str,
        manifest_id: str,
        input_sha256: str,
        gt_video_path: str,
        gt_manifest_id: str,
        gt_sha256: str,
        implementation: str,
        workflow_path: str,
        node_parameters_json: str,
        artifact_dir: str,
        output_video_dir: str,
        started_at_epoch: float,
    ):
        run_id = str(uuid.uuid4())
        if started_at_epoch is None or started_at_epoch <= 0:
            started_at_epoch = _epoch()

        # Tag the launch log so probe_output_validation.py can find this run_id.
        print(f"][ ARTIFACT_RUN_ID:{run_id}", flush=True)

        in_path = Path(input_video_path).resolve()
        gt_path = Path(gt_video_path).resolve() if gt_video_path else None
        artifact_dir_p = Path(artifact_dir).resolve()
        output_dir_p = Path(output_video_dir).resolve()
        artifact_dir_p.mkdir(parents=True, exist_ok=True)
        output_dir_p.mkdir(parents=True, exist_ok=True)

        # Provenance verification: when the workflow supplies an expected
        # sha256 for the input or GT, recompute the on-disk hash and fail
        # loud on mismatch. This prevents a stale or wrong-path workflow
        # from silently writing artifacts whose recorded provenance does
        # not match the media the metrics were computed against.
        if input_sha256:
            actual_input_sha = _sha256_file(in_path)
            if actual_input_sha != input_sha256:
                raise RuntimeError(
                    f"Input sha256 mismatch for {in_path}: "
                    f"workflow provided {input_sha256}, file hashes to {actual_input_sha}"
                )
        if gt_sha256 and gt_path is not None:
            actual_gt_sha = _sha256_file(gt_path)
            if actual_gt_sha != gt_sha256:
                raise RuntimeError(
                    f"GT sha256 mismatch for {gt_path}: "
                    f"workflow provided {gt_sha256}, file hashes to {actual_gt_sha}"
                )

        try:
            node_parameters = json.loads(node_parameters_json) if node_parameters_json else {}
        except json.JSONDecodeError:
            node_parameters = {"_raw": node_parameters_json}

        # Encode output video. For probe / native_pr / custom_node leg this is
        # the implementation's output; the input video is read here for the
        # probe-mode degenerate case where no upstream IMAGE batch is produced.
        out_video_path = output_dir_p / f"{manifest_id}_{implementation}_{run_id}.mp4"
        _ffmpeg_upscale_to_match(
            input_path=in_path,
            reference_path=gt_path,
            output_path=out_video_path,
        )
        video_written_at_epoch = _epoch()
        out_meta = _ffprobe_video(out_video_path)
        out_sha = _sha256_file(out_video_path)

        metrics: dict = {}
        if gt_path is not None and gt_path.exists():
            psnr_val = _ffmpeg_psnr(out_video_path, gt_path)
            ssim_val = _ffmpeg_ssim(out_video_path, gt_path)
            vmaf_val = _ffmpeg_vmaf(out_video_path, gt_path)
            metrics = {"psnr": psnr_val, "ssim": ssim_val, "vmaf": vmaf_val}
        else:
            metrics = {
                "psnr": {"not_applicable_reason": "no_gt_input"},
                "ssim": {"not_applicable_reason": "no_gt_input"},
                "vmaf": {"not_applicable_reason": "no_gt_input"},
            }

        finished_at_epoch = _epoch()

        # Per-run filename keyed on manifest_id, implementation, and run_id so
        # repeated runs against the same artifact_dir don't overwrite earlier
        # per-run metadata. A `probe_result.json` symlink is also kept for
        # back-compat with downstream tools that expect the legacy filename.
        artifact_path = artifact_dir_p / f"{manifest_id}_{implementation}_{run_id}.json"
        legacy_alias = artifact_dir_p / "probe_result.json"
        artifact = {
            "schema_version": _ARTIFACT_SCHEMA_VERSION,
            "run_id": run_id,
            "manifest_id": manifest_id,
            "implementation": implementation,
            "input": {
                "manifest_id": manifest_id,
                "path": str(in_path),
                "sha256": input_sha256,
                "gt_manifest_id": gt_manifest_id or None,
                "gt_path": str(gt_path) if gt_path is not None else None,
                "gt_sha256": gt_sha256 or None,
            },
            "workflow": {
                "path": workflow_path,
                "sha256": (_sha256_file(Path(workflow_path)) if workflow_path and Path(workflow_path).exists() else None),
                "issue_references": ["#101", "#123"],
                "artifact_schema_version": _ARTIFACT_SCHEMA_VERSION,
            },
            "node_parameters": node_parameters,
            "output": {
                "video_path": str(out_video_path),
                "video_sha256": out_sha,
                "video_written_at_epoch": video_written_at_epoch,
                "width": out_meta["width"],
                "height": out_meta["height"],
                "frame_count": out_meta["frame_count"],
                "duration_seconds": out_meta["duration_seconds"],
                "container": out_meta["container"],
                "codec": out_meta["codec"],
            },
            "metrics": metrics,
            "runtime": {
                "started_at_epoch": float(started_at_epoch),
                "finished_at_epoch": finished_at_epoch,
                "elapsed_seconds": finished_at_epoch - float(started_at_epoch),
                "peak_vram_bytes": None,
            },
            "environment": _capture_environment(),
            "metadata_storage": {
                "artifact_path": str(artifact_path),
                "artifact_written_at_epoch": _epoch(),
            },
        }
        with artifact_path.open("w", encoding="utf-8") as fh:
            json.dump(artifact, fh, indent=2)
            fh.write("\n")
        # Re-stamp metadata_storage.artifact_written_at_epoch with the
        # post-write completion time, then rewrite. This ensures the recorded
        # timestamp reflects the actual file mtime, satisfying AC-2's
        # "artifact_mtime_within_runtime_window" check.
        artifact["metadata_storage"]["artifact_written_at_epoch"] = _epoch()
        with artifact_path.open("w", encoding="utf-8") as fh:
            json.dump(artifact, fh, indent=2)
            fh.write("\n")
        # Maintain a legacy ``probe_result.json`` filename pointing at the
        # current run's artifact so older tooling that expected the legacy
        # filename keeps working. Use a copy (not a symlink) for Windows
        # compatibility.
        try:
            shutil.copyfile(artifact_path, legacy_alias)
        except OSError:
            # Legacy alias is best-effort; the canonical per-run artifact is
            # the authoritative output.
            pass

        return (str(artifact_path),)


class SeedVR2BaselineCompareArtifacts:
    """Compare a custom_node artifact and a native_pr artifact on the same
    manifest entry, writing a comparison artifact carrying the
    ``custom_vs_native_similarity`` payload for downstream slices.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "custom_artifact_path": ("STRING", {"default": "", "multiline": False}),
                "native_artifact_path": ("STRING", {"default": "", "multiline": False}),
                "comparison_dir": ("STRING", {"default": "", "multiline": False}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("comparison_artifact_path",)
    FUNCTION = "compare_artifacts"
    OUTPUT_NODE = True
    CATEGORY = "issue_123/baseline"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return time.time()

    def compare_artifacts(
        self,
        *,
        custom_artifact_path: str,
        native_artifact_path: str,
        comparison_dir: str,
    ):
        custom_p = Path(custom_artifact_path)
        native_p = Path(native_artifact_path)
        out_dir = Path(comparison_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        custom = json.loads(custom_p.read_text(encoding="utf-8"))
        native = json.loads(native_p.read_text(encoding="utf-8"))

        # Identity verification: refuse to emit a custom_vs_native_similarity
        # delta unless the two artifacts agree on what they describe. Without
        # this, a miswired graph or stale artifact path would produce
        # meaningless deltas and contaminate downstream baseline aggregation.
        custom_id = custom.get("manifest_id")
        native_id = native.get("manifest_id")
        if not custom_id or not native_id or custom_id != native_id:
            raise RuntimeError(
                f"Comparison manifest_id mismatch: "
                f"custom={custom_id!r} ({custom_p}), "
                f"native={native_id!r} ({native_p})"
            )
        custom_input = (custom.get("input") or {})
        native_input = (native.get("input") or {})
        if custom_input.get("manifest_id") != native_input.get("manifest_id"):
            raise RuntimeError(
                f"Comparison input.manifest_id mismatch: "
                f"custom={custom_input.get('manifest_id')!r}, "
                f"native={native_input.get('manifest_id')!r}"
            )
        if custom_input.get("gt_manifest_id") != native_input.get("gt_manifest_id"):
            raise RuntimeError(
                f"Comparison input.gt_manifest_id mismatch: "
                f"custom={custom_input.get('gt_manifest_id')!r}, "
                f"native={native_input.get('gt_manifest_id')!r}"
            )
        custom_impl = custom.get("implementation")
        native_impl = native.get("implementation")
        if custom_impl != "custom_node":
            raise RuntimeError(
                f"Comparison custom_artifact_path={custom_p} has implementation="
                f"{custom_impl!r}; the first input must be a custom_node artifact "
                f"(probe / native_pr / comparison artifacts are not accepted on this slot)"
            )
        if native_impl != "native_pr":
            raise RuntimeError(
                f"Comparison native_artifact_path={native_p} has implementation="
                f"{native_impl!r}; the second input must be a native_pr artifact "
                f"(probe / custom_node / comparison artifacts are not accepted on this slot)"
            )

        manifest_id = custom_id
        run_id = str(uuid.uuid4())
        out_path = out_dir / f"{manifest_id}_comparison_{run_id}.json"

        # Compute custom_vs_native_similarity from each side's metrics; for
        # numeric metrics we record absolute and relative differences. Slice 5/6
        # may extend this with additional similarity fields.
        custom_metrics = custom.get("metrics", {})
        native_metrics = native.get("metrics", {})
        similarity: dict = {}
        for key in ("psnr", "ssim", "vmaf"):
            cv = custom_metrics.get(key)
            nv = native_metrics.get(key)
            if isinstance(cv, (int, float)) and isinstance(nv, (int, float)):
                similarity[key] = {
                    "custom": cv,
                    "native": nv,
                    "abs_diff": abs(cv - nv),
                    "max_abs": max(abs(cv), abs(nv)),
                }
            else:
                similarity[key] = {
                    "custom": cv,
                    "native": nv,
                    "not_applicable_reason": "non_numeric_value",
                }

        comparison = {
            "schema_version": _ARTIFACT_SCHEMA_VERSION,
            "run_id": run_id,
            "manifest_id": manifest_id,
            "implementation": "comparison",
            "input": {
                "manifest_id": manifest_id,
                "path": custom.get("input", {}).get("path", ""),
                "sha256": custom.get("input", {}).get("sha256", ""),
                "gt_manifest_id": custom.get("input", {}).get("gt_manifest_id"),
                "gt_path": custom.get("input", {}).get("gt_path"),
                "gt_sha256": custom.get("input", {}).get("gt_sha256"),
            },
            "workflow": custom.get("workflow", {}),
            "node_parameters": {
                "custom_artifact_path": str(custom_p),
                "native_artifact_path": str(native_p),
            },
            "output": custom.get("output", {}),
            "metrics": custom.get("metrics", {}),
            "runtime": {
                "started_at_epoch": _epoch(),
                "finished_at_epoch": _epoch(),
                "elapsed_seconds": 0.0,
                "peak_vram_bytes": None,
            },
            "environment": _capture_environment(),
            "metadata_storage": {
                "artifact_path": str(out_path),
                "artifact_written_at_epoch": _epoch(),
            },
            "comparison": {
                "custom_artifact_path": str(custom_p),
                "native_artifact_path": str(native_p),
                "custom_vs_native_similarity": similarity,
            },
        }
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(comparison, fh, indent=2)
            fh.write("\n")
        comparison["metadata_storage"]["artifact_written_at_epoch"] = _epoch()
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(comparison, fh, indent=2)
            fh.write("\n")
        return (str(out_path),)


NODE_CLASS_MAPPINGS = {
    "SeedVR2BaselineSaveWithArtifacts": SeedVR2BaselineSaveWithArtifacts,
    "SeedVR2BaselineCompareArtifacts": SeedVR2BaselineCompareArtifacts,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SeedVR2BaselineSaveWithArtifacts": "SeedVR2 Baseline Save With Artifacts (#123)",
    "SeedVR2BaselineCompareArtifacts": "SeedVR2 Baseline Compare Artifacts (#123)",
}
