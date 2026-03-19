"""Serializers for custom node data types.

These serializers exist because of specific custom node conversions (DA3,
GeometryPack, etc.) and are not required by core ComfyUI.  New custom node
conversions that introduce types not covered here should add their
serializers to this file.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from pyisolate.interfaces import SerializerRegistryProtocol  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)

_announced: set[str] = set()


def _announce(name: str, desc: str) -> None:
    if name not in _announced:
        _announced.add(name)
        logger.info("][ Serializer: %s — %s", name, desc)


def register_custom_node_serializers(registry: SerializerRegistryProtocol) -> None:
    """Register all custom-node-originated serializers."""

    # -- PLY (comfy_api.latest._util.ply_types) --------------------------------
    # PLY point cloud container from comfy_api.
    # Origin: ComfyUI comfy_api by ComfyOrg (Alexander Piskun)
    # Used by: ComfyUI-DepthAnythingV3, ComfyUI-GeometryPack

    def serialize_ply(obj: Any) -> Dict[str, Any]:
        _announce("PLY", "comfy_api PLY (by ComfyOrg) serializer 1.0 (base64/tensors) for ComfyUI-DepthAnythingV3, ComfyUI-GeometryPack")
        import base64
        import torch
        if obj.raw_data is not None:
            return {
                "__type__": "PLY",
                "raw_data": base64.b64encode(obj.raw_data).decode("ascii"),
            }
        result: Dict[str, Any] = {"__type__": "PLY", "points": torch.from_numpy(obj.points)}
        if obj.colors is not None:
            result["colors"] = torch.from_numpy(obj.colors)
        if obj.confidence is not None:
            result["confidence"] = torch.from_numpy(obj.confidence)
        if obj.view_id is not None:
            result["view_id"] = torch.from_numpy(obj.view_id)
        return result

    def deserialize_ply(data: Any) -> Any:
        import base64
        from comfy_api.latest._util.ply_types import PLY
        if "raw_data" in data:
            return PLY(raw_data=base64.b64decode(data["raw_data"]))
        return PLY(
            points=data["points"],
            colors=data.get("colors"),
            confidence=data.get("confidence"),
            view_id=data.get("view_id"),
        )

    registry.register("PLY", serialize_ply, deserialize_ply, data_type=True)

    # -- NPZ (comfy_api.latest._util.npz_types) --------------------------------
    # NPZ depth map frame container from comfy_api.
    # Origin: ComfyUI comfy_api by ComfyOrg (Alexander Piskun)
    # Used by: ComfyUI-DepthAnythingV3

    def serialize_npz(obj: Any) -> Dict[str, Any]:
        _announce("NPZ", "comfy_api NPZ (by ComfyOrg) serializer 1.0 (base64 frames) for ComfyUI-DepthAnythingV3")
        import base64
        return {
            "__type__": "NPZ",
            "frames": [base64.b64encode(f).decode("ascii") for f in obj.frames],
        }

    def deserialize_npz(data: Any) -> Any:
        import base64
        from comfy_api.latest._util.npz_types import NPZ
        return NPZ(frames=[base64.b64decode(f) for f in data["frames"]])

    registry.register("NPZ", serialize_npz, deserialize_npz, data_type=True)

    # -- File3D (comfy_api.latest._util.geometry_types) -------------------------
    # 3D geometry file container from comfy_api.
    # Origin: ComfyUI comfy_api by ComfyOrg (Alexander Piskun)
    # Used by: ComfyUI-DepthAnythingV3, ComfyUI-GeometryPack

    def serialize_file3d(obj: Any) -> Dict[str, Any]:
        _announce("File3D", "comfy_api File3D (by ComfyOrg/Alexander Piskun) serializer 1.0 (base64, format) for ComfyUI-DepthAnythingV3, ComfyUI-GeometryPack")
        import base64
        return {
            "__type__": "File3D",
            "format": obj.format,
            "data": base64.b64encode(obj.get_bytes()).decode("ascii"),
        }

    def deserialize_file3d(data: Any) -> Any:
        import base64
        from io import BytesIO
        from comfy_api.latest._util.geometry_types import File3D
        return File3D(BytesIO(base64.b64decode(data["data"])), file_format=data["format"])

    registry.register("File3D", serialize_file3d, deserialize_file3d, data_type=True)

    # -- VIDEO (comfy_api.latest._input_impl.video_types) -----------------------
    # Video frame/audio container from comfy_api.
    # Origin: ComfyUI comfy_api by ComfyOrg (Alexander Piskun)
    # Used by: ComfyUI-VideoHelperSuite, ComfyUI-WanVideoWrapper, and other video node packs

    def serialize_video(obj: Any) -> Dict[str, Any]:
        _announce("VIDEO", "comfy_api Video (by ComfyOrg/Alexander Piskun) serializer 1.0 (tensors, fraction, dict) for video node packs")
        components = obj.get_components()
        images = components.images.detach() if components.images.requires_grad else components.images
        result: Dict[str, Any] = {
            "__type__": "VIDEO",
            "images": images,
            "frame_rate_num": components.frame_rate.numerator,
            "frame_rate_den": components.frame_rate.denominator,
        }
        if components.audio is not None:
            waveform = components.audio["waveform"]
            if waveform.requires_grad:
                waveform = waveform.detach()
            result["audio_waveform"] = waveform
            result["audio_sample_rate"] = components.audio["sample_rate"]
        if components.metadata is not None:
            result["metadata"] = components.metadata
        return result

    def deserialize_video(data: Any) -> Any:
        from fractions import Fraction
        from comfy_api.latest._input_impl.video_types import VideoFromComponents
        from comfy_api.latest._util.video_types import VideoComponents
        audio = None
        if "audio_waveform" in data:
            audio = {"waveform": data["audio_waveform"], "sample_rate": data["audio_sample_rate"]}
        components = VideoComponents(
            images=data["images"],
            frame_rate=Fraction(data["frame_rate_num"], data["frame_rate_den"]),
            audio=audio,
            metadata=data.get("metadata"),
        )
        return VideoFromComponents(components)

    registry.register("VIDEO", serialize_video, deserialize_video, data_type=True)
    registry.register("VideoFromFile", serialize_video, deserialize_video, data_type=True)
    registry.register("VideoFromComponents", serialize_video, deserialize_video, data_type=True)
