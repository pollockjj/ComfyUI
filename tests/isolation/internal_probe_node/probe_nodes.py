from __future__ import annotations

import torch


class InternalIsolationProbeImage:
    CATEGORY = "tests/isolation"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    def run(self):
        image = torch.zeros((1, 2, 2, 3), dtype=torch.float32)
        image[:, :, :, 0] = 1.0
        return (image,)


class InternalIsolationProbeAudio:
    CATEGORY = "tests/isolation"
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    def run(self):
        waveform = torch.zeros((1, 1, 32), dtype=torch.float32)
        return ({"waveform": waveform, "sample_rate": 44100},)


NODE_CLASS_MAPPINGS = {
    "InternalIsolationProbeImage": InternalIsolationProbeImage,
    "InternalIsolationProbeAudio": InternalIsolationProbeAudio,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "InternalIsolationProbeImage": "Internal Isolation Probe Image",
    "InternalIsolationProbeAudio": "Internal Isolation Probe Audio",
}

