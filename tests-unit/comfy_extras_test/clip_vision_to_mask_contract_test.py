import sys

import pytest
import torch
from unittest.mock import MagicMock, patch

mock_nodes = MagicMock()
mock_nodes.MAX_RESOLUTION = 16384
mock_server = MagicMock()

with patch.dict("sys.modules", {"nodes": mock_nodes, "server": mock_server}):
    import comfy_extras.nodes_mask as nodes_mask
    import comfy_extras.nodes_images as nodes_images

ClipVisionToMask = nodes_mask.ClipVisionToMask
MaskToImage = nodes_mask.MaskToImage
ImageFromBatch = nodes_images.ImageFromBatch


class FakeClipVisionOutput:
    def __init__(self, **values):
        self.__dict__.update(values)

    def __getitem__(self, key):
        return getattr(self, key)


class TestClipVisionToMaskContract:
    def test_non_birefnet_layout_raises(self):
        clip_vision_output = FakeClipVisionOutput(
            last_hidden_state=torch.zeros((1, 257, 1024), dtype=torch.float32),
            source_image_sizes=[(64, 64)],
            clip_vision_model_type="clip_vision_model",
        )

        with pytest.raises(ValueError, match="ClipVisionToMask expects a 4D single-channel BiRefNet mask tensor"):
            ClipVisionToMask.execute(clip_vision_output)

    def test_source_size_length_mismatch_raises(self):
        clip_vision_output = FakeClipVisionOutput(
            last_hidden_state=torch.zeros((2, 1, 4, 4), dtype=torch.float32),
            source_image_sizes=[(8, 8)],
            clip_vision_model_type="birefnet",
        )

        with pytest.raises(ValueError, match="ClipVisionToMask source_image_sizes length must equal batch size"):
            ClipVisionToMask.execute(clip_vision_output)

    def test_uncropped_source_restore(self):
        clip_vision_output = FakeClipVisionOutput(
            last_hidden_state=torch.zeros((2, 1, 4, 4), dtype=torch.float32),
            source_image_sizes=[(6, 6), (6, 6)],
            clip_vision_model_type="birefnet",
            source_restore_crop_mode="none",
            preprocess_image_sizes=[(4, 4), (4, 4)],
        )

        def fake_upscale(sample, width, height, method, crop):
            assert (width, height, method, crop) == (6, 6, "bilinear", "disabled")
            return torch.full((sample.shape[0], sample.shape[1], height, width), 0.75, dtype=sample.dtype)

        with patch.object(nodes_mask.comfy.utils, "common_upscale", side_effect=fake_upscale) as common_upscale:
            result = ClipVisionToMask.execute(clip_vision_output)

        assert result[0].shape == (2, 1, 6, 6)
        assert common_upscale.call_count == 2

    def test_mixed_batch_source_restore_is_deferred_until_image_from_batch(self):
        clip_vision_output = FakeClipVisionOutput(
            last_hidden_state=torch.zeros((2, 1, 4, 4), dtype=torch.float32),
            source_image_sizes=[(6, 6), (2, 3)],
            clip_vision_model_type="birefnet",
            source_restore_crop_mode="none",
            preprocess_image_sizes=[(4, 4), (4, 4)],
        )

        mask = ClipVisionToMask.execute(clip_vision_output)[0]
        assert mask.shape == (2, 1, 4, 4)
        assert mask.source_image_sizes == [(6, 6), (2, 3)]

        image = MaskToImage.execute(mask)[0]

        def fake_upscale(sample, width, height, method, crop):
            assert (width, height, method, crop) == (3, 2, "bilinear", "disabled")
            return torch.full((sample.shape[0], sample.shape[1], height, width), 0.5, dtype=sample.dtype)

        with patch.object(nodes_images.comfy.utils, "common_upscale", side_effect=fake_upscale) as common_upscale:
            result = ImageFromBatch.execute(image, 1, 1)

        assert result[0].shape == (1, 2, 3, 3)
        assert common_upscale.call_count == 1

    def test_center_crop_restore_is_skipped_after_image_from_batch(self):
        clip_vision_output = FakeClipVisionOutput(
            last_hidden_state=torch.zeros((1, 1, 4, 4), dtype=torch.float32),
            source_image_sizes=[(8, 6)],
            clip_vision_model_type="birefnet",
            source_restore_crop_mode="center",
            preprocess_image_sizes=[(4, 4)],
        )

        mask = ClipVisionToMask.execute(clip_vision_output)[0]
        image = MaskToImage.execute(mask)[0]

        with patch.object(nodes_images.comfy.utils, "common_upscale") as common_upscale:
            result = ImageFromBatch.execute(image, 0, 1)

        assert result[0].shape == (1, 4, 4, 3)
        assert common_upscale.call_count == 0
