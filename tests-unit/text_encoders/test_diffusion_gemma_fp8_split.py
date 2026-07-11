import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from comfy.cli_args import args
if not torch.cuda.is_available():
    args.cpu = True

from comfy.text_encoders.diffusion_gemma import diffusion_gemma_detect  # noqa: E402


class TestDiffusionGemmaFp8Split(unittest.TestCase):
    def test_split_expert_state_dict_detects_unfused_runtime(self):
        sd = {
            "model.decoder.norm.weight": torch.ones(4, dtype=torch.bfloat16),
            "model.decoder.layers.0.experts.gate_proj.weight": torch.empty(2, 3, 4),
        }

        self.assertTrue(diffusion_gemma_detect((sd,))["unfused_experts"])

if __name__ == "__main__":
    unittest.main()
