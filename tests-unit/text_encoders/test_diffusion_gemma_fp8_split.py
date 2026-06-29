import os
import sys
import unittest
from unittest import mock

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from comfy.cli_args import args
if not torch.cuda.is_available():
    args.cpu = True

from comfy.text_encoders.diffusion_gemma import (  # noqa: E402
    DiffusionGemmaClipModel,
    DiffusionGemmaConfig,
    DiffusionGemmaExperts,
    diffusion_gemma_detect,
    sd1_clip,
)


class FakeOps:
    class MoEExperts(torch.nn.Module):
        def __init__(self, num_experts, in_features, out_features, bias=False, device=None, dtype=None):
            super().__init__()
            self.num_experts = num_experts
            self.in_features = in_features
            self.out_features = out_features
            self.layout_type = None


def _small_config(unfused_experts):
    return DiffusionGemmaConfig(
        hidden_size=4,
        moe_intermediate_size=3,
        num_experts=2,
        top_k_experts=1,
        unfused_experts=unfused_experts,
    )


class TestDiffusionGemmaFp8Split(unittest.TestCase):
    def test_split_expert_state_dict_detects_unfused_runtime(self):
        sd = {
            "model.decoder.norm.weight": torch.ones(4, dtype=torch.bfloat16),
            "model.decoder.layers.0.experts.gate_proj.weight": torch.empty(2, 3, 4),
        }

        self.assertTrue(diffusion_gemma_detect((sd,))["unfused_experts"])

    def test_quantized_unfused_experts_bypass_grouped_bank_dequant_path(self):
        experts = DiffusionGemmaExperts(_small_config(unfused_experts=True), ops=FakeOps)
        experts.grouped_min_tokens = 1
        experts.gate_proj.layout_type = "TensorCoreFP8E4M3Layout"
        calls = []

        def grouped(hidden_states, top_k_index, top_k_weights):
            raise AssertionError("quantized unfused experts must not use grouped bank bmm")

        def loop(hidden_states, top_k_index, top_k_weights):
            calls.append("loop")
            return torch.ones_like(hidden_states)

        experts._forward_grouped = grouped
        experts._forward_loop = loop
        hidden_states = torch.zeros(2, 4)
        top_k_index = torch.zeros(2, 1, dtype=torch.long)
        top_k_weights = torch.ones(2, 1)

        out = experts(hidden_states, top_k_index, top_k_weights)

        self.assertEqual(calls, ["loop"])
        self.assertTrue(torch.equal(out, torch.ones_like(hidden_states)))

    def test_nonquantized_unfused_experts_keep_grouped_path(self):
        experts = DiffusionGemmaExperts(_small_config(unfused_experts=True), ops=FakeOps)
        experts.grouped_min_tokens = 1
        calls = []

        def grouped(hidden_states, top_k_index, top_k_weights):
            calls.append("grouped")
            return torch.ones_like(hidden_states)

        def loop(hidden_states, top_k_index, top_k_weights):
            raise AssertionError("nonquantized unfused experts should keep grouped routing")

        experts._forward_grouped = grouped
        experts._forward_loop = loop
        hidden_states = torch.zeros(2, 4)
        top_k_index = torch.zeros(2, 1, dtype=torch.long)
        top_k_weights = torch.ones(2, 1)

        out = experts(hidden_states, top_k_index, top_k_weights)

        self.assertEqual(calls, ["grouped"])
        self.assertTrue(torch.equal(out, torch.ones_like(hidden_states)))

    def test_split_clip_model_allows_quantized_matmul(self):
        captured = {}

        class SplitClip(DiffusionGemmaClipModel):
            config_overrides = {"unfused_experts": True}

        def fake_init(self, *args, **kwargs):
            captured.update(kwargs)

        with mock.patch.object(sd1_clip.SDClipModel, "__init__", fake_init):
            SplitClip(dtype=torch.bfloat16)

        linear = captured["model_options"]["custom_operations"].Linear(4, 4, device="cpu", dtype=torch.bfloat16)
        self.assertFalse(linear._full_precision_mm)


if __name__ == "__main__":
    unittest.main()
