import os
import sys
import unittest
from unittest import mock

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from comfy.cli_args import args
if not torch.cuda.is_available():
    args.cpu = True

import comfy.quant_ops  # noqa: E402
from comfy.text_encoders.diffusion_gemma import (  # noqa: E402
    DiffusionGemmaConfig,
    DiffusionGemmaExperts,
    DiffusionGemmaRouter,
)


class FakeOps:
    Linear = torch.nn.Linear

    class MoEExperts(torch.nn.Module):
        def __init__(self, num_experts, in_features, out_features, bias=False, device=None, dtype=None):
            super().__init__()
            self.num_experts = num_experts
            self.in_features = in_features
            self.out_features = out_features
            self.layout_type = None


def _router():
    config = DiffusionGemmaConfig(hidden_size=4, num_experts=128, top_k_experts=8)
    router = DiffusionGemmaRouter(config, device="cpu", dtype=torch.float32, ops=FakeOps)
    with torch.no_grad():
        router.scale.fill_(1)
        router.per_expert_scale.copy_(torch.linspace(0.5, 1.5, 128))
        router.proj.weight.copy_(torch.arange(512, dtype=torch.float32).reshape(128, 4) / 512)
    return router


class TestDiffusionGemmaRouting(unittest.TestCase):
    def test_native_flag_uses_public_kitchen_result(self):
        router = _router()
        hidden = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        weights = torch.ones(3, 8, dtype=torch.float32)
        ids = torch.zeros(3, 8, dtype=torch.int32)

        with mock.patch.object(
            comfy.quant_ops.ck, "gemma4_fused_routing", return_value=(weights, ids), create=True
        ) as fused:
            result = router(hidden, use_fused_routing=True)

        fused.assert_called_once()
        self.assertIs(result[0], weights)
        self.assertIs(result[1], ids)

    def test_no_capable_backend_preserves_torch_fallback(self):
        router = _router()
        hidden = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        expected = router(hidden, use_fused_routing=False)
        error = comfy.quant_ops.ck.NoCapableBackendError("gemma4_fused_routing", {})

        with mock.patch.object(
            comfy.quant_ops.ck, "gemma4_fused_routing", side_effect=error, create=True
        ):
            actual = router(hidden, use_fused_routing=True)

        self.assertTrue(torch.equal(actual[0], expected[0]))
        self.assertTrue(torch.equal(actual[1], expected[1]))
        self.assertEqual(actual[1].dtype, torch.int64)

    def test_default_path_does_not_call_kitchen(self):
        router = _router()
        hidden = torch.arange(12, dtype=torch.float32).reshape(3, 4)

        with mock.patch.object(comfy.quant_ops.ck, "gemma4_fused_routing", create=True) as fused:
            _, ids = router(hidden)

        fused.assert_not_called()
        self.assertEqual(ids.dtype, torch.int64)

    def test_non_capability_errors_propagate(self):
        router = _router()
        hidden = torch.arange(12, dtype=torch.float32).reshape(3, 4)

        with mock.patch.object(
            comfy.quant_ops.ck,
            "gemma4_fused_routing",
            side_effect=ValueError("bad routing"),
            create=True,
        ), self.assertRaisesRegex(ValueError, "bad routing"):
            router(hidden, use_fused_routing=True)

    def test_native_moe_failure_converts_ids_before_grouped_fallback(self):
        config = DiffusionGemmaConfig(
            hidden_size=4, moe_intermediate_size=3, num_experts=2, top_k_experts=1
        )
        experts = DiffusionGemmaExperts(config, ops=FakeOps)
        experts._has_fused_nvfp4_banks = mock.Mock(return_value=True)
        experts._supports_native_fused_nvfp4 = mock.Mock(return_value=True)
        experts._supports_grouped_fused_nvfp4 = mock.Mock(return_value=True)
        error = comfy.quant_ops.ck.NoCapableBackendError("fused_moe_nvfp4", {})
        experts._forward_native_fused_nvfp4 = mock.Mock(side_effect=error)

        def grouped(hidden_states, top_k_index, _top_k_weights):
            self.assertEqual(top_k_index.dtype, torch.int64)
            return hidden_states

        experts._forward_grouped_fused_nvfp4 = grouped
        hidden = torch.zeros(2, 4)
        ids = torch.zeros(2, 1, dtype=torch.int32)
        weights = torch.ones(2, 1)

        self.assertTrue(torch.equal(experts(hidden, ids, weights), hidden))


if __name__ == "__main__":
    unittest.main()
