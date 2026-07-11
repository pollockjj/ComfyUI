import os
import sys
import types
import unittest

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from comfy.cli_args import args
if not torch.cuda.is_available():
    args.cpu = True

from comfy.quant_ops import QuantizedTensor, TensorCoreMXFP8Layout  # noqa: E402
from comfy.text_encoders.diffusion_gemma import DiffusionGemmaExperts, diffusion_gemma_detect  # noqa: E402


class TestDiffusionGemmaFp8Split(unittest.TestCase):
    def test_split_expert_state_dict_detects_unfused_runtime(self):
        sd = {
            "model.decoder.norm.weight": torch.ones(4, dtype=torch.bfloat16),
            "model.decoder.layers.0.experts.gate_proj.weight": torch.empty(2, 3, 4),
        }

        self.assertTrue(diffusion_gemma_detect((sd,))["unfused_experts"])

    @staticmethod
    def _mxfp8_bank(qdata_shape, scale_shape, quant_format="mxfp8"):
        params = TensorCoreMXFP8Layout.Params(
            scale=torch.empty(scale_shape, dtype=torch.float8_e8m0fnu, device="meta"),
            orig_dtype=torch.bfloat16,
            orig_shape=qdata_shape,
        )
        return types.SimpleNamespace(
            quant_format=quant_format,
            weight=QuantizedTensor(
                torch.empty(qdata_shape, dtype=torch.float8_e4m3fn, device="meta"),
                "TensorCoreMXFP8Layout",
                params,
            ),
            bias=None,
            _full_precision_mm=False,
            weight_function=[],
            bias_function=[],
            weight_lowvram_function=None,
            bias_lowvram_function=None,
        )

    def test_mxfp8_expert_bank_contract(self):
        experts = DiffusionGemmaExperts.__new__(DiffusionGemmaExperts)
        torch.nn.Module.__init__(experts)
        experts.unfused = True
        experts._bank_mode = None
        experts._fused_banks_compatible = False
        experts._grouped_mxfp8_compatible = False
        experts._banks = (
            self._mxfp8_bank((128, 704, 2816), (128, 768, 88)),
            self._mxfp8_bank((128, 704, 2816), (128, 768, 88)),
            self._mxfp8_bank((128, 2816, 704), (128, 2816, 24)),
        )

        experts._configure_loaded_banks(experts, None)
        self.assertEqual(experts._bank_mode, "unfused_mxfp8")
        self.assertTrue(experts._grouped_mxfp8_compatible)

        experts._banks = (self._mxfp8_bank((128, 704, 2816), (128, 704, 88)), *experts._banks[1:])
        with self.assertRaisesRegex(ValueError, "MXFP8 expert bank contract mismatch"):
            experts._configure_loaded_banks(experts, None)

    def test_fused_mxfp8_expert_bank_contract(self):
        experts = DiffusionGemmaExperts.__new__(DiffusionGemmaExperts)
        torch.nn.Module.__init__(experts)
        experts.unfused = False
        experts._bank_mode = None
        experts._fused_banks_compatible = False
        experts._fused_mxfp8_banks_compatible = False
        experts._grouped_mxfp8_compatible = False
        experts._banks = (
            self._mxfp8_bank((128, 1408, 2816), (128, 1408, 88), experts.fused_mxfp8_format),
            self._mxfp8_bank((128, 2816, 704), (128, 2816, 24), experts.fused_mxfp8_format),
        )

        experts._configure_loaded_banks(experts, None)
        self.assertEqual(experts._bank_mode, "fused_mxfp8")
        self.assertTrue(experts._fused_mxfp8_banks_compatible)

        experts._banks = (self._mxfp8_bank((128, 1408, 2816), (128, 1408, 84), experts.fused_mxfp8_format), experts._banks[1])
        with self.assertRaisesRegex(ValueError, "fused MXFP8 expert bank contract mismatch"):
            experts._configure_loaded_banks(experts, None)

if __name__ == "__main__":
    unittest.main()
