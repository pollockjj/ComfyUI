import contextlib
import os
import sys
import types
import unittest
from unittest import mock

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from comfy.cli_args import args
if not torch.cuda.is_available():
    args.cpu = True

from comfy.quant_ops import QuantizedTensor, TensorCoreMXFP8Layout  # noqa: E402
from comfy.text_encoders.diffusion_gemma import (  # noqa: E402
    DiffusionGenerate,
    DiffusionGemmaExperts,
    _ConditionedDenoiseStepGraph,
    _shared_mxfp8_input,
    diffusion_gemma_detect,
    diffusion_gemma_te,
)


class TestDiffusionGemmaFp8Split(unittest.TestCase):
    def test_quantized_diffusion_gemma_enables_native_compute(self):
        self.assertTrue(diffusion_gemma_te(llama_quantization_metadata={"mixed_ops": True}).supports_native_quantized_compute)
        self.assertFalse(diffusion_gemma_te().supports_native_quantized_compute)

    def test_dense_mxfp8_projections_share_one_activation_quantization(self):
        modules = tuple(self._mxfp8_bank((32, 32), (128, 4), device="cpu") for _ in range(3))
        for module in modules:
            module.layout_type = "TensorCoreMXFP8Layout"
            module.input_scale = None
            module.comfy_force_cast_weights = False
        x = torch.zeros((1, 32, 32), dtype=torch.bfloat16)
        sentinel = object()

        with mock.patch.object(QuantizedTensor, "from_float", return_value=sentinel) as quantize:
            self.assertIs(_shared_mxfp8_input(x, modules), sentinel)

        quantize.assert_called_once_with(mock.ANY, "TensorCoreMXFP8Layout", scale=None)
        quantized_source = quantize.call_args.args[0]
        self.assertEqual(quantized_source.shape, (32, 32))
        self.assertEqual(quantized_source.data_ptr(), x.data_ptr())

    def test_decoder_graph_requires_resident_weights_and_static_vram(self):
        generate = DiffusionGenerate()
        resident = types.SimpleNamespace(
            parameters=lambda: [types.SimpleNamespace(device=torch.device("cuda:0"))],
            modules=lambda: [],
        )
        generate.model = types.SimpleNamespace(decoder=resident)

        with (
            mock.patch.object(torch.cuda, "get_device_capability", return_value=(12, 0)),
            mock.patch.object(torch.cuda.memory, "get_allocator_backend", return_value="native"),
            mock.patch.object(args, "disable_dynamic_vram", True),
            mock.patch.object(sys.modules["comfy.quant_ops"].ck, "reserve_cuda_stream_workspaces", create=True),
            mock.patch.object(sys.modules["comfy.quant_ops"].ck, "release_cuda_stream_workspaces", create=True),
        ):
            self.assertTrue(generate._use_conditioned_decoder_graph(torch.device("cuda:0"), torch.bfloat16))
            resident.parameters = lambda: [types.SimpleNamespace(device=torch.device("cpu"))]
            self.assertFalse(generate._use_conditioned_decoder_graph(torch.device("cuda:0"), torch.bfloat16))

    def test_conditioned_denoise_graph_capture_is_the_second_step(self):
        current_canvas = torch.zeros((1, 4), dtype=torch.long)
        self_conditioning_logits = torch.zeros((1, 4, 8), dtype=torch.bfloat16)
        output = torch.empty((1, 4, 2), dtype=torch.bfloat16)
        processed_logits = torch.zeros((1, 4, 8), dtype=torch.float32)
        model = mock.Mock(return_value=(output, None, None))
        model.config = types.SimpleNamespace(vocab_size=8)
        owner = types.SimpleNamespace(
            model=model,
            _native_processed_logits=mock.Mock(return_value=processed_logits),
        )
        stream = mock.Mock()
        current_stream = mock.Mock()
        graph = mock.Mock()
        generator = torch.Generator().manual_seed(7)
        ck = sys.modules["comfy.quant_ops"].ck

        with (
            mock.patch.object(torch.cuda, "CUDAGraph", return_value=graph),
            mock.patch.object(torch.cuda, "graph", return_value=contextlib.nullcontext()),
            mock.patch.object(torch.cuda, "stream", return_value=contextlib.nullcontext()),
            mock.patch.object(torch.cuda, "current_stream", return_value=current_stream),
            mock.patch.object(
                ck,
                "categorical_stats_sample",
                return_value=(torch.zeros((1, 4)), torch.zeros((1, 4), dtype=torch.long),
                              torch.ones((1, 4), dtype=torch.long)),
            ),
        ):
            decoder_graph = _ConditionedDenoiseStepGraph(
                owner, current_canvas, self_conditioning_logits, [], None, None,
                torch.bfloat16, stream, generator, 2.0, 0.1)
            graph.replay.assert_called_once_with()
            decoder_graph.replay(
                torch.ones_like(current_canvas), torch.ones_like(self_conditioning_logits), 3.0)

        self.assertEqual(graph.replay.call_count, 2)
        graph.register_generator_state.assert_called_once_with(generator)
        self.assertTrue(torch.equal(decoder_graph.current_canvas, torch.ones_like(current_canvas)))
        self.assertTrue(torch.equal(
            decoder_graph.self_conditioning_logits, torch.ones_like(self_conditioning_logits)))
        self.assertEqual(decoder_graph.inverse_temperature.item(), 3.0)
        self.assertEqual(current_stream.wait_stream.call_count, 2)

    def test_native_processed_logits_uses_device_inverse_temperature(self):
        generate = DiffusionGenerate()
        generate.model = types.SimpleNamespace(config=types.SimpleNamespace(final_logit_softcapping=30.0))
        generate._raw_logits = mock.Mock(return_value=torch.ones((2, 3), dtype=torch.bfloat16))
        ck = sys.modules["comfy.quant_ops"].ck

        with mock.patch.object(ck, "softcap_scale", return_value=torch.full((2, 3), 4.0)) as softcap:
            result = generate._native_processed_logits(torch.empty(0), torch.tensor(0.25))

        softcap.assert_called_once_with(mock.ANY, 30.0, 1.0)
        self.assertTrue(torch.equal(result, torch.ones((2, 3))))

    def test_split_expert_state_dict_detects_unfused_runtime(self):
        sd = {
            "model.decoder.norm.weight": torch.ones(4, dtype=torch.bfloat16),
            "model.decoder.layers.0.experts.gate_proj.weight": torch.empty(2, 3, 4),
        }

        self.assertTrue(diffusion_gemma_detect((sd,))["unfused_experts"])

    @staticmethod
    def _mxfp8_bank(qdata_shape, scale_shape, quant_format="mxfp8", device="meta"):
        params = TensorCoreMXFP8Layout.Params(
            scale=torch.empty(scale_shape, dtype=torch.float8_e8m0fnu, device=device),
            orig_dtype=torch.bfloat16,
            orig_shape=qdata_shape,
        )
        return types.SimpleNamespace(
            quant_format=quant_format,
            weight=QuantizedTensor(
                torch.empty(qdata_shape, dtype=torch.float8_e4m3fn, device=device),
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
