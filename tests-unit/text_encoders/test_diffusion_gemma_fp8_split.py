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
    _ConditionedDecoderGraph,
    _mxfp8_self_conditioning,
    _ConditionedDecoderGraphCache,
    _append_preallocated_kv,
    _shared_mxfp8_input,
    diffusion_gemma_detect,
    diffusion_gemma_te,
)


class TestDiffusionGemmaFp8Split(unittest.TestCase):
    def test_mxfp8_self_conditioning_uses_sixteen_bf16_chunks(self):
        weight = self._mxfp8_bank((262144, 32), (262144, 4), device="cpu").weight
        probabilities = torch.ones((1, 262144), dtype=torch.bfloat16)

        with (
            mock.patch.object(sys.modules["comfy.quant_ops"].ck, "mxfp8_embedding", create=True, return_value=torch.ones((16384, 32), dtype=torch.bfloat16)) as dequantize,
            mock.patch.object(torch, "mm", side_effect=lambda a, b, out_dtype: a.float() @ b.float()),
        ):
            output = _mxfp8_self_conditioning(probabilities, weight)

        self.assertEqual(dequantize.call_count, 16)
        self.assertEqual(output.dtype, torch.float32)
        self.assertTrue(torch.equal(output, torch.full((1, 32), 262144.0)))

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

    def test_conditioned_decoder_graph_defers_first_replay(self):
        static_canvas = torch.empty((1, 4), dtype=torch.long)
        static_logits = torch.empty((1, 4, 8), dtype=torch.bfloat16)
        output = torch.empty((1, 4, 2), dtype=torch.bfloat16)
        owner = types.SimpleNamespace(model=mock.Mock(return_value=(output, None, None)))
        stream = mock.Mock()
        current_stream = mock.Mock()
        graph = mock.Mock()

        with (
            mock.patch.object(torch.cuda, "CUDAGraph", return_value=graph),
            mock.patch.object(torch.cuda, "graph", return_value=contextlib.nullcontext()),
            mock.patch.object(torch.cuda, "stream", return_value=contextlib.nullcontext()),
            mock.patch.object(torch.cuda, "current_stream", return_value=current_stream),
        ):
            decoder_graph = _ConditionedDecoderGraph(
                owner, static_canvas, static_logits, [], None, None, torch.bfloat16, stream)
            graph.replay.assert_not_called()
            self.assertIs(decoder_graph.replay(torch.ones_like(static_canvas), torch.ones_like(static_logits)), output)

        graph.replay.assert_called_once_with()
        self.assertTrue(torch.equal(static_canvas, torch.ones_like(static_canvas)))
        self.assertTrue(torch.equal(static_logits, torch.ones_like(static_logits)))
        current_stream.wait_stream.assert_called_once_with(stream)

    def test_preallocated_kv_uses_capacity_prefix_without_reallocation(self):
        past_key = torch.full((1, 1, 6, 2), -1.0)
        past_value = torch.full_like(past_key, -2.0)
        xk = torch.ones((1, 1, 2, 2))
        xv = torch.full_like(xk, 2.0)

        key, value, next_len = _append_preallocated_kv(past_key, past_value, xk, xv, 2)

        self.assertEqual(next_len, 4)
        self.assertEqual(key.shape[2], 4)
        self.assertEqual(key.data_ptr(), past_key.data_ptr())
        self.assertEqual(value.data_ptr(), past_value.data_ptr())
        self.assertTrue(torch.equal(key[:, :, 2:], xk))
        self.assertTrue(torch.equal(value[:, :, 2:], xv))

    def test_sliding_kv_compaction_preserves_backing_pointer(self):
        generate = DiffusionGenerate()
        generate.model = types.SimpleNamespace(
            decoder=types.SimpleNamespace(layers=[types.SimpleNamespace(sliding_window=5)]))
        key = torch.arange(8.0).view(1, 1, 8, 1)
        value = key + 10
        key_ptr = key.data_ptr()

        reserved = generate._reserve_kv_cache([(key, value, 8, 8)], 2)[0]

        self.assertEqual(reserved[0].data_ptr(), key_ptr)
        self.assertEqual(reserved[3], 4)
        self.assertTrue(torch.equal(reserved[0][0, 0, :4, 0], torch.arange(4.0, 8.0)))
        self.assertTrue(torch.equal(reserved[1][0, 0, :4, 0], torch.arange(14.0, 18.0)))

    def test_graph_cache_releases_workspace_and_graphs(self):
        stream = mock.Mock()
        graph = mock.Mock()
        ck = sys.modules["comfy.quant_ops"].ck
        with (
            mock.patch.object(torch.cuda, "Stream", return_value=stream),
            mock.patch.object(torch.cuda, "device", return_value=contextlib.nullcontext()),
            mock.patch.object(torch.cuda, "graph_pool_handle", return_value="pool"),
            mock.patch.object(ck, "reserve_cuda_stream_workspaces", create=True) as reserve,
            mock.patch.object(ck, "release_cuda_stream_workspaces", create=True) as release,
        ):
            cache = _ConditionedDecoderGraphCache("key", torch.device("cpu"), 1, 4, 8, torch.bfloat16)
            cache.graphs["geometry"] = graph
            cache.close()

        reserve.assert_called_once_with(stream)
        graph.close.assert_called_once_with()
        release.assert_called_once_with(stream)

    def test_graph_cache_key_mismatch_closes_stale_cache(self):
        generate = DiffusionGenerate()
        parameter = torch.nn.Parameter(torch.ones(1))
        generate.model = types.SimpleNamespace(
            decoder=types.SimpleNamespace(parameters=lambda: [parameter], buffers=lambda: []),
            config=types.SimpleNamespace(canvas_length=4, vocab_size=8),
        )
        stale = mock.Mock(key="stale")
        replacement = mock.Mock()
        generate._conditioned_decoder_graph_cache = stale
        embeds = torch.empty((1, 3, 2))

        with mock.patch(
            "comfy.text_encoders.diffusion_gemma._ConditionedDecoderGraphCache",
            return_value=replacement,
        ):
            self.assertIs(generate._get_decoder_graph_cache(embeds, 2, torch.bfloat16), replacement)

        stale.close.assert_called_once_with()
        self.assertIs(generate._conditioned_decoder_graph_cache, replacement)

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
