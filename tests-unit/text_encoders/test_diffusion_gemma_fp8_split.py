import contextlib
import ctypes
import json
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

import comfy.model_management  # noqa: E402
from comfy.ops import mixed_precision_ops  # noqa: E402
from comfy.quant_ops import (  # noqa: E402
    QuantizedTensor,
    TensorCoreConvRotW4A4Layout,
    TensorCoreMXFP8Layout,
    TensorWiseINT8Layout,
)
from comfy.text_encoders.diffusion_gemma import (  # noqa: E402
    DiffusionGenerate,
    DiffusionGemmaAttention,
    DiffusionGemmaConfig,
    DiffusionGemmaExperts,
    DiffusionGemmaClipModel,
    DiffusionGemmaMLP,
    _REQUEST_W4_ACTIVATION_DTYPE,
    _REQUEST_W4A8_LAYERS,
    _ConditionedDecoderGraph,
    _ConditionedDecoderGraphExecution,
    _int8_self_conditioning,
    _append_preallocated_kv,
    _make_dg_scaled_embedding,
    _shared_mxfp8_input,
    diffusion_gemma_detect,
    diffusion_gemma_te,
)
from comfy.text_encoders.gemma4 import _Gemma4TokenBatch  # noqa: E402


class TestDiffusionGemmaFp8Split(unittest.TestCase):
    def test_embedding_scale_matches_weight_dtype_on_output_device(self):
        class Ops:
            class Embedding(torch.nn.Embedding):
                def forward(self, input_ids, out_dtype=None):
                    return super().forward(input_ids).to(out_dtype or self.weight.dtype)

                def weighted_embedding(self, probabilities):
                    return probabilities @ self.weight

        embedding = _make_dg_scaled_embedding(Ops, 2, 3, "cpu", torch.bfloat16)
        embedding.weight.detach().fill_(1)
        expected_scale = torch.tensor(3 ** 0.5, dtype=torch.bfloat16)

        output = embedding(torch.tensor([0]), out_dtype=torch.bfloat16)
        weighted = embedding.weighted_embedding(torch.tensor([[1.0, 0.0]], dtype=torch.bfloat16))

        self.assertTrue(torch.equal(output, torch.ones_like(output) * expected_scale))
        self.assertTrue(torch.equal(weighted, torch.ones_like(weighted) * expected_scale))

    @staticmethod
    def _qkv_marker(layer):
        global_layer = layer % 6 == 5
        config = {
            "format": "mxfp8",
            "full_precision_matrix_mult": False,
            "artifact_contract": "diffusiongemma_mxfp8_qkv_fused.v1",
            "projection_order": ["q_proj", "k_proj"] if global_layer else ["q_proj", "k_proj", "v_proj"],
            "projection_splits": [8192, 1024] if global_layer else [4096, 2048, 2048],
        }
        return torch.tensor(list(json.dumps(config).encode()), dtype=torch.uint8)

    def test_mxfp8_self_conditioning_cleans_up_failed_resident_weight(self):
        embedding = mixed_precision_ops().Embedding(128, 128, device="cpu", dtype=torch.bfloat16)
        bank = self._mxfp8_bank((128, 128), (128, 4), device="cpu")
        embedding.weight = torch.nn.Parameter(bank.weight, requires_grad=False)
        for name, value in vars(bank).items():
            if name != "weight":
                setattr(embedding, name, value)
        embedding.layout_type = "TensorCoreMXFP8Layout"
        embedding.comfy_force_cast_weights = False
        resident = embedding.weight
        probabilities = torch.ones((1, 128), dtype=torch.float32)
        offload_token = object()

        with (
            mock.patch("comfy.ops.cast_bias_weight", return_value=(resident, None, offload_token)),
            mock.patch.object(sys.modules["comfy.quant_ops"].ck, "mxfp8_weighted_embedding", create=True, side_effect=RuntimeError("sentinel")) as weighted,
            mock.patch("comfy.ops.uncast_bias_weight") as uncast,
            self.assertRaisesRegex(RuntimeError, "sentinel"),
        ):
            embedding.weighted_embedding(probabilities)

        qdata, scales, actual_probabilities = weighted.call_args.args
        self.assertIs(qdata, resident._qdata)
        self.assertIs(scales, resident._params.scale)
        self.assertEqual(actual_probabilities.dtype, torch.bfloat16)
        uncast.assert_called_once_with(embedding, resident, None, offload_token)

    def test_int8_self_conditioning_dequantizes_bounded_rows(self):
        weight = self._int8_bank((2, 64), (2, 1), 64, device="cpu").weight
        probabilities = torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16)
        dequantized = torch.arange(128, dtype=torch.bfloat16).reshape(2, 64)

        with (
            mock.patch("comfy.text_encoders.diffusion_gemma._INT8_SELF_CONDITIONING_CHUNK", 1),
            mock.patch.object(QuantizedTensor, "dequantize", side_effect=[dequantized[:1], dequantized[1:]]) as dequantize,
            mock.patch.object(torch, "mm", side_effect=lambda a, b, out_dtype: a.float() @ b.float()),
        ):
            output = _int8_self_conditioning(probabilities, weight)

        self.assertEqual(dequantize.call_count, 2)
        self.assertTrue(torch.equal(output, probabilities.float() @ dequantized.float()))

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

    def test_dense_mxfp8_fuses_activation_quantization(self):
        x = torch.zeros((1, 32, 32), dtype=torch.bfloat16)
        gate, up = torch.ones_like(x), torch.full_like(x, 2)
        qdata = torch.empty((32, 32), dtype=torch.float8_e4m3fn)
        scales = torch.empty((128, 4), dtype=torch.float8_e8m0fnu)
        mlp = types.SimpleNamespace(gate_proj=object(), up_proj=object(), down_proj=object())
        linear = mock.Mock(side_effect=[gate, up, "output"])
        with (
            mock.patch("comfy.text_encoders.diffusion_gemma._shared_mxfp8_input", return_value=object()),
            mock.patch("comfy.text_encoders.diffusion_gemma._linear_from_shared_input", linear),
            mock.patch("comfy.text_encoders.diffusion_gemma._native_mxfp8_linear", return_value=True),
            mock.patch.object(sys.modules["comfy.quant_ops"].ck, "gelu_tanh_multiply_quantize_mxfp8", return_value=(qdata, scales)) as fused,
        ):
            self.assertEqual(DiffusionGemmaMLP.forward(mlp, x), "output")
        self.assertEqual((fused.call_count, fused.call_args.kwargs), (1, {"pad_32x": False}))
        self.assertIsInstance(linear.call_args_list[2].args[1], QuantizedTensor)

    def test_aimdo_snapshot_rejects_missing_page_before_fault(self):
        residency = [0]
        vbar = types.SimpleNamespace(
            base_addr=4096,
            get_watermark=lambda: 1,
            get_residency=lambda: residency,
            loaded_size=lambda: 32 * 1024 * 1024,
        )
        module = torch.nn.Linear(1, 1, bias=False)
        signature = (ctypes.c_uint32 * 1)(7)
        module._v = (vbar, 4096, 1024)
        module._v_signature = signature
        module._v_weight = torch.ones(1)
        module._v_bias = None
        root_module = types.SimpleNamespace(modules=lambda: [module])
        current_signature = (ctypes.c_uint32 * 1)(7)

        def fault_resident(allocation):
            residency[0] = 3
            return current_signature

        with (
            mock.patch("comfy.memory_management.aimdo_enabled", True),
            mock.patch("comfy_aimdo.model_vbar.vbar_fault", side_effect=fault_resident) as fault,
            mock.patch("comfy_aimdo.model_vbar.vbar_unpin") as unpin,
        ):
            self.assertIsNone(comfy.model_management.acquire_dynamic_vram_graph_lease(
                root_module, "patch"
            ))
            fault.assert_not_called()
            residency[0] = 1
            lease = comfy.model_management.acquire_dynamic_vram_graph_lease(
                root_module, "patch"
            )
            self.assertIsNotNone(lease)
            residency[0] = 3
            self.assertTrue(lease.valid("patch"))
            residency[0] = 0
            self.assertFalse(lease.valid("patch"))
            residency[0] = 3
            module._v_signature = (ctypes.c_uint32 * 1)(7)
            self.assertFalse(lease.valid("patch"))
            lease.release()
            lease.release()

        fault.assert_called_once_with(module._v)
        unpin.assert_called_once_with(module._v)

    def test_aimdo_lease_unwinds_signature_mismatch(self):
        vbar = types.SimpleNamespace(
            base_addr=4096,
            get_watermark=lambda: 1,
            get_residency=lambda: [1],
            loaded_size=lambda: 32 * 1024 * 1024,
        )
        module = torch.nn.Linear(1, 1, bias=False)
        module._v = (vbar, 4096, 1024)
        module._v_signature = (ctypes.c_uint32 * 1)(7)
        module._v_weight = torch.ones(1)
        module._v_bias = None
        root_module = types.SimpleNamespace(modules=lambda: [module])

        with (
            mock.patch("comfy.memory_management.aimdo_enabled", True),
            mock.patch(
                "comfy_aimdo.model_vbar.vbar_fault",
                return_value=(ctypes.c_uint32 * 1)(8),
            ),
            mock.patch("comfy_aimdo.model_vbar.vbar_unpin") as unpin,
        ):
            self.assertIsNone(comfy.model_management.acquire_dynamic_vram_graph_lease(
                root_module, "patch"
            ))

        unpin.assert_called_once_with(module._v)

    def test_kitchen_graph_replays_then_retires_on_invalid_lease(self):
        static_canvas = torch.empty((1, 4), dtype=torch.long)
        static_logits = torch.empty((1, 4, 8), dtype=torch.bfloat16)
        output = torch.empty((1, 4, 2), dtype=torch.bfloat16)
        owner = types.SimpleNamespace(
            model=mock.Mock(return_value=(output, None, None)), _weight_patches_uuid="patch"
        )
        execution = types.SimpleNamespace(
            stream=mock.Mock(), static_canvas=static_canvas, static_logits=static_logits)
        current_stream = mock.Mock()
        graph_exec = mock.Mock()
        residency_lease = mock.Mock()
        residency_lease.valid.side_effect = [True, True, False]
        canvas = torch.ones_like(static_canvas)
        logits = torch.ones_like(static_logits)
        position_ids = torch.arange(4).unsqueeze(0)
        freqs = (torch.ones(1), torch.zeros(1))

        with (
            mock.patch(
                "comfy.model_management.acquire_dynamic_vram_graph_lease",
                return_value=residency_lease,
            ),
            mock.patch.object(torch.cuda, "stream", return_value=contextlib.nullcontext()),
            mock.patch.object(torch.cuda, "current_stream", return_value=current_stream),
            mock.patch.object(sys.modules["comfy.quant_ops"].ck, "begin_cuda_graph_capture", create=True),
            mock.patch.object(sys.modules["comfy.quant_ops"].ck, "end_cuda_graph_capture", create=True, return_value=graph_exec),
        ):
            decoder_graph = _ConditionedDecoderGraph.capture(
                owner, execution, canvas, logits, [], position_ids, freqs, torch.bfloat16)
            graph_exec.replay.assert_not_called()
            self.assertIs(decoder_graph.replay(canvas, logits, [], position_ids, freqs), output)
            self.assertIsNone(decoder_graph.replay(canvas, logits, [], position_ids, freqs))

        graph_exec.replay.assert_called_once_with(execution.stream)
        graph_exec.reset.assert_called_once_with()
        residency_lease.release.assert_called_once_with()
        self.assertTrue(torch.equal(static_canvas, canvas))
        self.assertTrue(torch.equal(static_logits, logits))
        current_stream.wait_stream.assert_called_once_with(execution.stream)

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

    def test_graph_capture_abort_releases_execution_scope(self):
        stream = mock.Mock()
        residency_lease = mock.Mock()
        owner = types.SimpleNamespace(
            model=mock.Mock(side_effect=RuntimeError("sentinel")), _weight_patches_uuid="patch"
        )
        ck = sys.modules["comfy.quant_ops"].ck
        with (
            mock.patch.object(torch.cuda, "Stream", return_value=stream),
            mock.patch.object(torch.cuda, "stream", return_value=contextlib.nullcontext()),
            mock.patch.object(torch.cuda, "current_stream", return_value=mock.Mock()),
            mock.patch(
                "comfy.model_management.acquire_dynamic_vram_graph_lease",
                return_value=residency_lease,
            ),
            mock.patch.object(ck, "reserve_cuda_stream_workspaces", create=True) as reserve,
            mock.patch.object(ck, "release_cuda_stream_workspaces", create=True) as release,
            mock.patch.object(ck, "begin_cuda_graph_capture", create=True) as begin,
            mock.patch.object(ck, "end_cuda_graph_capture", create=True) as end,
            mock.patch.object(ck, "abort_cuda_graph_capture", create=True) as abort,
        ):
            reserve.return_value = True
            release.return_value = True
            execution = _ConditionedDecoderGraphExecution(
                torch.device("cpu"), 1, 4, 8, torch.bfloat16)
            with self.assertRaisesRegex(RuntimeError, "sentinel"):
                execution.capture(
                    owner, torch.ones(1, 4, dtype=torch.long),
                    torch.ones(1, 4, 8, dtype=torch.bfloat16), [],
                    torch.arange(4).unsqueeze(0), (torch.ones(1), torch.zeros(1)), torch.bfloat16)
            execution.close()

        reserve.assert_called_once_with(stream)
        begin.assert_called_once_with(stream)
        abort.assert_called_once_with(stream)
        end.assert_not_called()
        residency_lease.release.assert_called_once_with()
        release.assert_called_once_with(stream)

    def test_split_expert_state_dict_detects_unfused_runtime(self):
        sd = {
            "model.decoder.norm.weight": torch.ones(4, dtype=torch.bfloat16),
            "model.decoder.layers.0.experts.gate_proj.weight": torch.empty(2, 3, 4),
        }

        self.assertTrue(diffusion_gemma_detect((sd,))["unfused_experts"])

    def test_fused_qkv_detection_requires_all_exact_markers(self):
        sd = {
            f"model.decoder.layers.{layer}.self_attn.qkv_proj.comfy_quant": self._qkv_marker(layer)
            for layer in range(30)
        }
        self.assertTrue(diffusion_gemma_detect((sd,))["fused_qkv"])
        sd["model.decoder.layers.5.self_attn.qkv_proj.comfy_quant"] = self._qkv_marker(0)
        with self.assertRaisesRegex(ValueError, "contract for layer 5"):
            diffusion_gemma_detect((sd,))
        del sd["model.decoder.layers.0.self_attn.qkv_proj.comfy_quant"]
        with self.assertRaisesRegex(ValueError, "requires all 30 layer markers"):
            diffusion_gemma_detect((sd,))

    def test_fused_global_qkv_projects_once(self):
        linear = mock.Mock(side_effect=lambda *args, **kwargs: torch.nn.Module())
        attention = DiffusionGemmaAttention(
            DiffusionGemmaConfig(fused_qkv=True), head_dim=512, num_kv_heads=2,
            has_v_proj=False, device="meta", dtype=torch.bfloat16,
            ops=types.SimpleNamespace(Linear=linear),
        )
        self.assertEqual(linear.call_args_list[0].args[:2], (2816, 9216))
        self.assertEqual(linear.call_count, 2)
        self.assertFalse(any(hasattr(attention, name) for name in ("q_proj", "k_proj", "v_proj")))

        module = attention.qkv_proj
        for name, value in vars(self._mxfp8_bank((9216, 2816), (9216, 88))).items():
            setattr(module, name, value)
        module.layout_type = "TensorCoreMXFP8Layout"
        projection = torch.empty((1, 1, 9216), dtype=torch.bfloat16)
        module.forward = mock.Mock(return_value=projection)
        hidden_states = torch.empty((1, 1, 2816), dtype=torch.bfloat16, device="meta")

        q, k, v = attention._project_fused_qkv(hidden_states)

        module.forward.assert_called_once()
        self.assertIs(k, v)
        self.assertEqual((q.shape[-1], k.shape[-1]), (8192, 1024))
        self.assertEqual(q.untyped_storage().data_ptr(), projection.untyped_storage().data_ptr())
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

    @staticmethod
    def _int8_bank(qdata_shape, scale_shape, group_size, device="meta", convrot=True):
        params = TensorWiseINT8Layout.Params(
            scale=torch.empty(scale_shape, dtype=torch.float32, device=device),
            convrot=convrot,
            convrot_groupsize=group_size,
            orig_dtype=torch.bfloat16,
            orig_shape=qdata_shape,
        )
        return types.SimpleNamespace(
            quant_format="int8_tensorwise",
            weight=QuantizedTensor(
                torch.empty(qdata_shape, dtype=torch.int8, device=device),
                "TensorWiseINT8Layout",
                params,
            ),
            bias=None,
            _full_precision_mm=False,
            weight_function=[],
            bias_function=[],
            weight_lowvram_function=None,
            bias_lowvram_function=None,
        )

    @staticmethod
    def _w4a4_bank(qdata_shape, scale_shape, orig_shape, group_size, device="meta", linear_dtype="int4"):
        params = TensorCoreConvRotW4A4Layout.Params(
            scale=torch.empty(scale_shape, dtype=torch.float32, device=device),
            convrot_groupsize=group_size,
            orig_dtype=torch.bfloat16,
            orig_shape=orig_shape,
            linear_dtype=linear_dtype,
        )
        return types.SimpleNamespace(
            quant_format="convrot_w4a4",
            weight=QuantizedTensor(
                torch.empty(qdata_shape, dtype=torch.int8, device=device),
                "TensorCoreConvRotW4A4Layout",
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

    def test_fused_int8_convrot_expert_bank_contract(self):
        experts = DiffusionGemmaExperts.__new__(DiffusionGemmaExperts)
        torch.nn.Module.__init__(experts)
        experts.unfused = False
        experts._bank_mode = None
        experts._banks = (
            self._int8_bank((128, 1408, 2816), (128, 1408, 1), 256),
            self._int8_bank((128, 2816, 704), (128, 2816, 1), 64),
        )

        experts._configure_loaded_banks(experts, None)
        self.assertEqual(experts._bank_mode, "fused_int8_convrot")
        self.assertTrue(experts._grouped_int8_convrot_compatible)

        experts._banks = (
            self._int8_bank((128, 1408, 2816), (128, 1408, 1), 256, convrot=False),
            self._int8_bank((128, 2816, 704), (128, 2816, 1), 64, convrot=False),
        )
        experts._configure_loaded_banks(experts, None)
        self.assertEqual(experts._bank_mode, "quantized")

        experts._banks = (
            self._int8_bank((128, 1408, 2816), (128, 1408, 1), 64),
            self._int8_bank((128, 2816, 704), (128, 2816, 1), 64),
        )
        with self.assertRaisesRegex(ValueError, "INT8 ConvRot expert bank contract mismatch"):
            experts._configure_loaded_banks(experts, None)

    def test_int8_convrot_experts_dispatch_only_packed_routes(self):
        experts = DiffusionGemmaExperts.__new__(DiffusionGemmaExperts)
        torch.nn.Module.__init__(experts)
        experts.num_experts = 2
        resident = lambda weight: types.SimpleNamespace(bank_resident=lambda hidden: contextlib.nullcontext(types.SimpleNamespace(_resident_bank=(weight, None))))  # noqa: E731
        gate_up = self._int8_bank((2, 4, 2), (2, 4, 1), 256, device="cpu").weight
        down = self._int8_bank((2, 2, 2), (2, 2, 1), 64, device="cpu").weight
        experts.gate_up_proj, experts.down_proj = resident(gate_up), resident(down)
        hidden, indices, weights = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.bfloat16), torch.tensor([[1], [0]]), torch.ones((2, 1), dtype=torch.float32)
        with mock.patch.object(sys.modules["comfy.quant_ops"], "grouped_int8_convrot_linear_packed",
                               side_effect=[torch.zeros((2, 4), dtype=torch.bfloat16), hidden.flip(0)]) as packed:
            output = experts._forward_grouped_int8_convrot(hidden, indices, weights)

        self.assertEqual((packed.call_count, packed.call_args_list[0].args[1].tolist()), (2, [0, 1, 2]))
        self.assertTrue(torch.equal(output, hidden))

    def test_w4a4_convrot_expert_bank_contract(self):
        experts = DiffusionGemmaExperts.__new__(DiffusionGemmaExperts)
        torch.nn.Module.__init__(experts)
        experts.unfused = False
        experts._bank_mode = None
        experts._banks = (
            self._w4a4_bank((128, 1408, 1408), (128, 1408), (128, 1408, 2816), 256),
            self._w4a4_bank((128, 2816, 352), (128, 2816), (128, 2816, 704), 64),
        )

        experts._configure_loaded_banks(experts, None)

        self.assertEqual(experts._bank_mode, "fused_w4a4_convrot")
        self.assertTrue(experts._grouped_w4a4_convrot_compatible)

        experts._banks = (
            self._w4a4_bank((128, 1408, 1408), (128, 1408), (128, 1408, 2816), 256, linear_dtype="int8"),
            self._w4a4_bank((128, 2816, 352), (128, 2816), (128, 2816, 704), 64, linear_dtype="int8"),
        )
        experts._configure_loaded_banks(experts, None)

        self.assertEqual(experts._bank_mode, "fused_w4a8_convrot")
        self.assertTrue(experts._grouped_w4a8_convrot_compatible)

    def test_experts_retain_decoder_layer_index(self):
        config = types.SimpleNamespace(num_experts=2, unfused_experts=False, hidden_size=4, moe_intermediate_size=2)
        ops = types.SimpleNamespace(MoEExperts=lambda **kwargs: torch.nn.Linear(1, 1, bias=False))
        experts = DiffusionGemmaExperts(config, layer_index=7, ops=ops)
        self.assertEqual(experts.layer_index, 7)

    def test_thinking_selection_is_request_scoped(self):
        model = DiffusionGemmaClipModel.__new__(DiffusionGemmaClipModel)
        model.execution_device = torch.device("cpu")
        seen, model.transformer = [], types.SimpleNamespace(generate=lambda **kwargs: seen.append((_REQUEST_W4_ACTIVATION_DTYPE.get(), _REQUEST_W4A8_LAYERS.get())))
        with mock.patch("comfy.text_encoders.diffusion_gemma.sd1_clip.SDClipModel.process_tokens", return_value=(torch.zeros(1), None, None, [])):
            with mock.patch.dict(os.environ, {"COMFY_DG_THINKING_W4A8_LAYERS": "1,7"}):
                model.generate(_Gemma4TokenBatch({"gemma4": [[(1, 1.0)]]}, thinking=True), generation_mode="diffusion")
            model.generate([[(1, 1.0)]], generation_mode="diffusion", thinking=False)
            with mock.patch.dict(os.environ, {}, clear=True):
                model.generate(_Gemma4TokenBatch({"gemma4": [[(1, 1.0)]]}, thinking=True), generation_mode="diffusion")
        self.assertEqual(seen, [("int4", frozenset({1, 7})), ("int8", frozenset()), ("int4", frozenset({18}))])
        self.assertIsNone(_REQUEST_W4_ACTIVATION_DTYPE.get())
        self.assertEqual(_REQUEST_W4A8_LAYERS.get(), frozenset())

if __name__ == "__main__":
    unittest.main()
