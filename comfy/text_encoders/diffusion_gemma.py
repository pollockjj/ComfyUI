import contextlib
import json
import math
import torch
import torch.nn as nn
from dataclasses import dataclass
from tqdm import tqdm

from comfy import sd1_clip
import comfy.ops
import comfy.quant_ops
import comfy.utils
import comfy.model_management
from comfy.quant_ops import QuantizedTensor
from comfy.ldm.modules.attention import optimized_attention_for_device
from comfy.rmsnorm import rms_norm
from comfy.text_encoders.llama import RMSNorm, MLP, BaseLlama
from comfy.text_encoders.gemma4 import (
    GEMMA4_VISION_31B_CONFIG,
    Gemma4VisionEncoder,
    Gemma4RMSNormProjector,
    Gemma4SDTokenizer,
    Gemma4Tokenizer,
    Gemma4Model,
    ClippedLinear,
    _apply_rotary_pos_emb,
)

@dataclass
class DiffusionGemmaConfig:
    vocab_size: int = 262144
    hidden_size: int = 2816
    intermediate_size: int = 2112
    num_hidden_layers: int = 30
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    num_global_key_value_heads: int = 2
    head_dim = 256
    global_head_dim = 512
    max_position_embeddings: int = 262144
    rms_norm_eps: float = 1e-6
    rope_theta = [1000000.0, 10000.0]
    partial_rotary_factor: float = 0.25
    sliding_attention = [1024, 1024, 1024, 1024, 1024, False]
    num_experts: int = 128
    top_k_experts: int = 8
    moe_intermediate_size: int = 704
    final_logit_softcapping: float = 30.0
    canvas_length: int = 256
    unfused_experts: bool = False
    fused_qkv: bool = False
    mlp_activation = "gelu_pytorch_tanh"
    qkv_bias = False
    stop_tokens = [1, 106, 50]
    pad_token_id: int = 0
    vision_config = GEMMA4_VISION_31B_CONFIG
    mm_tokens_per_image = 280


def _gelu_tanh(x):
    return torch.nn.functional.gelu(x, approximate="tanh")


def _shared_mxfp8_input(x, modules):
    """Quantize a shared dense input once when every projection can consume MXFP8."""
    if x.requires_grad or x.ndim < 2 or x.dtype not in (torch.float16, torch.bfloat16):
        return None
    for module in modules:
        if not _native_mxfp8_linear(module, x):
            return None
    x_2d = x.reshape(-1, x.shape[-1]) if x.ndim >= 3 else x
    scale = getattr(modules[0], "input_scale", None)
    if scale is not None:
        scale = comfy.model_management.cast_to_device(scale, x.device, None)
    return QuantizedTensor.from_float(x_2d, modules[0].layout_type, scale=scale)


def _native_mxfp8_linear(module, x):
    weight = getattr(module, "weight", None)
    return (
        getattr(module, "quant_format", None) == "mxfp8"
        and getattr(module, "layout_type", None) == "TensorCoreMXFP8Layout"
        and isinstance(weight, QuantizedTensor)
        and weight.device == x.device
        and not getattr(module, "_full_precision_mm", False)
        and not getattr(module, "comfy_force_cast_weights", False)
        and not getattr(module, "weight_function", None)
        and not getattr(module, "bias_function", None)
        and getattr(module, "weight_lowvram_function", None) is None
        and getattr(module, "bias_lowvram_function", None) is None
    )


def _linear_from_shared_input(module, quantized_input, original_shape):
    output = module(quantized_input)
    if len(original_shape) >= 3:
        output = output.reshape((*original_shape[:-1], module.weight.shape[0]))
    return output


_INT8_SELF_CONDITIONING_CHUNK = 65504  # Largest 32-row multiple below CUDA's 65,535 grid-y limit.
_INT8_SM86_RESIDENT_EXPERT_LAYERS = 1
_INT8_SM86_MIN_DEVICE_MEMORY = 24_000_000_000


def _native_mxfp8_embedding(module):
    return (
        getattr(module, "quant_format", None) == "mxfp8"
        and isinstance(module.weight, QuantizedTensor)
        and module.weight._layout_cls == "TensorCoreMXFP8Layout"
        and len(module.weight_function) == 0
        and getattr(module, "weight_lowvram_function", None) is None
        and not getattr(module, "comfy_force_cast_weights", False)
    )


def _native_int8_embedding(module):
    weight = module.weight
    return (
        getattr(module, "quant_format", None) == "int8_tensorwise"
        and isinstance(weight, QuantizedTensor)
        and weight._layout_cls == "TensorWiseINT8Layout"
        and weight._params.convrot is True
        and len(module.weight_function) == 0
        and getattr(module, "weight_lowvram_function", None) is None
        and not getattr(module, "comfy_force_cast_weights", False)
    )


def _int8_self_conditioning(probabilities, weight):
    if probabilities.dtype != torch.bfloat16 or weight._layout_cls != "TensorWiseINT8Layout":
        raise ValueError("DiffusionGemma INT8 self-conditioning requires BF16 probabilities and INT8 weights")
    params = weight._params
    if params.convrot is not True:
        raise ValueError("DiffusionGemma INT8 self-conditioning requires ConvRot weights")

    qdata = weight._qdata
    flat = probabilities.reshape(-1, probabilities.shape[-1])
    output = torch.zeros((flat.shape[0], qdata.shape[1]), device=qdata.device, dtype=torch.float32)
    for start in range(0, qdata.shape[0], _INT8_SELF_CONDITIONING_CHUNK):
        end = min(start + _INT8_SELF_CONDITIONING_CHUNK, qdata.shape[0])
        chunk_params = type(params)(
            scale=params.scale[start:end],
            convrot=True,
            convrot_groupsize=params.convrot_groupsize,
            orig_dtype=params.orig_dtype,
            orig_shape=(end - start, qdata.shape[1]),
        )
        chunk = QuantizedTensor(qdata[start:end], weight._layout_cls, chunk_params).dequantize()
        partial = torch.mm(flat[:, start:end], chunk, out_dtype=torch.float32)
        output.add_(partial)
        del chunk, partial
    return output.reshape(*probabilities.shape[:-1], qdata.shape[1])


class DiffusionGemmaMLP(MLP):
    def forward(self, x):
        quantized_input = _shared_mxfp8_input(x, (self.gate_proj, self.up_proj))
        if quantized_input is None:
            return super().forward(x)
        gate = _linear_from_shared_input(self.gate_proj, quantized_input, x.shape)
        up = _linear_from_shared_input(self.up_proj, quantized_input, x.shape)
        if not _native_mxfp8_linear(self.down_proj, gate):
            return self.down_proj(self.activation(gate) * up)

        gate_2d = gate.reshape(-1, gate.shape[-1])
        up_2d = up.reshape(-1, up.shape[-1])
        padded_shape = comfy.quant_ops.TensorCoreMXFP8Layout.get_padded_shape(
            tuple(gate_2d.shape))
        qdata, block_scale = comfy.quant_ops.ck.gelu_tanh_multiply_quantize_mxfp8(
            gate_2d, up_2d, pad_32x=padded_shape != tuple(gate_2d.shape))
        params = comfy.quant_ops.TensorCoreMXFP8Layout.Params(
            scale=block_scale,
            orig_dtype=gate.dtype,
            orig_shape=tuple(gate_2d.shape),
        )
        quantized_product = QuantizedTensor(
            qdata, "TensorCoreMXFP8Layout", params)
        return _linear_from_shared_input(
            self.down_proj, quantized_product, gate.shape)


def _make_dg_scaled_embedding(ops, vocab_size, hidden_size, device, dtype):
    # Reference casts sqrt(hidden_size) to the weight dtype before multiplying.
    class ScaledEmbedding(ops.Embedding):
        def _embedding_scale(self, output):
            scale_dtype = self.weight._params.orig_dtype if isinstance(self.weight, QuantizedTensor) else self.weight.dtype
            return torch.tensor(hidden_size ** 0.5, dtype=scale_dtype, device=output.device)

        def forward(self, input_ids, out_dtype=None):
            out = super().forward(input_ids, out_dtype=out_dtype)
            return out * self._embedding_scale(out)

        def weighted_embedding(self, probabilities):
            out = super().weighted_embedding(probabilities)
            return out * self._embedding_scale(out)
    return ScaledEmbedding(vocab_size, hidden_size, device=device, dtype=dtype)


def _append_preallocated_kv(past_key, past_value, xk, xv, cache_len):
    capacity = past_key.shape[2]
    next_cache_len = cache_len + xk.shape[2]
    if past_value.shape[2] != capacity or cache_len < 0 or cache_len > capacity:
        raise RuntimeError("DiffusionGemma KV cache metadata is invalid")
    if next_cache_len > capacity:
        return None
    past_key[:, :, cache_len:next_cache_len].copy_(xk)
    past_value[:, :, cache_len:next_cache_len].copy_(xv)
    return past_key[:, :, :next_cache_len], past_value[:, :, :next_cache_len], next_cache_len


class DiffusionGemmaAttention(nn.Module):
    def __init__(self, config, head_dim, num_kv_heads, has_v_proj, device=None, dtype=None, ops=None):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.inner_size = self.num_heads * head_dim
        self.hidden_size = config.hidden_size
        self.fused_qkv = config.fused_qkv

        kv_size = num_kv_heads * head_dim
        if self.fused_qkv:
            self.qkv_splits = (self.inner_size, kv_size, kv_size) if has_v_proj else (self.inner_size, kv_size)
            self.qkv_proj = ops.Linear(config.hidden_size, sum(self.qkv_splits), bias=config.qkv_bias, device=device, dtype=dtype)
        else:
            self.q_proj = ops.Linear(config.hidden_size, self.inner_size, bias=config.qkv_bias, device=device, dtype=dtype)
            self.k_proj = ops.Linear(config.hidden_size, kv_size, bias=config.qkv_bias, device=device, dtype=dtype)
            if has_v_proj:
                self.v_proj = ops.Linear(config.hidden_size, kv_size, bias=config.qkv_bias, device=device, dtype=dtype)
            else:
                self.v_proj = None
        self.o_proj = ops.Linear(self.inner_size, config.hidden_size, bias=False, device=device, dtype=dtype)

        self.q_norm = RMSNorm(head_dim, eps=config.rms_norm_eps, device=device, dtype=dtype)
        self.k_norm = RMSNorm(head_dim, eps=config.rms_norm_eps, device=device, dtype=dtype)

    def _project_fused_qkv(self, hidden_states):
        module = self.qkv_proj
        weight = getattr(module, "weight", None)
        qdata = getattr(weight, "_qdata", None)
        params = getattr(weight, "_params", None)
        scale = getattr(params, "scale", None)
        expected_shape = (sum(self.qkv_splits), self.hidden_size)
        quant_format = getattr(module, "quant_format", None)
        mxfp8 = (
            quant_format == "mxfp8"
            and getattr(module, "layout_type", None) == "TensorCoreMXFP8Layout"
            and getattr(weight, "_layout_cls", None) == "TensorCoreMXFP8Layout"
            and isinstance(qdata, torch.Tensor)
            and qdata.dtype == torch.float8_e4m3fn
            and isinstance(scale, torch.Tensor)
            and scale.dtype == torch.float8_e8m0fnu
            and tuple(scale.shape) == (expected_shape[0], (expected_shape[1] + 31) // 32)
        )
        int8_convrot = (
            quant_format == "int8_tensorwise"
            and getattr(module, "layout_type", None) == "TensorWiseINT8Layout"
            and getattr(weight, "_layout_cls", None) == "TensorWiseINT8Layout"
            and isinstance(qdata, torch.Tensor)
            and qdata.dtype == torch.int8
            and isinstance(scale, torch.Tensor)
            and scale.dtype == torch.float32
            and tuple(scale.shape) == (expected_shape[0], 1)
            and getattr(params, "convrot", False) is True
            and getattr(params, "convrot_groupsize", None) == 256
        )
        # Validate storage only; the linear module owns DynamicVRAM device placement
        # and patch application.
        if (
            not (mxfp8 or int8_convrot)
            or not isinstance(weight, QuantizedTensor)
            or tuple(qdata.shape) != expected_shape
            or tuple(getattr(params, "orig_shape", ())) != expected_shape
            or getattr(module, "bias", None) is not None
            or getattr(module, "_full_precision_mm", False)
            or getattr(module, "comfy_force_cast_weights", False)
        ):
            raise RuntimeError(
                f"DiffusionGemma fused QKV requires a native MXFP8 or INT8 ConvRot "
                f"weight with shape {expected_shape}"
            )

        projections = torch.split(module(hidden_states), self.qkv_splits, dim=-1)
        if len(projections) == 2:
            return projections[0], projections[1], projections[1]
        return projections

    def forward(self, hidden_states, attention_mask=None, freqs_cis=None, past_key_value=None,
                sliding_window=None, update_cache=True):
        batch_size, seq_length, _ = hidden_states.shape

        if self.fused_qkv:
            xq, xk, xv = self._project_fused_qkv(hidden_states)
        else:
            qkv_modules = (self.q_proj, self.k_proj) if self.v_proj is None else (self.q_proj, self.k_proj, self.v_proj)
            quantized_input = _shared_mxfp8_input(hidden_states, qkv_modules)

            def project(module):
                if quantized_input is None:
                    return module(hidden_states)
                return _linear_from_shared_input(module, quantized_input, hidden_states.shape)

            xq = project(self.q_proj)
            xk = project(self.k_proj)
            xv = xk if self.v_proj is None else project(self.v_proj)

        xq = xq.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        xq = self.q_norm(xq)
        xq = _apply_rotary_pos_emb(xq, freqs_cis)

        xk = xk.view(batch_size, seq_length, self.num_kv_heads, self.head_dim)
        xv = xv.view(batch_size, seq_length, self.num_kv_heads, self.head_dim)
        xk = self.k_norm(xk)
        xv = rms_norm(xv)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)
        xk = _apply_rotary_pos_emb(xk, freqs_cis)

        present_key_value = None
        if past_key_value is not None:
            preallocated_cache = False
            if len(past_key_value) > 0:
                if len(past_key_value) == 4:
                    past_key, past_value, cumulative_len, cache_len = past_key_value
                    appended = _append_preallocated_kv(past_key, past_value, xk, xv, cache_len)
                    if appended is not None:
                        xk, xv, next_cache_len = appended
                        preallocated_cache = True
                    else:
                        xk = torch.cat((past_key[:, :, :cache_len], xk), dim=2)
                        xv = torch.cat((past_value[:, :, :cache_len], xv), dim=2)
                else:
                    past_key, past_value, cumulative_len = past_key_value
                    xk = torch.cat((past_key, xk), dim=2)
                    xv = torch.cat((past_value, xv), dim=2)
            else:
                cumulative_len = 0
            if update_cache:
                new_cumulative = cumulative_len + seq_length
                if preallocated_cache:
                    present_key_value = (past_key, past_value, new_cumulative, next_cache_len)
                elif sliding_window is not None and xk.shape[2] > sliding_window - 1:
                    present_key_value = (
                        xk[:, :, -(sliding_window - 1):].clone(),
                        xv[:, :, -(sliding_window - 1):].clone(),
                        new_cumulative,
                    )
                else:
                    present_key_value = (xk, xv, new_cumulative)

        expand_kv = (self.num_heads != self.num_kv_heads and
                     attention_mask is not None and
                     sliding_window is not None and
                     xk.shape[2] >= sliding_window)
        if expand_kv:
            xk = xk.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
            xv = xv.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
        gqa_kwargs = {} if expand_kv else ({"enable_gqa": True} if self.num_heads != self.num_kv_heads else {})
        output = optimized_attention_for_device(xq.device, mask=attention_mask is not None, small_input=True)(
            xq, xk, xv, self.num_heads, mask=attention_mask, skip_reshape=True, scale=1.0, **gqa_kwargs)

        return self.o_proj(output), present_key_value


class DiffusionGemmaRouter(nn.Module):
    def __init__(self, config, device=None, dtype=None, ops=None):
        super().__init__()
        self.top_k = config.top_k_experts
        self.scalar_root_size = config.hidden_size ** -0.5
        self.proj = ops.Linear(config.hidden_size, config.num_experts, bias=False, device=device, dtype=dtype)
        self.scale = nn.Parameter(torch.empty(config.hidden_size, device=device, dtype=dtype))
        self.per_expert_scale = nn.Parameter(torch.empty(config.num_experts, device=device, dtype=dtype))

    def forward(self, hidden_states, apply_expert_scale=True):
        hidden_states = rms_norm(hidden_states)
        scale = comfy.ops.cast_to_input(self.scale, hidden_states, copy=False)
        hidden_states = hidden_states * scale * self.scalar_root_size

        expert_scores = self.proj(hidden_states)
        router_probabilities = torch.nn.functional.softmax(expert_scores, dim=-1, dtype=torch.float32)
        top_k_weights, top_k_index = torch.topk(router_probabilities, k=self.top_k, dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        if apply_expert_scale:
            per_expert_scale = comfy.ops.cast_to_input(self.per_expert_scale, expert_scores, copy=False)
            top_k_weights = top_k_weights * per_expert_scale[top_k_index]
        return top_k_weights, top_k_index


def _dequant_bank(module, weight, dtype):
    # Chunk over experts to cap the fp32 transient while the allocator reuses identical layer shapes.
    qdata = weight._qdata
    params = weight._params
    E = qdata.shape[0]
    scale = params.scale
    if scale.dim() == 0:
        scale = scale.expand(E)
    out_f = module.out_features
    in_f = module.in_features
    if hasattr(params, "block_scale"):  # block-scaled (nvfp4)
        block_scale = params.block_scale
        if qdata.shape[1] == block_scale.shape[1]:
            # row-padded bank: tile-local swizzle makes one flat dequant valid
            rows = qdata.shape[1]
            if params.scale.dim() == 0:
                # shared bank scale: single kernel call straight to compute dtype
                flat_params = type(params)(scale=params.scale, orig_dtype=dtype, orig_shape=(E * rows, in_f),
                                           block_scale=block_scale.reshape(E * rows, -1))
                w = weight.layout_cls.dequantize(qdata.reshape(E * rows, -1), flat_params)
                return w.view(E, rows, in_f)[:, :out_f]
            unit = torch.ones((), device=qdata.device, dtype=torch.float32)
            flat_params = type(params)(scale=unit, orig_dtype=torch.float32, orig_shape=(E * rows, in_f),
                                       block_scale=block_scale.reshape(E * rows, -1))
            w = weight.layout_cls.dequantize(qdata.reshape(E * rows, -1), flat_params)
            w = (w.view(E, rows, in_f)[:, :out_f] * scale.view(-1, 1, 1)).to(dtype)
            return w
        w = torch.empty((E, out_f, in_f), dtype=dtype, device=qdata.device)
        for i in range(E):
            expert_params = type(params)(scale=scale[i], orig_dtype=dtype, orig_shape=(out_f, in_f),
                                         block_scale=block_scale[i])
            w[i] = weight.layout_cls.dequantize(qdata[i], expert_params)
        return w
    w = torch.empty((E, out_f, in_f), dtype=dtype, device=qdata.device)
    chunk = 32
    for i in range(0, E, chunk):
        w[i:i + chunk] = (qdata[i:i + chunk].to(torch.float32) * scale[i:i + chunk].view(-1, 1, 1)).to(dtype)
    return w


def _bank_bmm(module, x):
    # Multiply x [E, C, in] by one [E, out, in] bank at a time to cap the dequant transient.
    weight, bias, offload_stream = comfy.ops.cast_bias_weight(module, x, offloadable=True)
    try:
        w = weight
        if isinstance(w, QuantizedTensor):
            w = _dequant_bank(module, w, x.dtype)
        out = torch.bmm(x, w.transpose(1, 2))
        if bias is not None:
            out = out + bias.unsqueeze(1)
        return out
    finally:
        comfy.ops.uncast_bias_weight(module, weight, bias, offload_stream)


class DiffusionGemmaExperts(nn.Module):
    grouped_bucket = 64
    grouped_nvfp4_bucket = 128
    grouped_mxfp8_bucket = 128
    grouped_min_tokens = 64
    fused_nvfp4_format = comfy.quant_ops.NVFP4_FUSED_MOE_FORMAT
    fused_mxfp8_format = comfy.quant_ops.MXFP8_FUSED_MOE_FORMAT

    def __init__(self, config, device=None, dtype=None, ops=None):
        super().__init__()
        self.num_experts = config.num_experts
        self.unfused = config.unfused_experts
        self._weight_patches_uuid = None
        self._native_nvfp4_alpha_cache = None
        self._bank_mode = None
        self._fused_banks_compatible = False
        self._fused_mxfp8_banks_compatible = False
        self._grouped_mxfp8_compatible = False
        self._grouped_int8_convrot_compatible = False
        E = config.num_experts
        H = config.hidden_size
        I = config.moe_intermediate_size
        if self.unfused:
            self.gate_proj = ops.MoEExperts(num_experts=E, in_features=H, out_features=I, bias=False, device=device, dtype=dtype)
            self.up_proj = ops.MoEExperts(num_experts=E, in_features=H, out_features=I, bias=False, device=device, dtype=dtype)
        else:
            self.gate_up_proj = ops.MoEExperts(num_experts=E, in_features=H, out_features=2 * I, bias=False, device=device, dtype=dtype)
        self.down_proj = ops.MoEExperts(num_experts=E, in_features=I, out_features=H, bias=False, device=device, dtype=dtype)
        self._banks = (
            (self.gate_proj, self.up_proj, self.down_proj)
            if self.unfused else (self.gate_up_proj, self.down_proj)
        )
        self.register_load_state_dict_post_hook(self._configure_loaded_banks)

    def _configure_loaded_banks(self, module, incompatible_keys):
        formats = tuple(bank.quant_format for bank in self._banks)
        int8_convrot = tuple(
            quant_format == "int8_tensorwise"
            and isinstance(bank.weight, QuantizedTensor)
            and getattr(getattr(bank.weight, "_params", None), "convrot", False) is True
            for bank, quant_format in zip(self._banks, formats)
        )
        if self.unfused:
            if all(quant_format == "nvfp4" for quant_format in formats):
                if not all(isinstance(bank.weight, QuantizedTensor) and bank.bias is None for bank in self._banks):
                    raise ValueError("DiffusionGemma NVFP4 expert banks are incomplete")
                self._bank_mode = "unfused_nvfp4"
            elif all(quant_format == "mxfp8" for quant_format in formats):
                expected = (
                    ((128, 704, 2816), (128, 768, 88)),
                    ((128, 704, 2816), (128, 768, 88)),
                    ((128, 2816, 704), (128, 2816, 24)),
                )
                for bank, (qdata_shape, scale_shape) in zip(self._banks, expected):
                    weight = bank.weight
                    if (
                        not isinstance(weight, QuantizedTensor)
                        or weight._layout_cls != "TensorCoreMXFP8Layout"
                        or bank.bias is not None
                        or weight._qdata.dtype != torch.float8_e4m3fn
                        or tuple(weight._qdata.shape) != qdata_shape
                        or weight._params.scale.dtype != torch.float8_e8m0fnu
                        or tuple(weight._params.scale.shape) != scale_shape
                        or tuple(weight._params.orig_shape) != qdata_shape
                    ):
                        raise ValueError("DiffusionGemma MXFP8 expert bank contract mismatch")
                self._bank_mode = "unfused_mxfp8"
            elif any(quant_format is not None for quant_format in formats):
                self._bank_mode = "quantized"
            else:
                self._bank_mode = "unquantized"
            self._refresh_bank_compatibility()
            return

        if any(quant_format == self.fused_nvfp4_format for quant_format in formats):
            if not all(quant_format == self.fused_nvfp4_format for quant_format in formats):
                raise ValueError("DiffusionGemma fused NVFP4 requires both expert banks")
            gate_up, down = self._banks
            if not all(
                isinstance(bank.weight, QuantizedTensor)
                and bank.bias is None
                and isinstance(bank.input_scale, torch.Tensor)
                and bank.input_scale.numel() == 1
                for bank in self._banks
            ):
                raise ValueError("DiffusionGemma fused NVFP4 banks are incomplete")
            if (
                tuple(gate_up.weight._qdata.shape) != (128, 1408, 1408)
                or tuple(down.weight._qdata.shape) != (128, 2816, 352)
            ):
                raise ValueError("DiffusionGemma fused NVFP4 bank shape mismatch")
            self._bank_mode = "fused_nvfp4"
        elif any(quant_format == self.fused_mxfp8_format for quant_format in formats):
            if not all(quant_format == self.fused_mxfp8_format for quant_format in formats):
                raise ValueError("DiffusionGemma fused MXFP8 requires both expert banks")
            expected = (
                ((128, 1408, 2816), (128, 1408, 88)),
                ((128, 2816, 704), (128, 2816, 24)),
            )
            for bank, (qdata_shape, scale_shape) in zip(self._banks, expected):
                weight = bank.weight
                if (
                    not isinstance(weight, QuantizedTensor)
                    or weight._layout_cls != "TensorCoreMXFP8Layout"
                    or bank.bias is not None
                    or weight._qdata.dtype != torch.float8_e4m3fn
                    or tuple(weight._qdata.shape) != qdata_shape
                    or weight._params.scale.dtype != torch.float8_e8m0fnu
                    or tuple(weight._params.scale.shape) != scale_shape
                    or tuple(weight._params.orig_shape) != qdata_shape
                ):
                    raise ValueError("DiffusionGemma fused MXFP8 expert bank contract mismatch")
            self._bank_mode = "fused_mxfp8"
        elif any(int8_convrot):
            if not all(int8_convrot):
                raise ValueError("DiffusionGemma INT8 ConvRot requires both expert banks")
            expected = (
                ((128, 1408, 2816), (128, 1408, 1), 256),
                ((128, 2816, 704), (128, 2816, 1), 64),
            )
            for bank, (qdata_shape, scale_shape, group_size) in zip(self._banks, expected):
                weight = bank.weight
                params = getattr(weight, "_params", None)
                if (
                    not isinstance(weight, QuantizedTensor)
                    or weight._layout_cls != "TensorWiseINT8Layout"
                    or bank.bias is not None
                    or weight._qdata.dtype != torch.int8
                    or tuple(weight._qdata.shape) != qdata_shape
                    or not weight._qdata.is_contiguous()
                    or params.scale.dtype != torch.float32
                    or tuple(params.scale.shape) != scale_shape
                    or not params.scale.is_contiguous()
                    or params.convrot is not True
                    or params.convrot_groupsize != group_size
                    or tuple(params.orig_shape) != qdata_shape
                ):
                    raise ValueError("DiffusionGemma INT8 ConvRot expert bank contract mismatch")
            self._bank_mode = "fused_int8_convrot"
        elif any(quant_format is not None for quant_format in formats):
            self._bank_mode = "quantized"
        else:
            self._bank_mode = "unquantized"
        self._refresh_bank_compatibility()

    def set_weight_patches_uuid(self, patches_uuid):
        self._weight_patches_uuid = patches_uuid
        self._refresh_bank_compatibility()

    def _refresh_bank_compatibility(self):
        self._fused_banks_compatible = self._bank_mode == "fused_nvfp4" and all(
            isinstance(bank.weight, QuantizedTensor)
            and not bank.weight_function
            and not bank.bias_function
            and bank.weight_lowvram_function is None
            and bank.bias_lowvram_function is None
            for bank in self._banks
        )
        self._fused_mxfp8_banks_compatible = self._bank_mode == "fused_mxfp8" and all(
            isinstance(bank.weight, QuantizedTensor)
            and not bank.weight_function
            and not bank.bias_function
            and bank.weight_lowvram_function is None
            and bank.bias_lowvram_function is None
            for bank in self._banks
        )
        self._grouped_mxfp8_compatible = self._bank_mode == "unfused_mxfp8" and all(
            bank._full_precision_mm is False
            and not bank.weight_function
            and not bank.bias_function
            and bank.weight_lowvram_function is None
            and bank.bias_lowvram_function is None
            for bank in self._banks
        )
        self._grouped_int8_convrot_compatible = self._bank_mode == "fused_int8_convrot" and all(
            isinstance(bank.weight, QuantizedTensor)
            and bank._full_precision_mm is False
            and not bank.weight_function
            and not bank.bias_function
            and bank.weight_lowvram_function is None
            and bank.bias_lowvram_function is None
            for bank in self._banks
        )

    @staticmethod
    def _grouped_nvfp4_mm(
        qdata,
        weight,
        input_scale,
        weight_scale,
        input_block_scale,
        weight_block_scale,
        group_size,
        out_dtype,
    ):
        ck = comfy.quant_ops.ck
        return ck.grouped_scaled_mm_nvfp4(
            qdata,
            weight,
            input_scale,
            weight_scale,
            input_block_scale,
            weight_block_scale,
            group_size,
            out_dtype=out_dtype,
        )

    @staticmethod
    def _grouped_mxfp8_mm(qdata, weight, input_block_scale, weight_block_scale, group_size, out_dtype):
        return comfy.quant_ops.ck.grouped_scaled_mm_mxfp8(
            qdata,
            weight,
            input_block_scale,
            weight_block_scale,
            group_size,
            out_dtype=out_dtype,
        )

    def _supports_sm120_nvfp4(self, hidden_states):
        return (
            hidden_states.is_cuda
            and torch.cuda.get_device_capability(hidden_states.device) == (12, 0)
        )

    def _supports_native_fused_nvfp4(self, hidden_states):
        return (
            self._supports_sm120_nvfp4(hidden_states)
            and hidden_states.dtype == torch.bfloat16
            and hidden_states.ndim == 2
            and hidden_states.shape[0] in (256, 340)
            and hidden_states.shape[1] == 2816
            and hidden_states.is_contiguous()
        )

    def _supports_native_fused_mxfp8(self, hidden_states):
        return (
            hidden_states.is_cuda
            and torch.cuda.get_device_capability(hidden_states.device) == (12, 0)
            and hidden_states.dtype in (torch.float16, torch.bfloat16)
            and hidden_states.ndim == 2
            and hidden_states.shape[0] in (256, 340)
            and hidden_states.shape[1] == 2816
            and hidden_states.is_contiguous()
        )

    def _supports_native_fused_mxfp8_scaled(self, hidden_states):
        return (
            self._bank_mode == "fused_mxfp8"
            and self._fused_mxfp8_banks_compatible
            and hidden_states.dtype == torch.bfloat16
            and self._supports_native_fused_mxfp8(hidden_states)
        )

    def forward(self, hidden_states, top_k_index, top_k_weights, expert_scale=None):
        if self._bank_mode is None:
            raise RuntimeError("DiffusionGemma expert banks were not configured after loading")
        if self._bank_mode == "fused_int8_convrot":
            if not self._grouped_int8_convrot_compatible:
                raise RuntimeError("DiffusionGemma INT8 ConvRot does not support patched expert banks")
            return self._forward_grouped_int8_convrot(hidden_states, top_k_index, top_k_weights)
        if self._bank_mode == "fused_nvfp4":
            if not self._fused_banks_compatible:
                raise RuntimeError("DiffusionGemma fused NVFP4 does not support patched expert banks")
            if self._supports_native_fused_nvfp4(hidden_states):
                try:
                    return self._forward_native_fused_nvfp4(hidden_states, top_k_index, top_k_weights)
                except comfy.quant_ops.ck.NoCapableBackendError:
                    pass
            if not self._supports_sm120_nvfp4(hidden_states):
                raise RuntimeError(
                    "DiffusionGemma fused NVFP4 v1 requires complete calibrated banks and the SM120 grouped kernel"
                )
            return self._forward_grouped_fused_nvfp4(hidden_states, top_k_index, top_k_weights)
        if self._bank_mode == "fused_mxfp8":
            if not self._fused_mxfp8_banks_compatible:
                raise RuntimeError("DiffusionGemma fused MXFP8 does not support patched expert banks")
            if self._supports_native_fused_mxfp8(hidden_states):
                try:
                    return self._forward_native_fused_mxfp8(
                        hidden_states, top_k_index, top_k_weights, expert_scale=expert_scale
                    )
                except comfy.quant_ops.ck.NoCapableBackendError:
                    pass
            if expert_scale is not None:
                top_k_weights = top_k_weights * expert_scale[top_k_index]
            if not self._supports_sm120_nvfp4(hidden_states):
                raise RuntimeError("DiffusionGemma fused MXFP8 v1 requires the SM120 grouped kernel")
            return self._forward_grouped_fused_mxfp8(hidden_states, top_k_index, top_k_weights)
        if (
            self._bank_mode == "unfused_nvfp4"
            and hidden_states.shape[0] >= self.grouped_min_tokens
            and self._supports_sm120_nvfp4(hidden_states)
        ):
            try:
                return self._forward_grouped_nvfp4(hidden_states, top_k_index, top_k_weights)
            except comfy.quant_ops.ck.NoCapableBackendError:
                pass
        if (
            self._grouped_mxfp8_compatible
            and hidden_states.shape[0] >= self.grouped_min_tokens
        ):
            try:
                return self._forward_grouped_mxfp8(hidden_states, top_k_index, top_k_weights)
            except comfy.quant_ops.ck.NoCapableBackendError:
                pass
        if hidden_states.shape[0] >= self.grouped_min_tokens and self._bank_mode == "unquantized":
            return self._forward_grouped(hidden_states, top_k_index, top_k_weights)
        return self._forward_loop(hidden_states, top_k_index, top_k_weights)

    def _forward_native_fused_nvfp4(self, hidden_states, top_k_index, top_k_weights):
        ck = comfy.quant_ops.ck
        num_tokens = hidden_states.shape[0]
        if (
            hidden_states.dtype != torch.bfloat16
            or hidden_states.shape[1:] != (2816,)
            or num_tokens not in (256, 340)
        ):
            raise RuntimeError(
                "DiffusionGemma native NVFP4 MoE requires BF16 hidden states [256|340, 2816]"
            )
        if not hidden_states.is_contiguous():
            raise RuntimeError("DiffusionGemma native NVFP4 hidden states must be contiguous")
        if tuple(top_k_index.shape) != (num_tokens, 8) or top_k_index.device != hidden_states.device:
            raise RuntimeError("DiffusionGemma native NVFP4 expert indices must be [N, 8] on the input device")
        if tuple(top_k_weights.shape) != (num_tokens, 8) or top_k_weights.device != hidden_states.device:
            raise RuntimeError("DiffusionGemma native NVFP4 expert weights must be [N, 8] on the input device")
        if top_k_weights.dtype != torch.float32 or not top_k_weights.is_contiguous():
            raise RuntimeError("DiffusionGemma native NVFP4 expert weights must be contiguous FP32")

        with contextlib.ExitStack() as stack:
            modules = (self.gate_up_proj, self.down_proj)
            banks = [stack.enter_context(module.bank_resident(hidden_states)) for module in modules]
            weights = []
            for module, bank in zip(modules, banks):
                weight, bias = bank._resident_bank
                if (
                    module.quant_format != self.fused_nvfp4_format
                    or not isinstance(weight, QuantizedTensor)
                    or bias is not None
                ):
                    raise RuntimeError("DiffusionGemma native NVFP4 requires complete unbiased banks")
                weights.append(weight)

            gate_up_weight, down_weight = weights
            gate_up_qdata = gate_up_weight._qdata
            down_qdata = down_weight._qdata
            if (
                gate_up_qdata.dtype != torch.uint8
                or tuple(gate_up_qdata.shape) != (128, 1408, 1408)
                or not gate_up_qdata.is_contiguous()
                or down_qdata.dtype != torch.uint8
                or tuple(down_qdata.shape) != (128, 2816, 352)
                or not down_qdata.is_contiguous()
            ):
                raise RuntimeError("DiffusionGemma native fused NVFP4 qdata contract mismatch")

            gate_up_params = gate_up_weight._params
            down_params = down_weight._params
            if (
                tuple(gate_up_params.scale.shape) != (128,)
                or gate_up_params.scale.dtype != torch.float32
                or tuple(gate_up_params.block_scale.shape) != (128, 1408, 176)
                or not gate_up_params.block_scale.is_contiguous()
                or tuple(down_params.scale.shape) != (128,)
                or down_params.scale.dtype != torch.float32
                or tuple(down_params.block_scale.shape) != (128, 2816, 44)
                or not down_params.block_scale.is_contiguous()
            ):
                raise RuntimeError("DiffusionGemma native fused NVFP4 scale contract mismatch")

            gate_up_input_scale = self.gate_up_proj.input_scale.to(
                device=hidden_states.device, dtype=torch.float32
            )
            down_input_scale = self.down_proj.input_scale.to(
                device=hidden_states.device, dtype=torch.float32
            )
            if gate_up_input_scale.numel() != 1 or down_input_scale.numel() != 1:
                raise RuntimeError("DiffusionGemma native NVFP4 requires scalar activation scales")

            alpha_sources = (
                gate_up_params.scale,
                down_params.scale,
                self.gate_up_proj.input_scale,
                self.down_proj.input_scale,
            )
            cache_key = (
                hidden_states.device,
                self._weight_patches_uuid,
                tuple(id(tensor) for tensor in alpha_sources),
                tuple(None if tensor.is_inference() else tensor._version for tensor in alpha_sources),
            )
            cached = self._native_nvfp4_alpha_cache
            if cached is None or cached[0] != cache_key:
                alphas = (
                    (gate_up_params.scale * gate_up_input_scale).contiguous(),
                    (down_params.scale * down_input_scale).contiguous(),
                )
                self._native_nvfp4_alpha_cache = (cache_key, alpha_sources, alphas)
            else:
                alphas = cached[2]

            return ck.fused_moe_nvfp4(
                hidden_states,
                top_k_index,
                top_k_weights,
                gate_up_qdata,
                gate_up_params.block_scale,
                down_qdata,
                down_params.block_scale,
                gate_up_input_scale,
                down_input_scale,
                alphas[0],
                alphas[1],
            )

    def _forward_native_fused_mxfp8(
        self, hidden_states, top_k_index, top_k_weights, expert_scale=None
    ):
        ck = comfy.quant_ops.ck
        num_tokens = hidden_states.shape[0]
        if (
            hidden_states.dtype not in (torch.float16, torch.bfloat16)
            or hidden_states.shape[1:] != (2816,)
            or num_tokens not in (256, 340)
            or not hidden_states.is_contiguous()
        ):
            raise RuntimeError("DiffusionGemma native MXFP8 MoE requires contiguous FP16/BF16 [256|340, 2816]")
        if tuple(top_k_index.shape) != (num_tokens, 8) or top_k_index.device != hidden_states.device:
            raise RuntimeError("DiffusionGemma native MXFP8 expert indices must be [N, 8] on the input device")
        if (
            tuple(top_k_weights.shape) != (num_tokens, 8)
            or top_k_weights.device != hidden_states.device
            or top_k_weights.dtype != torch.float32
            or not top_k_weights.is_contiguous()
        ):
            raise RuntimeError("DiffusionGemma native MXFP8 expert weights must be contiguous FP32 [N, 8]")

        with contextlib.ExitStack() as stack:
            modules = (self.gate_up_proj, self.down_proj)
            banks = [stack.enter_context(module.bank_resident(hidden_states)) for module in modules]
            weights = []
            for module, bank in zip(modules, banks):
                weight, bias = bank._resident_bank
                if (
                    module.quant_format != self.fused_mxfp8_format
                    or not isinstance(weight, QuantizedTensor)
                    or weight._layout_cls != "TensorCoreMXFP8Layout"
                    or bias is not None
                ):
                    raise RuntimeError("DiffusionGemma native MXFP8 requires complete unbiased banks")
                weights.append(weight)

            gate_up, down = weights
            if (
                tuple(gate_up._qdata.shape) != (128, 1408, 2816)
                or tuple(gate_up._params.scale.shape) != (128, 1408, 88)
                or tuple(down._qdata.shape) != (128, 2816, 704)
                or tuple(down._params.scale.shape) != (128, 2816, 24)
            ):
                raise RuntimeError("DiffusionGemma native fused MXFP8 bank shape mismatch")
            if expert_scale is not None:
                if (
                    expert_scale.dtype != torch.bfloat16
                    or expert_scale.shape != (128,)
                    or expert_scale.device != hidden_states.device
                    or not expert_scale.is_contiguous()
                ):
                    raise RuntimeError("DiffusionGemma native MXFP8 expert scale must be contiguous BF16 [128]")
                return ck.fused_moe_mxfp8_scaled(
                    hidden_states,
                    top_k_index,
                    top_k_weights,
                    expert_scale,
                    gate_up._qdata,
                    gate_up._params.scale,
                    down._qdata,
                    down._params.scale,
                )
            return ck.fused_moe_mxfp8(
                hidden_states,
                top_k_index,
                top_k_weights,
                gate_up._qdata,
                gate_up._params.scale,
                down._qdata,
                down._params.scale,
            )

    def _forward_grouped_nvfp4(self, hidden_states, top_k_index, top_k_weights):
        ck = comfy.quant_ops.ck
        N, H = hidden_states.shape
        E = self.num_experts
        K = top_k_index.shape[-1]

        flat_experts = top_k_index.reshape(-1)
        counts = torch.bincount(flat_experts, minlength=E)
        C = -(-int(counts.max()) // self.grouped_nvfp4_bucket) * self.grouped_nvfp4_bucket
        order = torch.argsort(flat_experts)
        sorted_experts = flat_experts[order]
        rank = torch.arange(N * K, device=flat_experts.device) - (counts.cumsum(0) - counts)[sorted_experts]
        slot = sorted_experts * C + rank

        gather_tok = torch.zeros(E * C, dtype=torch.long, device=flat_experts.device)
        gather_tok[slot] = order // K
        x = hidden_states[gather_tok].view(E, C, H)

        def quantize(tensor):
            flat = tensor.flatten(0, 1)
            scale = (torch.amax(flat.abs()) / (448.0 * 6.0)).to(torch.float32)
            qdata, block_scale = ck.quantize_nvfp4(flat, scale)
            return qdata, scale, block_scale

        def grouped_linear(qdata, input_scale, input_block_scale, weight):
            params = weight._params
            return self._grouped_nvfp4_mm(
                qdata,
                weight._qdata,
                input_scale,
                params.scale,
                input_block_scale,
                params.block_scale,
                C,
                out_dtype=hidden_states.dtype,
            )

        with contextlib.ExitStack() as stack:
            modules = (self.gate_proj, self.up_proj, self.down_proj)
            banks = [stack.enter_context(module.bank_resident(hidden_states)) for module in modules]
            weights = []
            for bank in banks:
                weight, bias = bank._resident_bank
                if not isinstance(weight, QuantizedTensor) or bias is not None:
                    raise RuntimeError("grouped DiffusionGemma NVFP4 requires unbiased resident quantized banks")
                weights.append(weight)

            qx, x_scale, x_block_scale = quantize(x)
            gate = grouped_linear(qx, x_scale, x_block_scale, weights[0])
            up = grouped_linear(qx, x_scale, x_block_scale, weights[1])
            intermediate = _gelu_tanh(gate) * up
            qi, i_scale, i_block_scale = quantize(intermediate)
            y = grouped_linear(qi, i_scale, i_block_scale, weights[2])

        pair_order = torch.empty(N * K, dtype=torch.long, device=flat_experts.device)
        pair_order[order] = slot
        y = y.reshape(E * C, H)[pair_order]
        y = y * top_k_weights.reshape(-1, 1)
        return y.view(N, K, H).sum(dim=1).to(hidden_states.dtype)

    def _forward_grouped_fused_nvfp4(self, hidden_states, top_k_index, top_k_weights):
        ck = comfy.quant_ops.ck
        N, H = hidden_states.shape
        E = self.num_experts
        K = top_k_index.shape[-1]

        flat_experts = top_k_index.reshape(-1)
        counts = torch.bincount(flat_experts, minlength=E)
        C = -(-int(counts.max()) // self.grouped_nvfp4_bucket) * self.grouped_nvfp4_bucket
        order = torch.argsort(flat_experts)
        sorted_experts = flat_experts[order]
        rank = torch.arange(N * K, device=flat_experts.device) - (counts.cumsum(0) - counts)[sorted_experts]
        slot = sorted_experts * C + rank

        gather_tok = torch.zeros(E * C, dtype=torch.long, device=flat_experts.device)
        gather_tok[slot] = order // K
        x = hidden_states[gather_tok].view(E, C, H)

        def quantize(tensor, scale):
            flat = tensor.flatten(0, 1)
            scale = scale.to(device=flat.device, dtype=torch.float32)
            qdata, block_scale = ck.quantize_nvfp4(flat, scale, hi_first=False)
            return qdata, scale, block_scale

        def grouped_linear(qdata, input_scale, input_block_scale, weight):
            params = weight._params
            return self._grouped_nvfp4_mm(
                qdata,
                weight._qdata,
                input_scale,
                params.scale,
                input_block_scale,
                params.block_scale,
                C,
                out_dtype=hidden_states.dtype,
            )

        with contextlib.ExitStack() as stack:
            modules = (self.gate_up_proj, self.down_proj)
            banks = [stack.enter_context(module.bank_resident(hidden_states)) for module in modules]
            weights = []
            for bank in banks:
                weight, bias = bank._resident_bank
                if not isinstance(weight, QuantizedTensor) or bias is not None:
                    raise RuntimeError("grouped DiffusionGemma fused NVFP4 requires unbiased resident banks")
                weights.append(weight)

            qx, x_scale, x_block_scale = quantize(x, self.gate_up_proj.input_scale)
            gate_up = grouped_linear(qx, x_scale, x_block_scale, weights[0])
            up, gate = gate_up.chunk(2, dim=-1)
            intermediate = _gelu_tanh(gate) * up
            qi, i_scale, i_block_scale = quantize(intermediate, self.down_proj.input_scale)
            y = grouped_linear(qi, i_scale, i_block_scale, weights[1])

        pair_order = torch.empty(N * K, dtype=torch.long, device=flat_experts.device)
        pair_order[order] = slot
        y = y.reshape(E * C, H)[pair_order]
        y = y * top_k_weights.reshape(-1, 1)
        return y.view(N, K, H).sum(dim=1).to(hidden_states.dtype)

    def _forward_grouped_fused_mxfp8(self, hidden_states, top_k_index, top_k_weights):
        ck = comfy.quant_ops.ck
        N, H = hidden_states.shape
        E = self.num_experts
        K = top_k_index.shape[-1]

        flat_experts = top_k_index.reshape(-1)
        counts = torch.bincount(flat_experts, minlength=E)
        C = -(-int(counts.max()) // self.grouped_mxfp8_bucket) * self.grouped_mxfp8_bucket
        order = torch.argsort(flat_experts)
        sorted_experts = flat_experts[order]
        rank = torch.arange(N * K, device=flat_experts.device) - (counts.cumsum(0) - counts)[sorted_experts]
        slot = sorted_experts * C + rank

        gather_tok = torch.zeros(E * C, dtype=torch.long, device=flat_experts.device)
        gather_tok[slot] = order // K
        x = hidden_states[gather_tok].view(E, C, H)

        def grouped_linear(tensor, weight):
            qdata, block_scale = ck.quantize_mxfp8(tensor.flatten(0, 1).contiguous(), pad_32x=False)
            return self._grouped_mxfp8_mm(
                qdata,
                weight._qdata,
                block_scale,
                weight._params.scale,
                C,
                hidden_states.dtype,
            )

        with contextlib.ExitStack() as stack:
            modules = (self.gate_up_proj, self.down_proj)
            banks = [stack.enter_context(module.bank_resident(hidden_states)) for module in modules]
            weights = []
            for bank in banks:
                weight, bias = bank._resident_bank
                if not isinstance(weight, QuantizedTensor) or bias is not None:
                    raise RuntimeError("grouped DiffusionGemma fused MXFP8 requires unbiased resident banks")
                weights.append(weight)

            gate_up = grouped_linear(x, weights[0])
            gate, up = gate_up.chunk(2, dim=-1)
            y = grouped_linear(_gelu_tanh(gate) * up, weights[1])

        pair_order = torch.empty(N * K, dtype=torch.long, device=flat_experts.device)
        pair_order[order] = slot
        y = y.reshape(E * C, H)[pair_order]
        y = y * top_k_weights.reshape(-1, 1)
        return y.view(N, K, H).sum(dim=1).to(hidden_states.dtype)

    def _forward_grouped_mxfp8(self, hidden_states, top_k_index, top_k_weights):
        ck = comfy.quant_ops.ck
        N, H = hidden_states.shape
        E = self.num_experts
        K = top_k_index.shape[-1]

        flat_experts = top_k_index.reshape(-1)
        counts = torch.bincount(flat_experts, minlength=E)
        C = -(-int(counts.max()) // self.grouped_mxfp8_bucket) * self.grouped_mxfp8_bucket
        order = torch.argsort(flat_experts)
        sorted_experts = flat_experts[order]
        rank = torch.arange(N * K, device=flat_experts.device) - (counts.cumsum(0) - counts)[sorted_experts]
        slot = sorted_experts * C + rank

        gather_tok = torch.zeros(E * C, dtype=torch.long, device=flat_experts.device)
        gather_tok[slot] = order // K
        x = hidden_states[gather_tok].view(E, C, H)

        def quantize(tensor):
            return ck.quantize_mxfp8(tensor.flatten(0, 1).contiguous(), pad_32x=False)

        def grouped_linear(qdata, input_block_scale, weight):
            return self._grouped_mxfp8_mm(
                qdata,
                weight._qdata,
                input_block_scale,
                weight._params.scale,
                C,
                hidden_states.dtype,
            )

        with contextlib.ExitStack() as stack:
            modules = (self.gate_proj, self.up_proj, self.down_proj)
            banks = [stack.enter_context(module.bank_resident(hidden_states)) for module in modules]
            weights = []
            for bank in banks:
                weight, bias = bank._resident_bank
                if (
                    not isinstance(weight, QuantizedTensor)
                    or weight._layout_cls != "TensorCoreMXFP8Layout"
                    or bias is not None
                ):
                    raise RuntimeError("grouped DiffusionGemma MXFP8 requires unbiased resident MXFP8 banks")
                weights.append(weight)

            qx, x_block_scale = quantize(x)
            gate = grouped_linear(qx, x_block_scale, weights[0])
            up = grouped_linear(qx, x_block_scale, weights[1])
            qi, i_block_scale = quantize(_gelu_tanh(gate) * up)
            y = grouped_linear(qi, i_block_scale, weights[2])

        pair_order = torch.empty(N * K, dtype=torch.long, device=flat_experts.device)
        pair_order[order] = slot
        y = y.reshape(E * C, H)[pair_order]
        y = y * top_k_weights.reshape(-1, 1)
        return y.view(N, K, H).sum(dim=1).to(hidden_states.dtype)

    def _forward_grouped_int8_convrot(self, hidden_states, top_k_index, top_k_weights):
        N, H = hidden_states.shape
        E = self.num_experts
        K = top_k_index.shape[-1]

        flat_experts = top_k_index.reshape(-1)
        order = torch.argsort(flat_experts)
        sorted_experts = flat_experts[order]
        positions = torch.arange(N * K, device=flat_experts.device)
        counts = torch.zeros(E, dtype=torch.int32, device=flat_experts.device)
        counts.scatter_add_(0, sorted_experts, torch.ones(N * K, dtype=torch.int32, device=flat_experts.device))
        expert_indptr = torch.zeros(E + 1, dtype=torch.int32, device=flat_experts.device)
        torch.cumsum(counts, dim=0, dtype=torch.int32, out=expert_indptr[1:])
        x = hidden_states[order // K]

        with contextlib.ExitStack() as stack:
            modules = (self.gate_up_proj, self.down_proj)
            banks = [stack.enter_context(module.bank_resident(hidden_states)) for module in modules]
            weights = []
            for bank in banks:
                weight, bias = bank._resident_bank
                if (
                    not isinstance(weight, QuantizedTensor)
                    or weight._layout_cls != "TensorWiseINT8Layout"
                    or bias is not None
                ):
                    raise RuntimeError("grouped DiffusionGemma INT8 ConvRot requires unbiased resident banks")
                weights.append(weight)

            gate_up = comfy.quant_ops.grouped_int8_convrot_linear_packed(
                x,
                expert_indptr,
                weights[0]._qdata,
                weights[0]._params.scale,
                weights[0]._params.convrot_groupsize,
                out_dtype=hidden_states.dtype,
            )
            gate, up = gate_up.chunk(2, dim=-1)
            intermediate = _gelu_tanh(gate) * up
            y = comfy.quant_ops.grouped_int8_convrot_linear_packed(
                intermediate,
                expert_indptr,
                weights[1]._qdata,
                weights[1]._params.scale,
                weights[1]._params.convrot_groupsize,
                out_dtype=hidden_states.dtype,
            )

        pair_order = torch.empty(N * K, dtype=torch.long, device=flat_experts.device)
        pair_order[order] = positions
        y = y[pair_order]
        y = y * top_k_weights.reshape(-1, 1)
        return y.view(N, K, H).sum(dim=1).to(hidden_states.dtype)

    def _forward_grouped(self, hidden_states, top_k_index, top_k_weights):
        N, H = hidden_states.shape
        E = self.num_experts
        K = top_k_index.shape[-1]

        flat_experts = top_k_index.reshape(-1)
        counts = torch.bincount(flat_experts, minlength=E)
        # Round the bucket up to limit allocator churn; padding is negligible beside bank dequant traffic.
        C = -(-int(counts.max()) // self.grouped_bucket) * self.grouped_bucket
        order = torch.argsort(flat_experts)
        sorted_experts = flat_experts[order]
        rank = torch.arange(N * K, device=flat_experts.device) - (counts.cumsum(0) - counts)[sorted_experts]
        slot = sorted_experts * C + rank

        gather_tok = torch.zeros(E * C, dtype=torch.long, device=flat_experts.device)
        gather_tok[slot] = order // K

        x = hidden_states[gather_tok].view(E, C, H)
        if self.unfused:
            gate = _bank_bmm(self.gate_proj, x)
            up = _bank_bmm(self.up_proj, x)
        else:
            gate, up = _bank_bmm(self.gate_up_proj, x).chunk(2, dim=-1)
        y = _bank_bmm(self.down_proj, _gelu_tanh(gate) * up)

        # pairs are a permutation: gather the real rows back to original pair order
        # and reduce densely — no atomics, and the weight multiply touches only N*K rows
        pair_order = torch.empty(N * K, dtype=torch.long, device=flat_experts.device)
        pair_order[order] = slot
        y = y.reshape(E * C, -1)[pair_order]
        y = y * top_k_weights.reshape(-1, 1)
        return y.view(N, K, H).sum(dim=1).to(hidden_states.dtype)

    def _forward_loop(self, hidden_states, top_k_index, top_k_weights):
        final_hidden_states = torch.zeros_like(hidden_states)
        expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        gate_up_banks = (self.gate_proj, self.up_proj) if self.unfused else (self.gate_up_proj,)
        with contextlib.ExitStack() as stack:
            banks = [stack.enter_context(b.bank_resident(hidden_states)) for b in gate_up_banks + (self.down_proj,)]
            down_bank = banks[-1]
            # Copy the hit list once because per-expert .item() calls serialize the CUDA stream.
            for expert_idx in expert_hit.flatten().tolist():
                top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
                current_state = hidden_states[token_idx]
                if self.unfused:
                    gate = banks[0].expert_linear(current_state, expert_idx)
                    up = banks[1].expert_linear(current_state, expert_idx)
                else:
                    gate, up = banks[0].expert_linear(current_state, expert_idx).chunk(2, dim=-1)
                current_hidden_states = _gelu_tanh(gate) * up
                current_hidden_states = down_bank.expert_linear(current_hidden_states, expert_idx)
                current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
                final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states


class DiffusionGemmaSelfConditioning(nn.Module):
    def __init__(self, config, device=None, dtype=None, ops=None):
        super().__init__()
        self.pre_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, device=device, dtype=dtype)
        self.gate_proj = ops.Linear(config.hidden_size, config.intermediate_size, bias=False, device=device, dtype=dtype)
        self.up_proj = ops.Linear(config.hidden_size, config.intermediate_size, bias=False, device=device, dtype=dtype)
        self.down_proj = ops.Linear(config.intermediate_size, config.hidden_size, bias=False, device=device, dtype=dtype)

    def forward(self, inputs_embeds, self_conditioning_signal):
        normed = self.pre_norm(self_conditioning_signal)
        sc_signal = self.down_proj(_gelu_tanh(self.gate_proj(normed)) * self.up_proj(normed))
        return rms_norm(inputs_embeds + sc_signal)


class DiffusionGemmaBlock(nn.Module):
    def __init__(self, config, index, device=None, dtype=None, ops=None):
        super().__init__()
        self.sliding_window = config.sliding_attention[index % len(config.sliding_attention)] or None
        is_sliding = self.sliding_window is not None
        head_dim = config.head_dim if is_sliding else config.global_head_dim
        num_kv_heads = config.num_key_value_heads if is_sliding else config.num_global_key_value_heads

        self.self_attn = DiffusionGemmaAttention(config, head_dim=head_dim, num_kv_heads=num_kv_heads,
                                                 has_v_proj=is_sliding, device=device, dtype=dtype, ops=ops)
        self.mlp = DiffusionGemmaMLP(config, device=device, dtype=dtype, ops=ops)
        self.router = DiffusionGemmaRouter(config, device=device, dtype=dtype, ops=ops)
        self.experts = DiffusionGemmaExperts(config, device=device, dtype=dtype, ops=ops)

        norm_kwargs = dict(eps=config.rms_norm_eps, device=device, dtype=dtype)
        self.input_layernorm = RMSNorm(config.hidden_size, **norm_kwargs)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, **norm_kwargs)
        self.pre_feedforward_layernorm = RMSNorm(config.hidden_size, **norm_kwargs)
        self.post_feedforward_layernorm = RMSNorm(config.hidden_size, **norm_kwargs)
        self.post_feedforward_layernorm_1 = RMSNorm(config.hidden_size, **norm_kwargs)
        self.post_feedforward_layernorm_2 = RMSNorm(config.hidden_size, **norm_kwargs)
        self.pre_feedforward_layernorm_2 = RMSNorm(config.hidden_size, **norm_kwargs)
        self.register_buffer("layer_scalar", torch.empty(1, device=device, dtype=dtype))

    def forward(self, x, attention_mask=None, freqs_cis=None, past_key_value=None,
                update_cache=True, layer_scalar=None):
        freqs_cis = freqs_cis[1] if self.sliding_window is not None else freqs_cis[0]

        residual = x
        x = self.input_layernorm(x)
        x, present_key_value = self.self_attn(
            hidden_states=x, attention_mask=attention_mask, freqs_cis=freqs_cis,
            past_key_value=past_key_value, sliding_window=self.sliding_window, update_cache=update_cache)
        x = self.post_attention_layernorm(x)
        x = residual + x

        residual = x
        h = self.pre_feedforward_layernorm(x)
        h = self.mlp(h)
        hidden_states_1 = self.post_feedforward_layernorm_1(h)

        flat = residual.reshape(-1, residual.shape[-1])
        defer_expert_scale = self.experts._supports_native_fused_mxfp8_scaled(flat)
        top_k_weights, top_k_index = self.router(flat, apply_expert_scale=not defer_expert_scale)
        expert_scale = (
            comfy.ops.cast_to_input(self.router.per_expert_scale, flat, copy=False)
            if defer_expert_scale else None
        )
        hidden_states_2 = self.experts(
            self.pre_feedforward_layernorm_2(flat),
            top_k_index,
            top_k_weights,
            expert_scale=expert_scale,
        )
        hidden_states_2 = self.post_feedforward_layernorm_2(hidden_states_2.reshape(residual.shape))

        x = self.post_feedforward_layernorm(hidden_states_1 + hidden_states_2)
        x = residual + x

        scalar = layer_scalar if layer_scalar is not None else self.layer_scalar
        x = x * comfy.ops.cast_to_input(scalar, x, copy=False)
        return x, present_key_value


class DiffusionGemmaDecoder(nn.Module):
    def __init__(self, config, device=None, dtype=None, ops=None):
        super().__init__()
        self.embed_tokens = _make_dg_scaled_embedding(ops, config.vocab_size, config.hidden_size, device, dtype)
        self.layers = nn.ModuleList([
            DiffusionGemmaBlock(config, index=i, device=device, dtype=dtype, ops=ops)
            for i in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, device=device, dtype=dtype)
        self.self_conditioning = DiffusionGemmaSelfConditioning(config, device=device, dtype=dtype, ops=ops)

        rope_angles_global = int(config.partial_rotary_factor * config.global_head_dim // 2)
        nope_global = config.global_head_dim // 2 - rope_angles_global
        global_inv = 1.0 / (config.rope_theta[0] ** (torch.arange(0, 2 * rope_angles_global, 2).float() / config.global_head_dim))
        if nope_global > 0:
            global_inv = torch.cat([global_inv, torch.zeros(nope_global)])
        self.register_buffer("_global_inv_freq", global_inv, persistent=False)
        sliding_inv = 1.0 / (config.rope_theta[1] ** (torch.arange(0, config.head_dim, 2).float() / config.head_dim))
        self.register_buffer("_sliding_inv_freq", sliding_inv, persistent=False)

    def _freqs_from_inv(self, inv_freq, position_ids, device, dtype):
        inv_exp = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(device)
        pos_exp = position_ids[:, None, :].float()
        freqs = (inv_exp @ pos_exp).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().unsqueeze(1).to(dtype), emb.sin().unsqueeze(1).to(dtype)

    def compute_freqs_cis(self, position_ids, device, dtype=None):
        global_freqs = self._freqs_from_inv(self._global_inv_freq, position_ids, device, dtype)
        sliding_freqs = self._freqs_from_inv(self._sliding_inv_freq, position_ids, device, dtype)
        return [global_freqs, sliding_freqs]


class _EncoderLayerScalar(nn.Module):
    def __init__(self, device=None, dtype=None):
        super().__init__()
        self.register_buffer("layer_scalar", torch.empty(1, device=device, dtype=dtype))


class _EncoderLanguageModel(nn.Module):
    # Encoder text weights are tied to the decoder; only the per-layer scalars are separate.
    def __init__(self, num_layers, device=None, dtype=None):
        super().__init__()
        self.layers = nn.ModuleList([_EncoderLayerScalar(device=device, dtype=dtype) for _ in range(num_layers)])


class DiffusionGemmaVisionTower(Gemma4VisionEncoder):
    def __init__(self, config, dtype=None, device=None, ops=None):
        super().__init__(config, dtype=dtype, device=device, ops=ops)
        self.register_buffer("std_bias", torch.empty(config["hidden_size"], device=device, dtype=dtype))
        self.register_buffer("std_scale", torch.empty(config["hidden_size"], device=device, dtype=dtype))
        # use_clipped_linears=False in this model: no clip buffers in the checkpoint
        for m in self.modules():
            if isinstance(m, ClippedLinear):
                for name in ("input_min", "input_max", "output_min", "output_max"):
                    m._buffers[name] = None
                m.forward = m.linear.__call__

    def forward(self, pixel_values, max_soft_tokens=None):
        x = super().forward(pixel_values, max_soft_tokens=max_soft_tokens)
        std_bias = comfy.model_management.cast_to_device(self.std_bias, x.device, torch.float32)
        std_scale = comfy.model_management.cast_to_device(self.std_scale, x.device, torch.float32)
        return ((x.float() - std_bias) * std_scale).to(x.dtype)


class DiffusionGemmaEncoderModule(nn.Module):
    def __init__(self, config, device=None, dtype=None, ops=None):
        super().__init__()
        self.vision_tower = DiffusionGemmaVisionTower(config.vision_config, dtype=dtype, device=device, ops=ops)
        self.embed_vision = Gemma4RMSNormProjector(config.vision_config["hidden_size"], config.hidden_size, dtype=dtype, device=device, ops=ops)
        self.language_model = _EncoderLanguageModel(config.num_hidden_layers, device=device, dtype=dtype)


class DiffusionGemmaModel(nn.Module):
    def __init__(self, config, device=None, dtype=None, ops=None):
        super().__init__()
        self.config = config
        self.decoder = DiffusionGemmaDecoder(config, device=device, dtype=dtype, ops=ops)
        self.encoder = DiffusionGemmaEncoderModule(config, device=device, dtype=dtype, ops=ops)

    @property
    def embed_tokens(self):
        return self.decoder.embed_tokens

    def get_past_len(self, past_key_values):
        for kv in past_key_values:
            if len(kv) >= 3:
                return kv[2]
        return 0

    def _cached_kv_lens(self, past_key_values):
        sliding_len = 0
        full_len = 0
        if past_key_values is not None:
            for i, layer in enumerate(self.decoder.layers):
                kv = past_key_values[i]
                if len(kv) == 0:
                    continue
                cache_len = kv[3] if len(kv) == 4 else kv[0].shape[2]
                if layer.sliding_window is not None:
                    sliding_len = cache_len
                else:
                    full_len = cache_len
        return full_len, sliding_len

    def _encoder_masks(self, q_pos, full_len, sliding_len, sliding_window, dtype, device, mm_spans=None, padding_mask=None):
        seq_len = q_pos.shape[0]
        past_len = int(q_pos[0])
        min_val = torch.finfo(dtype).min

        q = q_pos[:, None]
        k_full = torch.arange(full_len + seq_len, device=device)[None, :]
        full_bool = k_full > q
        k_sliding = torch.cat((torch.arange(past_len - sliding_len, past_len, device=device), q_pos))[None, :]
        sliding_bool = (k_sliding > q) | ((q - k_sliding) >= sliding_window)

        if mm_spans is not None:
            for start, end in mm_spans:
                q_in = (q_pos >= start) & (q_pos < end)
                full_bool = full_bool & ~(q_in[:, None] & ((k_full[0] >= start) & (k_full[0] < end))[None, :])
                sliding_bool = sliding_bool & ~(q_in[:, None] & ((k_sliding[0] >= start) & (k_sliding[0] < end))[None, :])

        def to_mask(bool_mask):
            mask = torch.zeros((1, 1) + bool_mask.shape, dtype=dtype, device=device)
            mask.masked_fill_(bool_mask[None, None], min_val)
            return mask

        masks = {"full": to_mask(full_bool), "sliding": to_mask(sliding_bool)}
        if padding_mask is not None and past_len == 0:
            pad = torch.zeros((padding_mask.shape[0], 1, 1, padding_mask.shape[1]), dtype=dtype, device=device)
            pad.masked_fill_((padding_mask == 0)[:, None, None, :], min_val)
            masks = {k: v + pad for k, v in masks.items()}
        return masks

    def forward(self, x, attention_mask=None, embeds=None, num_tokens=None, intermediate_output=None,
                final_layer_norm_intermediate=True, dtype=None, position_ids=None, embeds_info=None,
                past_key_values=None, input_ids=None, mode="encoder", self_conditioning_logits=None,
                mm_spans=None, freqs_cis=None):
        if embeds is not None:
            x = embeds
        else:
            x = self.decoder.embed_tokens(x, out_dtype=dtype)

        if mode == "decoder":
            embed_module = self.decoder.embed_tokens
            if self_conditioning_logits is not None:
                if _native_int8_embedding(embed_module):
                    weight, _, offload_stream = comfy.ops.cast_bias_weight(
                        embed_module, device=x.device, dtype=embed_module.weight.dtype, offloadable=True)
                    try:
                        if not isinstance(weight, QuantizedTensor):
                            raise RuntimeError("DiffusionGemma INT8 embedding did not remain quantized after device cast")
                        probabilities = self_conditioning_logits.softmax(dim=-1, dtype=torch.float32).to(torch.bfloat16)
                        soft_embeddings = _int8_self_conditioning(probabilities, weight)
                        scale = torch.tensor(
                            self.config.hidden_size ** 0.5,
                            dtype=weight._params.orig_dtype,
                            device=soft_embeddings.device,
                        )
                        soft_embeddings = (soft_embeddings * scale).to(x.dtype)
                    finally:
                        comfy.ops.uncast_bias_weight(embed_module, weight, None, offload_stream)
                else:
                    probabilities = self_conditioning_logits.softmax(dim=-1, dtype=torch.float32)
                    soft_embeddings = embed_module.weighted_embedding(probabilities).to(x.dtype)
            else:
                soft_embeddings = torch.zeros_like(x)
            x = self.decoder.self_conditioning(x, soft_embeddings)

        seq_len = x.shape[1]
        past_len = 0
        if past_key_values is not None and len(past_key_values) > 0:
            past_len = self.get_past_len(past_key_values)

        if position_ids is None:
            position_ids = torch.arange(past_len, past_len + seq_len, device=x.device).unsqueeze(0)

        if freqs_cis is None:
            freqs_cis = self.decoder.compute_freqs_cis(position_ids, x.device, dtype=x.dtype)

        if mode == "decoder":
            masks = {"full": None, "sliding": None}
        else:
            if embeds_info and mm_spans is None:
                mm_spans = [(e["index"], e["index"] + e["size"]) for e in embeds_info if e.get("type") == "image"]
            full_len, sliding_len = self._cached_kv_lens(past_key_values)
            padding_mask = attention_mask if attention_mask is not None else None
            if seq_len == 1 and past_len > 0:
                masks = {"full": None, "sliding": None}
            else:
                masks = self._encoder_masks(position_ids[0], full_len, sliding_len,
                                            self.config.sliding_attention[0], x.dtype, x.device,
                                            mm_spans=mm_spans if mm_spans else None,
                                            padding_mask=padding_mask)

        update_cache = (mode == "encoder") and past_key_values is not None
        intermediate = None
        next_key_values = []
        for i, layer in enumerate(self.decoder.layers):
            past_kv = past_key_values[i] if past_key_values is not None and len(past_key_values) > 0 else None
            layer_scalar = None
            if mode == "encoder":
                layer_scalar = self.encoder.language_model.layers[i].layer_scalar
            mask = masks["sliding"] if layer.sliding_window is not None else masks["full"]
            x, current_kv = layer(x, attention_mask=mask, freqs_cis=freqs_cis, past_key_value=past_kv,
                                  update_cache=update_cache, layer_scalar=layer_scalar)
            next_key_values.append(current_kv if current_kv is not None else ())
            if i == intermediate_output:
                intermediate = x.clone()

        x = self.decoder.norm(x)
        return x, intermediate, next_key_values


class _StableAndConfidentStopping:
    def __init__(self, stability_threshold, confidence_threshold):
        self.stability_threshold = stability_threshold
        self.confidence_threshold = confidence_threshold
        self.argmax_canvas_history = None
        self.history_length = 0

    def __call__(self, argmax_canvas, token_entropy):
        if self.stability_threshold == 0:
            stable = torch.ones((token_entropy.shape[0]), device=token_entropy.device, dtype=torch.bool)
        else:
            if self.argmax_canvas_history is None:
                self.argmax_canvas_history = torch.full(
                    (self.stability_threshold, argmax_canvas.shape[0], argmax_canvas.shape[1]),
                    -1, dtype=argmax_canvas.dtype, device=argmax_canvas.device)
            self.argmax_canvas_history = torch.roll(self.argmax_canvas_history, shifts=-1, dims=0)
            self.argmax_canvas_history[-1] = argmax_canvas
            self.history_length += 1
            stable = (self.argmax_canvas_history == argmax_canvas[None, :, :]).all(dim=-1).all(dim=0)
            stable = stable & (self.history_length >= self.stability_threshold)

        confident = torch.mean(token_entropy, dim=-1) < self.confidence_threshold
        return stable & confident


def _diffusion_probs_and_entropy(processed_logits):
    distribution = torch.distributions.Categorical(logits=processed_logits)
    return distribution.probs, distribution.entropy()


def _entropy_bound_accept(current_canvas, denoiser_canvas, entropy_bound, token_entropy):
    sorted_token_entropy, sorted_indices = torch.sort(token_entropy, dim=-1, descending=False)
    cumulative_entropy = torch.cumsum(sorted_token_entropy, dim=-1)
    sorted_selection_mask = cumulative_entropy - sorted_token_entropy <= entropy_bound
    accepted_token_mask = torch.scatter(
        input=torch.zeros_like(sorted_selection_mask), dim=-1, index=sorted_indices, src=sorted_selection_mask)
    accepted_canvas = torch.where(accepted_token_mask, denoiser_canvas, current_canvas)
    return accepted_canvas, accepted_token_mask, token_entropy


def _graph_tensor_fingerprint(tensor):
    if tensor is None:
        return None
    version = None if tensor.is_inference() else tensor._version
    return (
        tensor.data_ptr(), tensor.storage_offset(), tuple(tensor.shape), tuple(tensor.stride()),
        tensor.dtype, tensor.device, version,
    )


def _graph_value_fingerprint(value):
    if isinstance(value, torch.Tensor):
        return _graph_tensor_fingerprint(value)
    if isinstance(value, (tuple, list)):
        return tuple(_graph_value_fingerprint(item) for item in value)
    return value


class _ConditionedDecoderGraph:
    def __init__(self, owner, execution, current_canvas, self_conditioning_logits, past_key_values,
                 position_ids, freqs_cis, execution_dtype, residency_lease):
        self.owner = owner
        self.stream = execution.stream
        self.current_canvas = execution.static_canvas
        self.self_conditioning_logits = execution.static_logits
        self.past_key_values = past_key_values
        self.position_ids = position_ids
        self.freqs_cis = freqs_cis
        self.execution_dtype = execution_dtype
        self.residency_lease = residency_lease
        self.geometry = self._geometry(past_key_values, position_ids, freqs_cis)
        self.graph = None
        self.output = None

        self.stream.wait_stream(torch.cuda.current_stream(current_canvas.device))
        with torch.cuda.stream(self.stream):
            self.current_canvas.copy_(current_canvas)
            self.self_conditioning_logits.copy_(self_conditioning_logits)
            comfy.quant_ops.ck.begin_cuda_graph_capture(self.stream)
            try:
                self.output, _, _ = owner.model(
                    self.current_canvas,
                    past_key_values=self.past_key_values,
                    mode="decoder",
                    self_conditioning_logits=self.self_conditioning_logits,
                    position_ids=self.position_ids,
                    dtype=self.execution_dtype,
                    freqs_cis=self.freqs_cis,
                )
            except BaseException:
                comfy.quant_ops.ck.abort_cuda_graph_capture(self.stream)
                raise
            self.graph = comfy.quant_ops.ck.end_cuda_graph_capture(
                self.stream,
                self.output,
                self.current_canvas,
                self.self_conditioning_logits,
                self.past_key_values,
                self.position_ids,
                self.freqs_cis,
            )
        self.stream.synchronize()

    @staticmethod
    def _geometry(past_key_values, position_ids, freqs_cis):
        return _graph_value_fingerprint((past_key_values, position_ids, freqs_cis))

    @classmethod
    def capture(cls, owner, execution, current_canvas, self_conditioning_logits, past_key_values,
                position_ids, freqs_cis, execution_dtype):
        residency_lease = comfy.model_management.acquire_dynamic_vram_graph_lease(
            owner.model.decoder, owner._weight_patches_uuid
        )
        if residency_lease is None:
            return None
        try:
            graph = cls(
                owner, execution, current_canvas, self_conditioning_logits, past_key_values,
                position_ids, freqs_cis, execution_dtype, residency_lease,
            )
        except BaseException:
            residency_lease.release()
            raise
        if not residency_lease.valid(owner._weight_patches_uuid, full=True) or graph.geometry != graph._geometry(
            past_key_values, position_ids, freqs_cis
        ):
            graph.close()
            return None
        return graph

    def replay(self, current_canvas, self_conditioning_logits, past_key_values, position_ids, freqs_cis):
        if (
            not self.residency_lease.valid(self.owner._weight_patches_uuid)
            or self.geometry != self._geometry(past_key_values, position_ids, freqs_cis)
        ):
            self.close()
            return None
        self.stream.wait_stream(torch.cuda.current_stream(current_canvas.device))
        with torch.cuda.stream(self.stream):
            self.current_canvas.copy_(current_canvas)
            self.self_conditioning_logits.copy_(self_conditioning_logits)
            self.graph.replay(self.stream)
        torch.cuda.current_stream(current_canvas.device).wait_stream(self.stream)
        return self.output

    def close(self):
        try:
            if self.graph is not None:
                self.stream.synchronize()
                self.graph.reset()
                self.graph = None
        finally:
            if self.residency_lease is not None:
                self.residency_lease.release()
                self.residency_lease = None
            self.output = None
            self.owner = None
            self.current_canvas = None
            self.self_conditioning_logits = None
            self.past_key_values = None
            self.position_ids = None
            self.freqs_cis = None


class _ConditionedDecoderGraphExecution:
    def __init__(self, device, batch, canvas_length, vocab_size, execution_dtype):
        self.stream = torch.cuda.Stream(device=device)
        self.static_canvas = torch.empty((batch, canvas_length), dtype=torch.long, device=device)
        self.static_logits = torch.empty(
            (batch, canvas_length, vocab_size), dtype=execution_dtype, device=device)
        self.graph = None
        self.workspace_reserved = comfy.quant_ops.ck.reserve_cuda_stream_workspaces(self.stream)
        if not self.workspace_reserved:
            raise RuntimeError("DiffusionGemma CUDA graph stream already owns Kitchen workspaces")

    def capture(self, owner, current_canvas, self_conditioning_logits, past_key_values,
                position_ids, freqs_cis, execution_dtype):
        self.graph = _ConditionedDecoderGraph.capture(
            owner, self, current_canvas, self_conditioning_logits, past_key_values,
            position_ids, freqs_cis, execution_dtype,
        )
        return self.graph is not None

    def replay(self, current_canvas, self_conditioning_logits, past_key_values, position_ids, freqs_cis):
        output = self.graph.replay(
            current_canvas, self_conditioning_logits, past_key_values, position_ids, freqs_cis)
        if output is None:
            self.graph = None
        return output

    def close(self):
        if self.graph is not None:
            self.graph.close()
            self.graph = None
        if self.stream is not None:
            self.stream.synchronize()
        if self.workspace_reserved:
            if not comfy.quant_ops.ck.release_cuda_stream_workspaces(self.stream):
                raise RuntimeError("DiffusionGemma CUDA graph Kitchen workspaces were not reserved")
            self.workspace_reserved = False
        self.static_canvas = None
        self.static_logits = None
        self.stream = None


class DiffusionGenerate:
    def _raw_logits(self, x):
        module = self.model.decoder.embed_tokens
        offload_stream = None
        native_mxfp8 = _native_mxfp8_embedding(module)
        native_int8 = _native_int8_embedding(module)
        if native_mxfp8 or native_int8:
            weight, _, offload_stream = comfy.ops.cast_bias_weight(
                module, device=x.device, dtype=module.weight.dtype, offloadable=True)
            if not isinstance(weight, QuantizedTensor):
                raise RuntimeError("DiffusionGemma quantized embedding did not remain quantized after device cast")
        elif module.comfy_cast_weights:
            weight, _, offload_stream = comfy.ops.cast_bias_weight(module, x, offloadable=True)
        else:
            weight = module.weight.to(x)
        if native_mxfp8:
            input_shape = x.shape
            quantized = QuantizedTensor.from_float(x.reshape(-1, input_shape[-1]), weight._layout_cls)
            logits = torch.nn.functional.linear(quantized, weight, None).reshape(*input_shape[:-1], weight.shape[0])
        elif native_int8:
            logits = torch.nn.functional.linear(x, weight, None)
        else:
            logits = torch.nn.functional.linear(x, weight, None)
        comfy.ops.uncast_bias_weight(module, weight, None, offload_stream)
        return logits

    def logits(self, x):
        logits = self._raw_logits(x).to(torch.float32)
        cap = self.model.config.final_logit_softcapping
        return torch.tanh(logits / cap) * cap

    def _use_native_sampling(self, device, execution_dtype):
        return (
            device.type == "cuda"
            and execution_dtype == torch.bfloat16
            and (
                torch.cuda.get_device_capability(device) == (12, 0)
                or (
                    torch.cuda.get_device_capability(device) == (8, 6)
                    and _native_int8_embedding(self.model.decoder.embed_tokens)
                )
            )
        )

    def _use_conditioned_decoder_graph(self, device, execution_dtype):
        return (
            device.type == "cuda"
            and execution_dtype == torch.bfloat16
            and torch.cuda.get_device_capability(device) == (12, 0)
            and torch.cuda.memory.get_allocator_backend() == "cudaMallocAsync"
            and comfy.model_management.dynamic_vram_graph_capture_enabled()
            and _native_mxfp8_embedding(self.model.decoder.embed_tokens)
        )

    def _acquire_sm86_int8_expert_residency(self, input):
        device = input.device
        layers = self.model.decoder.layers
        if (
            device.type != "cuda"
            or torch.cuda.get_device_capability(device) != (8, 6)
            or torch.cuda.get_device_properties(device).total_memory < _INT8_SM86_MIN_DEVICE_MEMORY
            or len(layers) < _INT8_SM86_RESIDENT_EXPERT_LAYERS
            or not _native_int8_embedding(self.model.decoder.embed_tokens)
            or not all(
                layer.experts._grouped_int8_convrot_compatible
                for layer in layers[:_INT8_SM86_RESIDENT_EXPERT_LAYERS]
            )
        ):
            return None

        stack = contextlib.ExitStack()
        try:
            for layer in layers[:_INT8_SM86_RESIDENT_EXPERT_LAYERS]:
                stack.enter_context(layer.experts.gate_up_proj.bank_resident(input))
                stack.enter_context(layer.experts.down_proj.bank_resident(input))
        except BaseException:
            stack.close()
            raise
        return stack

    def init_kv_cache(self, batch, max_cache_len, device, execution_dtype):
        return [() for _ in range(self.model.config.num_hidden_layers)]

    def _reserve_kv_cache(self, past_key_values, reserve):
        for i, (layer, kv) in enumerate(zip(self.model.decoder.layers, past_key_values)):
            if len(kv) == 4:
                key, value, cumulative_len, cache_len = kv
            else:
                key, value, cumulative_len = kv
                cache_len = key.shape[2]
            keep = min(cache_len, layer.sliding_window - 1) if layer.sliding_window is not None else cache_len
            needed = keep + reserve
            if key.shape[2] >= needed and value.shape[2] >= needed:
                if keep and cache_len != keep:
                    compact_key = key[:, :, cache_len - keep:cache_len].clone()
                    compact_value = value[:, :, cache_len - keep:cache_len].clone()
                    key[:, :, :keep].copy_(compact_key)
                    value[:, :, :keep].copy_(compact_value)
                past_key_values[i] = (key, value, cumulative_len, keep)
            else:
                shape = (key.shape[0], key.shape[1], needed, key.shape[3])
                next_key = key.new_empty(shape)
                next_value = value.new_empty(shape)
                next_key[:, :, :keep].copy_(key[:, :, cache_len - keep:cache_len])
                next_value[:, :, :keep].copy_(value[:, :, cache_len - keep:cache_len])
                past_key_values[i] = (next_key, next_value, cumulative_len, keep)
        return past_key_values

    def generate(self, embeds=None, max_length=256, seed=42, max_denoising_steps=48, entropy_bound=0.1,
                 t_min=0.4, t_max=0.8, stability_threshold=1, confidence_threshold=0.005,
                 execution_dtype=None, mm_spans=None, stop_tokens=None, **kwargs):
        device = embeds.device
        config = self.model.config

        if stop_tokens is None:
            stop_tokens = config.stop_tokens
        if execution_dtype is None:
            if comfy.model_management.should_use_bf16(device):
                execution_dtype = torch.bfloat16
            else:
                execution_dtype = torch.float32
        embeds = embeds.to(execution_dtype)
        if embeds.ndim == 2:
            embeds = embeds.unsqueeze(0)

        canvas_length = config.canvas_length
        vocab_size = config.vocab_size
        generator = torch.Generator(device=device).manual_seed(seed) if seed is not None else None
        stop_token_ids = set(stop_tokens)
        pbar = comfy.utils.ProgressBar(max_length)
        tq = tqdm(
            total=max_length,
            desc="Generating tokens",
            unit="it",
            smoothing=0,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        )
        max_new_canvases = math.ceil(max_length / canvas_length)
        use_native_sampling = self._use_native_sampling(device, execution_dtype)
        use_decoder_graph = self._use_conditioned_decoder_graph(device, execution_dtype)

        past_key_values = self.init_kv_cache(embeds.shape[0], 0, device, execution_dtype)
        _, _, past_key_values = self.model(None, embeds=embeds, past_key_values=past_key_values,
                                           mode="encoder", mm_spans=mm_spans)
        past_key_values = self._reserve_kv_cache(past_key_values, canvas_length)
        cur_len = embeds.shape[1]

        generated_token_ids = []
        commit_canvas = None
        use_fused_native_sampling = use_native_sampling

        for canvas_idx in range(max_new_canvases):
            if commit_canvas is not None:
                commit_embeds = self.model.decoder.embed_tokens(commit_canvas).to(execution_dtype)
                _, _, past_key_values = self.model(None, embeds=commit_embeds, past_key_values=past_key_values,
                                                   mode="encoder")
                past_key_values = self._reserve_kv_cache(past_key_values, canvas_length)

            current_canvas = torch.randint(low=0, high=vocab_size, size=(1, canvas_length),
                                           device=device, generator=generator)
            self_conditioning_logits = None
            argmax_canvas = current_canvas
            decoder_position_ids = torch.arange(cur_len, cur_len + canvas_length, device=device).unsqueeze(0)
            decoder_freqs_cis = self.model.decoder.compute_freqs_cis(
                decoder_position_ids, device, dtype=execution_dtype
            )
            stopping = _StableAndConfidentStopping(stability_threshold, confidence_threshold)
            graph_execution = None
            graph_capture_attempted = False
            expert_residency = self._acquire_sm86_int8_expert_residency(embeds)
            try:
                if use_decoder_graph:
                    try:
                        graph_execution = _ConditionedDecoderGraphExecution(
                            device, embeds.shape[0], canvas_length, vocab_size, execution_dtype)
                    except torch.OutOfMemoryError:
                        use_decoder_graph = False
                for cur_step in reversed(range(1, max_denoising_steps + 1)):
                    comfy.model_management.throw_exception_if_processing_interrupted()
                    if self_conditioning_logits is not None and graph_execution is not None and graph_execution.graph is not None:
                        x = graph_execution.replay(
                            current_canvas,
                            self_conditioning_logits,
                            past_key_values,
                            decoder_position_ids,
                            decoder_freqs_cis,
                        )
                        if x is None:
                            graph_execution.close()
                            graph_execution = None
                            use_decoder_graph = False
                    else:
                        x = None
                    if x is None:
                        x, _, _ = self.model(
                            current_canvas, past_key_values=past_key_values, mode="decoder",
                            self_conditioning_logits=self_conditioning_logits,
                            position_ids=decoder_position_ids, dtype=execution_dtype,
                            freqs_cis=decoder_freqs_cis,
                        )
                        if (
                            self_conditioning_logits is not None
                            and graph_execution is not None
                            and not graph_capture_attempted
                        ):
                            graph_capture_attempted = True
                            try:
                                graph_captured = graph_execution.capture(
                                    self,
                                    current_canvas,
                                    self_conditioning_logits,
                                    past_key_values,
                                    decoder_position_ids,
                                    decoder_freqs_cis,
                                    execution_dtype,
                                )
                            except torch.OutOfMemoryError:
                                graph_execution.close()
                                graph_execution = None
                                use_decoder_graph = False
                            else:
                                if not graph_captured:
                                    graph_execution.close()
                                    graph_execution = None
                                    use_decoder_graph = False

                    temperature = t_min + ((t_max - t_min) * (cur_step / max_denoising_steps))
                    next_self_conditioning_logits = None
                    if not use_native_sampling:
                        processed_logits = self.logits(x) / temperature
                        probs, token_entropy = _diffusion_probs_and_entropy(processed_logits)
                        argmax_canvas = torch.argmax(processed_logits, dim=-1)
                        denoiser_canvas = torch.multinomial(
                            probs.view(-1, vocab_size), num_samples=1, generator=generator
                        ).squeeze(-1).view(1, canvas_length)
                        del probs
                    else:
                        raw_logits = self._raw_logits(x)
                        if use_fused_native_sampling:
                            sampling_noise = torch.empty(
                                raw_logits.shape, dtype=torch.float32, device=raw_logits.device
                            ).exponential_(generator=generator)
                            (
                                processed_logits,
                                next_self_conditioning_logits,
                                token_entropy,
                                argmax_canvas,
                                denoiser_canvas,
                            ) = comfy.quant_ops.ck.softcap_categorical_stats_sample(
                                raw_logits,
                                sampling_noise,
                                self.model.config.final_logit_softcapping,
                                1.0 / temperature,
                            )
                        else:
                            processed_logits = comfy.quant_ops.ck.softcap_scale(
                                raw_logits,
                                self.model.config.final_logit_softcapping,
                                1.0 / temperature,
                            )
                            sampling_noise = torch.empty_like(processed_logits).exponential_(generator=generator)
                            token_entropy, argmax_canvas, denoiser_canvas = comfy.quant_ops.ck.categorical_stats_sample(
                                processed_logits, sampling_noise
                            )
                        del raw_logits
                        del sampling_noise

                    accepted_canvas, accepted_mask, token_entropy = _entropy_bound_accept(
                        current_canvas, denoiser_canvas, entropy_bound, token_entropy)
                    random_canvas = torch.randint(low=0, high=vocab_size, size=(1, canvas_length),
                                                  device=device, generator=generator)
                    current_canvas = torch.where(accepted_mask, accepted_canvas, random_canvas)

                    finished_denoising = stopping(argmax_canvas, token_entropy)
                    should_stop = bool(torch.all(finished_denoising))
                    if not should_stop:
                        self_conditioning_logits = (
                            next_self_conditioning_logits
                            if next_self_conditioning_logits is not None
                            else processed_logits.to(execution_dtype)
                        )
                    del processed_logits, next_self_conditioning_logits, token_entropy
                    if should_stop:
                        break
            finally:
                if graph_execution is not None:
                    graph_execution.close()
                if expert_residency is not None:
                    expert_residency.close()

            del self_conditioning_logits
            canvas_ids = argmax_canvas[0].tolist()
            remaining = max_length - len(generated_token_ids)
            first_eos = next((i for i, token_id in enumerate(canvas_ids) if token_id in stop_token_ids), None)
            if first_eos is not None:
                generated_token_ids.extend(canvas_ids[:min(first_eos + 1, remaining)])
            else:
                generated_token_ids.extend(canvas_ids[:remaining])
            output_tokens = len(generated_token_ids)
            pbar.update_absolute(output_tokens, max_length)
            tq.n = output_tokens
            tq.refresh()
            if first_eos is not None or output_tokens >= max_length:
                break
            cur_len += canvas_length
            commit_canvas = argmax_canvas

        output_tokens = len(generated_token_ids)
        pbar.update_absolute(output_tokens, max_length)
        tq.n = output_tokens
        tq.refresh()
        tq.close()
        return generated_token_ids


class DiffusionGemma26B(BaseLlama, DiffusionGenerate, torch.nn.Module):
    def __init__(self, config_dict, dtype, device, operations):
        super().__init__()
        config = DiffusionGemmaConfig(**config_dict)
        self.num_layers = config.num_hidden_layers
        self.model = DiffusionGemmaModel(config, device=device, dtype=dtype, ops=operations)
        self.dtype = dtype
        self._weight_patches_uuid = None

    def preprocess_embed(self, embed, device):
        if embed["type"] == "image":
            image = embed.pop("data").movedim(-1, 1)  # [B, H, W, C] -> [B, C, H, W]
            max_soft_tokens = embed.get("max_soft_tokens", None)
            vision_out = self.model.encoder.vision_tower(image.to(device, dtype=torch.float32), max_soft_tokens=max_soft_tokens)
            return self.model.encoder.embed_vision(vision_out), None
        return None, None


class DiffusionGemmaSDTokenizer(Gemma4SDTokenizer):
    embedding_size = 2816


class DiffusionGemmaTokenizer(Gemma4Tokenizer):
    tokenizer_class = DiffusionGemmaSDTokenizer


class DiffusionGemmaClipModel(Gemma4Model):
    model_class = DiffusionGemma26B
    config_overrides = {}

    def __init__(self, device="cpu", layer="all", layer_idx=None, dtype=None, attention_mask=True, model_options={}):
        if model_options.get("custom_operations") is None:
            model_options = model_options.copy()
            quant_config = model_options.get("quantization_metadata") or {}
            model_options["custom_operations"] = comfy.ops.mixed_precision_ops(quant_config, dtype, full_precision_mm=False)
        self.dtypes = set()
        self.dtypes.add(dtype)
        sd1_clip.SDClipModel.__init__(self, device=device, layer=layer, layer_idx=layer_idx,
                                      textmodel_json_config=dict(self.config_overrides), dtype=dtype,
                                      special_tokens={"start": 2, "pad": 0}, layer_norm_hidden_state=False,
                                      model_class=self.model_class, enable_attention_masks=attention_mask,
                                      return_attention_masks=attention_mask, model_options=model_options)

    def generate(self, tokens, generation_mode=None, **kwargs):
        diffusion_keys = (
            "max_denoising_steps",
            "entropy_bound",
            "t_min",
            "t_max",
            "stability_threshold",
            "confidence_threshold",
        )
        if generation_mode != "diffusion" and not any(key in kwargs for key in diffusion_keys):
            raise ValueError("DiffusionGemma requires diffusion sampling mode in the Generate Text node")
        if isinstance(tokens, dict):
            tokens = next(iter(tokens.values()))
        tokens_only = [[t[0] for t in b] for b in tokens]
        embeds, _, _, embeds_info = sd1_clip.SDClipModel.process_tokens(self, tokens_only, self.execution_device)
        mm_spans = [(e["index"], e["index"] + e["size"]) for e in embeds_info if e.get("type") == "image"]
        for k in ("do_sample", "temperature", "top_k", "top_p", "min_p", "repetition_penalty", "presence_penalty"):
            kwargs.pop(k, None)
        return self.transformer.generate(embeds=embeds, mm_spans=mm_spans if mm_spans else None, **kwargs)


_DIFFUSION_GEMMA_MXFP8_QKV_CONTRACT = "diffusiongemma_mxfp8_qkv_fused.v1"
_DIFFUSION_GEMMA_INT8_QKV_CONTRACT = "diffusiongemma_int8_convrot_qkv_fused.v1"
_DIFFUSION_GEMMA_QKV_LAYER_COUNT = 30


def _detect_diffusion_gemma_fused_qkv(sd):
    marker_keys = [
        f"model.decoder.layers.{layer}.self_attn.qkv_proj.comfy_quant"
        for layer in range(_DIFFUSION_GEMMA_QKV_LAYER_COUNT)
    ]
    has_qkv_payload = any(
        key.startswith("model.decoder.layers.") and ".self_attn.qkv_proj." in key
        for key in sd
    )
    if not has_qkv_payload:
        return False

    missing = [layer for layer, key in enumerate(marker_keys) if key not in sd]
    if missing:
        raise ValueError(f"DiffusionGemma fused QKV requires all 30 layer markers; missing layers: {missing}")

    fused_format = None
    for layer, key in enumerate(marker_keys):
        marker = sd[key]
        if not isinstance(marker, torch.Tensor) or marker.device.type != "cpu" or marker.dtype != torch.uint8 or marker.ndim != 1:
            raise ValueError(f"Invalid DiffusionGemma fused QKV marker tensor for layer {layer}")
        try:
            config = json.loads(marker.numpy().tobytes())
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"Invalid DiffusionGemma fused QKV marker JSON for layer {layer}") from error
        if not isinstance(config, dict):
            raise ValueError(f"Invalid DiffusionGemma fused QKV contract for layer {layer}: expected object")

        marker_format = config.get("format")
        if marker_format not in ("mxfp8", "int8_tensorwise"):
            raise ValueError(f"Invalid DiffusionGemma fused QKV format for layer {layer}: {marker_format}")
        if fused_format is None:
            fused_format = marker_format
        elif marker_format != fused_format:
            raise ValueError(
                f"DiffusionGemma fused QKV cannot mix {fused_format} and {marker_format} markers"
            )

        global_layer = layer % 6 == 5
        expected = {
            "format": marker_format,
            "artifact_contract": (
                _DIFFUSION_GEMMA_MXFP8_QKV_CONTRACT
                if marker_format == "mxfp8"
                else _DIFFUSION_GEMMA_INT8_QKV_CONTRACT
            ),
            "projection_order": ["q_proj", "k_proj"] if global_layer else ["q_proj", "k_proj", "v_proj"],
            "projection_splits": [8192, 1024] if global_layer else [4096, 2048, 2048],
        }
        if marker_format == "mxfp8":
            expected["full_precision_matrix_mult"] = False
        else:
            expected.update({"convrot": True, "convrot_groupsize": 256})
        mismatches = {
            name: (config.get(name), value)
            for name, value in expected.items()
            if config.get(name) != value
        }
        if mismatches:
            raise ValueError(f"Invalid DiffusionGemma fused QKV contract for layer {layer}: {mismatches}")
    return True


def diffusion_gemma_detect(clip_data):
    sd = clip_data[0]
    out = {}
    norm = sd.get("model.decoder.norm.weight", None)
    if norm is not None:
        out["dtype_llama"] = norm.dtype
    quantization_metadata = comfy.utils.detect_layer_quantization(sd, "")
    if quantization_metadata is not None:
        out["llama_quantization_metadata"] = quantization_metadata
    if "model.decoder.layers.0.experts.gate_proj.weight" in sd:
        out["unfused_experts"] = True
    if _detect_diffusion_gemma_fused_qkv(sd):
        out["fused_qkv"] = True
    return out


def diffusion_gemma_te(dtype_llama=None, llama_quantization_metadata=None, unfused_experts=False, fused_qkv=False):
    class DiffusionGemmaClipModel_(DiffusionGemmaClipModel):
        config_overrides = {"unfused_experts": unfused_experts, "fused_qkv": fused_qkv}

    class DiffusionGemmaTEModel_(sd1_clip.SD1ClipModel):
        supports_native_quantized_compute = llama_quantization_metadata is not None

        def __init__(self, device="cpu", dtype=None, model_options={}):
            if llama_quantization_metadata is not None:
                model_options = model_options.copy()
                model_options["quantization_metadata"] = llama_quantization_metadata
            if dtype_llama is not None:
                dtype = dtype_llama
            super().__init__(device=device, dtype=dtype, name="gemma4", clip_model=DiffusionGemmaClipModel_, model_options=model_options)
            self.current_weight_patches_uuid = None

        def generate(self, tokens, **kwargs):
            transformer = self.gemma4.transformer
            transformer._weight_patches_uuid = self.current_weight_patches_uuid
            for layer in transformer.model.decoder.layers:
                layer.experts.set_weight_patches_uuid(self.current_weight_patches_uuid)
            return super().generate(tokens, **kwargs)

        def memory_estimation_function(self, tokens, device=None):
            # logits/softmax fp32 buffers + dequantized expert bank + tied-embed cast
            return 4 * 1024 * 1024 * 1024
    return DiffusionGemmaTEModel_
