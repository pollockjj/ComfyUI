import os
import time as _time

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Any, Tuple
import math
from tqdm import tqdm
import comfy.utils

STATIC_KV = os.environ.get("COMFY_STATIC_KV", "0").lower() not in {"0", "false", "no", "off"}
STATIC_KV_FUSED_SAMPLER = os.environ.get("COMFY_STATIC_KV_FUSED_SAMPLER", "0").lower() not in {"0", "false", "no", "off"}
STATIC_KV_KEEP_INT8 = os.environ.get("COMFY_STATIC_KV_KEEP_INT8", "0").lower() not in {"0", "false", "no", "off"}
STATIC_KV_COMBO_KERNELS = os.environ.get("COMFY_STATIC_KV_COMBO_KERNELS", "0").lower() not in {"0", "false", "no", "off"}
STATIC_KV_FUSED_MLP = os.environ.get("COMFY_STATIC_KV_FUSED_MLP", "0").lower() not in {"0", "false", "no", "off"}

_STATIC_DECODE_COMBO_OPTIONS = {
    "triton.cudagraphs": True,
    "combo_kernels": True,
    "benchmark_combo_kernel": True,
}


def _compile_static_decode(fn):
    if STATIC_KV_COMBO_KERNELS:
        return torch.compile(fn, dynamic=False, options=_STATIC_DECODE_COMBO_OPTIONS)
    return torch.compile(fn, mode="reduce-overhead", dynamic=False)


def _mtp_compile_configuration():
    mode = os.environ.get("COMFY_MTP_COMPILE_MODE", "default")
    options = tuple(sorted(_STATIC_DECODE_COMBO_OPTIONS.items())) if STATIC_KV_COMBO_KERNELS else ()
    return mode, options


def _dequant_qt_chunked(weight, dtype, step=65535):
    # CUDA convrot dequant kernel caps rows per launch at gridDim.y; row scales
    # and rotation groups are row-local so row slices are exact
    qdata = weight._qdata
    if qdata.dim() != 2 or qdata.shape[0] <= step:
        return weight.dequantize().to(dtype)
    params = weight._params
    rows, cols = qdata.shape
    scale = params.scale
    w = torch.empty((rows, cols), dtype=dtype, device=qdata.device)
    for i in range(0, rows, step):
        n = min(step, rows - i)
        p = type(params)(scale=scale[i:i + n], orig_dtype=dtype, orig_shape=(n, cols),
                         convrot=getattr(params, "convrot", False),
                         convrot_groupsize=getattr(params, "convrot_groupsize", 256))
        w[i:i + n] = weight.layout_cls.dequantize(qdata[i:i + n], p)
    return w


def _freeze_resident_weights(root, ref_input):
    """Resolve each cast-managed module's effective (weight, bias) once, install
    them as plain on-device parameters and disable per-call cast so a captured
    decode step is pure tensor compute. Big 2D int8 weights stay quantized —
    F.linear dispatches them to the fused kernel. Returns a restore list."""
    import comfy.ops as _ops
    from comfy.quant_ops import QuantizedTensor as _QT
    frozen = []
    for m in root.modules():
        if not getattr(m, "comfy_cast_weights", False):
            continue
        w = getattr(m, "weight", None)
        if w is None:
            continue
        if len(getattr(m, "weight_function", ())) or len(getattr(m, "bias_function", ())):
            continue
        is_qt = isinstance(w, _QT) or isinstance(getattr(w, "data", None), _QT)
        # keeping big weights quantized (as the cast GPU-resident QT — the raw
        # parameter presents off-device under aimdo) avoids a ~13 GB bf16 working
        # set; the in-graph fused int8 GEMV carries its own per-call cost
        keep_qt = (STATIC_KV_KEEP_INT8 and is_qt
                   and getattr(m, "quant_format", None) == "int8_tensorwise"
                   and w.dim() == 2 and min(w.shape) >= 1536
                   and not isinstance(m, torch.nn.Embedding)
                   and getattr(m, "bias", None) is None)
        if keep_qt:
            rw, _ = _ops.cast_bias_weight(m, input=ref_input, offloadable=False)
            if isinstance(rw, _QT) and rw._qdata.is_cuda:
                frozen.append((m, m._parameters.get("weight"), None, True))
                m._parameters["weight"] = torch.nn.Parameter(rw, requires_grad=False)
                m.comfy_cast_weights = False
                continue
        warm_cache = getattr(m, "_int8_dequant_weight_cache", None)
        if warm_cache is not None and getattr(m, "bias", None) is None:
            # adopt the warm step's cached dequant: resident, stable, zero new
            # allocation; displaced quantized originals move to host so the bf16
            # working set fits VRAM (restored host-side; next cast re-stages)
            old = m._parameters.get("weight")
            if is_qt and old is not None:
                old = torch.nn.Parameter(old.data.to("cpu"), requires_grad=False)
            frozen.append((m, old, None, True))
            m._parameters["weight"] = torch.nn.Parameter(warm_cache[1], requires_grad=False)
            m.comfy_cast_weights = False
            continue
        rw, rb = _ops.cast_bias_weight(m, input=ref_input, offloadable=False)
        if isinstance(rw, _QT):
            rw = _dequant_qt_chunked(rw, ref_input.dtype)
        frozen.append((m, m._parameters.get("weight"), m._parameters.get("bias"), True))
        m._parameters["weight"] = torch.nn.Parameter(rw.contiguous(), requires_grad=False)
        if rb is not None:
            m._parameters["bias"] = torch.nn.Parameter(rb.contiguous(), requires_grad=False)
        m.comfy_cast_weights = False
    return frozen


def _restore_frozen_weights(frozen):
    for m, w, b, flag in frozen:
        if w is not None:
            m._parameters["weight"] = w
        if b is not None:
            m._parameters["bias"] = b
        m.comfy_cast_weights = flag

from comfy.ldm.modules.attention import optimized_attention_for_device
import comfy.model_management
import comfy.ops
import comfy.ldm.common_dit
import comfy.clip_model

from . import qwen_vl

@dataclass
class Llama2Config:
    vocab_size: int = 128320
    hidden_size: int = 4096
    intermediate_size: int = 14336
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    max_position_embeddings: int = 8192
    rms_norm_eps: float = 1e-5
    rope_theta: float = 500000.0
    transformer_type: str = "llama"
    head_dim = 128
    rms_norm_add = False
    mlp_activation = "silu"
    qkv_bias = False
    rope_dims = None
    q_norm = None
    k_norm = None
    rope_scale = None
    final_norm: bool = True
    lm_head: bool = False

@dataclass
class Mistral3Small24BConfig:
    vocab_size: int = 131072
    hidden_size: int = 5120
    intermediate_size: int = 32768
    num_hidden_layers: int = 40
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    max_position_embeddings: int = 8192
    rms_norm_eps: float = 1e-5
    rope_theta: float = 1000000000.0
    transformer_type: str = "llama"
    head_dim = 128
    rms_norm_add = False
    mlp_activation = "silu"
    qkv_bias = False
    rope_dims = None
    q_norm = None
    k_norm = None
    rope_scale = None
    final_norm: bool = True
    lm_head: bool = False

@dataclass
class Ministral3_3BConfig:
    vocab_size: int = 131072
    hidden_size: int = 3072
    intermediate_size: int = 9216
    num_hidden_layers: int = 26
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    max_position_embeddings: int = 262144
    rms_norm_eps: float = 1e-5
    rope_theta: float = 1000000.0
    transformer_type: str = "llama"
    head_dim = 128
    rms_norm_add = False
    mlp_activation = "silu"
    qkv_bias = False
    rope_dims = None
    q_norm = None
    k_norm = None
    rope_scale = None
    final_norm: bool = True
    lm_head: bool = False
    stop_tokens = [2]

@dataclass
class Qwen25_3BConfig:
    vocab_size: int = 151936
    hidden_size: int = 2048
    intermediate_size: int = 11008
    num_hidden_layers: int = 36
    num_attention_heads: int = 16
    num_key_value_heads: int = 2
    max_position_embeddings: int = 128000
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    transformer_type: str = "llama"
    head_dim = 128
    rms_norm_add = False
    mlp_activation = "silu"
    qkv_bias = True
    rope_dims = None
    q_norm = None
    k_norm = None
    rope_scale = None
    final_norm: bool = True
    lm_head: bool = False

@dataclass
class Qwen3_06BConfig:
    vocab_size: int = 151936
    hidden_size: int = 1024
    intermediate_size: int = 3072
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    transformer_type: str = "llama"
    head_dim = 128
    rms_norm_add = False
    mlp_activation = "silu"
    qkv_bias = False
    rope_dims = None
    q_norm = "gemma3"
    k_norm = "gemma3"
    rope_scale = None
    final_norm: bool = True
    lm_head: bool = False
    stop_tokens = [151643, 151645]

@dataclass
class Qwen3_06B_ACE15_Config:
    vocab_size: int = 151669
    hidden_size: int = 1024
    intermediate_size: int = 3072
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    transformer_type: str = "llama"
    head_dim = 128
    rms_norm_add = False
    mlp_activation = "silu"
    qkv_bias = False
    rope_dims = None
    q_norm = "gemma3"
    k_norm = "gemma3"
    rope_scale = None
    final_norm: bool = True
    lm_head: bool = False
    stop_tokens = [151643, 151645]

@dataclass
class Qwen3_2B_ACE15_lm_Config:
    vocab_size: int = 217204
    hidden_size: int = 2048
    intermediate_size: int = 6144
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    max_position_embeddings: int = 40960
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    transformer_type: str = "llama"
    head_dim = 128
    rms_norm_add = False
    mlp_activation = "silu"
    qkv_bias = False
    rope_dims = None
    q_norm = "gemma3"
    k_norm = "gemma3"
    rope_scale = None
    final_norm: bool = True
    lm_head: bool = False
    stop_tokens = [151643, 151645]

@dataclass
class Qwen3_4B_ACE15_lm_Config:
    vocab_size: int = 217204
    hidden_size: int = 2560
    intermediate_size: int = 9728
    num_hidden_layers: int = 36
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    max_position_embeddings: int = 40960
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    transformer_type: str = "llama"
    head_dim = 128
    rms_norm_add = False
    mlp_activation = "silu"
    qkv_bias = False
    rope_dims = None
    q_norm = "gemma3"
    k_norm = "gemma3"
    rope_scale = None
    final_norm: bool = True
    lm_head: bool = False
    stop_tokens = [151643, 151645]

@dataclass
class Qwen3_4BConfig:
    vocab_size: int = 151936
    hidden_size: int = 2560
    intermediate_size: int = 9728
    num_hidden_layers: int = 36
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    max_position_embeddings: int = 40960
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    transformer_type: str = "llama"
    head_dim = 128
    rms_norm_add = False
    mlp_activation = "silu"
    qkv_bias = False
    rope_dims = None
    q_norm = "gemma3"
    k_norm = "gemma3"
    rope_scale = None
    final_norm: bool = True
    lm_head: bool = False
    stop_tokens = [151643, 151645]

@dataclass
class Qwen3_8BConfig:
    vocab_size: int = 151936
    hidden_size: int = 4096
    intermediate_size: int = 12288
    num_hidden_layers: int = 36
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    max_position_embeddings: int = 40960
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    transformer_type: str = "llama"
    head_dim = 128
    rms_norm_add = False
    mlp_activation = "silu"
    qkv_bias = False
    rope_dims = None
    q_norm = "gemma3"
    k_norm = "gemma3"
    rope_scale = None
    final_norm: bool = True
    lm_head: bool = True
    stop_tokens = [151643, 151645]

@dataclass
class Qwen3VL_8BConfig(Qwen3_8BConfig):
    max_position_embeddings: int = 262144
    rope_theta: float = 5000000.0
    rope_dims = [24, 20, 20]
    interleaved_mrope = True

@dataclass
class Qwen3VL_4BConfig(Qwen3VL_8BConfig):
    hidden_size: int = 2560
    intermediate_size: int = 9728
    lm_head: bool = False  # 4B ties word embeddings

@dataclass
class Ovis25_2BConfig:
    vocab_size: int = 151936
    hidden_size: int = 2048
    intermediate_size: int = 6144
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    max_position_embeddings: int = 40960
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    transformer_type: str = "llama"
    head_dim = 128
    rms_norm_add = False
    mlp_activation = "silu"
    qkv_bias = False
    rope_dims = None
    q_norm = "gemma3"
    k_norm = "gemma3"
    rope_scale = None
    final_norm: bool = True
    lm_head: bool = False

@dataclass
class Qwen25_7BVLI_Config:
    vocab_size: int = 152064
    hidden_size: int = 3584
    intermediate_size: int = 18944
    num_hidden_layers: int = 28
    num_attention_heads: int = 28
    num_key_value_heads: int = 4
    max_position_embeddings: int = 128000
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    transformer_type: str = "llama"
    head_dim = 128
    rms_norm_add = False
    mlp_activation = "silu"
    qkv_bias = True
    rope_dims = [16, 24, 24]
    q_norm = None
    k_norm = None
    rope_scale = None
    final_norm: bool = True
    lm_head: bool = False

@dataclass
class Gemma2_2B_Config:
    vocab_size: int = 256000
    hidden_size: int = 2304
    intermediate_size: int = 9216
    num_hidden_layers: int = 26
    num_attention_heads: int = 8
    num_key_value_heads: int = 4
    max_position_embeddings: int = 8192
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    transformer_type: str = "gemma2"
    head_dim = 256
    rms_norm_add = True
    mlp_activation = "gelu_pytorch_tanh"
    qkv_bias = False
    rope_dims = None
    q_norm = None
    k_norm = None
    sliding_attention = None
    rope_scale = None
    final_norm: bool = True
    lm_head: bool = False
    stop_tokens = [1]

@dataclass
class Gemma3_4B_Config:
    vocab_size: int = 262208
    hidden_size: int = 2560
    intermediate_size: int = 10240
    num_hidden_layers: int = 34
    num_attention_heads: int = 8
    num_key_value_heads: int = 4
    max_position_embeddings: int = 131072
    rms_norm_eps: float = 1e-6
    rope_theta = [1000000.0, 10000.0]
    transformer_type: str = "gemma3"
    head_dim = 256
    rms_norm_add = True
    mlp_activation = "gelu_pytorch_tanh"
    qkv_bias = False
    rope_dims = None
    q_norm = "gemma3"
    k_norm = "gemma3"
    sliding_attention = [1024, 1024, 1024, 1024, 1024, False]
    rope_scale = [8.0, 1.0]
    final_norm: bool = True
    lm_head: bool = False
    stop_tokens = [1, 106]

GEMMA3_VISION_CONFIG = {"num_channels": 3, "hidden_act": "gelu_pytorch_tanh", "hidden_size": 1152, "image_size": 896, "intermediate_size": 4304, "model_type": "siglip_vision_model", "num_attention_heads": 16, "num_hidden_layers": 27, "patch_size": 14}

@dataclass
class Gemma3_4B_Vision_Config(Gemma3_4B_Config):
    vision_config = GEMMA3_VISION_CONFIG
    mm_tokens_per_image = 256

@dataclass
class Gemma3_12B_Config:
    vocab_size: int = 262208
    hidden_size: int = 3840
    intermediate_size: int = 15360
    num_hidden_layers: int = 48
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    max_position_embeddings: int = 131072
    rms_norm_eps: float = 1e-6
    rope_theta = [1000000.0, 10000.0]
    transformer_type: str = "gemma3"
    head_dim = 256
    rms_norm_add = True
    mlp_activation = "gelu_pytorch_tanh"
    qkv_bias = False
    rope_dims = None
    q_norm = "gemma3"
    k_norm = "gemma3"
    sliding_attention = [1024, 1024, 1024, 1024, 1024, False]
    rope_scale = [8.0, 1.0]
    final_norm: bool = True
    lm_head: bool = False
    vision_config = GEMMA3_VISION_CONFIG
    mm_tokens_per_image = 256
    stop_tokens = [1, 106]

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5, add=False, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(dim, device=device, dtype=dtype))
        self.add = add

    def forward(self, x: torch.Tensor):
        w = self.weight
        if self.add:
            w = w + 1.0

        return comfy.ldm.common_dit.rms_norm(x, w, self.eps)



def precompute_freqs_cis(head_dim, position_ids, theta, rope_scale=None, rope_dims=None, device=None, interleaved_mrope=False):
    if not isinstance(theta, list):
        theta = [theta]

    out = []
    for index, t in enumerate(theta):
        theta_numerator = torch.arange(0, head_dim, 2, device=device).float()
        inv_freq = 1.0 / (t ** (theta_numerator / head_dim))

        if rope_scale is not None:
            if isinstance(rope_scale, list):
                inv_freq /= rope_scale[index]
            else:
                inv_freq /= rope_scale

        inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        if rope_dims is not None and position_ids.shape[0] > 1 and interleaved_mrope:
            # Qwen3-VL interleaved MRoPE: T-freqs by default, H/W replace every 3rd dim.
            freqs_inter = freqs[0].clone()
            for axis_idx, offset in ((1, 1), (2, 2)):
                length = rope_dims[axis_idx] * 3
                idx = slice(offset, length, 3)
                freqs_inter[..., idx] = freqs[axis_idx, ..., idx]
            emb = torch.cat((freqs_inter, freqs_inter), dim=-1)
            cos = emb.cos().unsqueeze(0)
            sin = emb.sin().unsqueeze(0)
        else:
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
            if rope_dims is not None and position_ids.shape[0] > 1:
                mrope_section = rope_dims * 2
                cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1).unsqueeze(0)
                sin = torch.cat([m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1).unsqueeze(0)
            else:
                cos = cos.unsqueeze(1)
                sin = sin.unsqueeze(1)
        sin_split = sin.shape[-1] // 2
        out.append((cos, sin[..., : sin_split], -sin[..., sin_split :]))

    if len(out) == 1:
        return out[0]

    return out

def apply_rope(xq, xk, freqs_cis):
    org_dtype = xq.dtype
    cos = freqs_cis[0]
    sin = freqs_cis[1]
    nsin = freqs_cis[2]

    q_embed = (xq * cos)
    q_split = q_embed.shape[-1] // 2
    q_embed[..., : q_split].addcmul_(xq[..., q_split :], nsin)
    q_embed[..., q_split :].addcmul_(xq[..., : q_split], sin)

    k_embed = (xk * cos)
    k_split = k_embed.shape[-1] // 2
    k_embed[..., : k_split].addcmul_(xk[..., k_split :], nsin)
    k_embed[..., k_split :].addcmul_(xk[..., : k_split], sin)

    return q_embed.to(org_dtype), k_embed.to(org_dtype)


class Attention(nn.Module):
    def __init__(self, config: Llama2Config, device=None, dtype=None, ops: Any = None):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.hidden_size = config.hidden_size

        self.head_dim = config.head_dim
        self.inner_size = self.num_heads * self.head_dim

        ops = ops or nn
        self.q_proj = ops.Linear(config.hidden_size, self.inner_size, bias=config.qkv_bias, device=device, dtype=dtype)
        self.k_proj = ops.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.qkv_bias, device=device, dtype=dtype)
        self.v_proj = ops.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.qkv_bias, device=device, dtype=dtype)
        self.o_proj = ops.Linear(self.inner_size, config.hidden_size, bias=False, device=device, dtype=dtype)

        self.q_norm = None
        self.k_norm = None

        if config.q_norm == "gemma3":
            self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps, add=config.rms_norm_add, device=device, dtype=dtype)
        if config.k_norm == "gemma3":
            self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps, add=config.rms_norm_add, device=device, dtype=dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        freqs_cis: Optional[torch.Tensor] = None,
        optimized_attention=None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        sliding_window: Optional[int] = None,
    ):
        batch_size, seq_length, _ = hidden_states.shape

        xq = self.q_proj(hidden_states)
        xk = self.k_proj(hidden_states)
        xv = self.v_proj(hidden_states)

        xq = xq.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        xk = xk.view(batch_size, seq_length, self.num_kv_heads, self.head_dim).transpose(1, 2)
        xv = xv.view(batch_size, seq_length, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if self.q_norm is not None:
            xq = self.q_norm(xq)
        if self.k_norm is not None:
            xk = self.k_norm(xk)

        xq, xk = apply_rope(xq, xk, freqs_cis=freqs_cis)

        present_key_value = None
        if past_key_value is not None:
            index = 0
            num_tokens = xk.shape[2]
            if len(past_key_value) > 0:
                past_key, past_value, index = past_key_value
                if past_key.shape[2] >= (index + num_tokens):
                    past_key[:, :, index:index + xk.shape[2]] = xk
                    past_value[:, :, index:index + xv.shape[2]] = xv
                    xk = past_key[:, :, :index + xk.shape[2]]
                    xv = past_value[:, :, :index + xv.shape[2]]
                    present_key_value = (past_key, past_value, index + num_tokens)
                else:
                    xk = torch.cat((past_key[:, :, :index], xk), dim=2)
                    xv = torch.cat((past_value[:, :, :index], xv), dim=2)
                    present_key_value = (xk, xv, index + num_tokens)
            else:
                present_key_value = (xk, xv, index + num_tokens)

            if sliding_window is not None and xk.shape[2] > sliding_window and seq_length == 1:
                xk = xk[:, :, -sliding_window:]
                xv = xv[:, :, -sliding_window:]
                attention_mask = attention_mask[..., -sliding_window:] if attention_mask is not None else None

        xk = xk.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
        xv = xv.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)

        output = optimized_attention(xq, xk, xv, self.num_heads, mask=attention_mask, skip_reshape=True)
        return self.o_proj(output), present_key_value

class MLP(nn.Module):
    def __init__(self, config: Llama2Config, device=None, dtype=None, ops: Any = None, intermediate_size=None):
        super().__init__()
        intermediate_size = intermediate_size or config.intermediate_size
        self.gate_proj = ops.Linear(config.hidden_size, intermediate_size, bias=False, device=device, dtype=dtype)
        self.up_proj = ops.Linear(config.hidden_size, intermediate_size, bias=False, device=device, dtype=dtype)
        self.down_proj = ops.Linear(intermediate_size, config.hidden_size, bias=False, device=device, dtype=dtype)
        if config.mlp_activation == "silu":
            self.activation = torch.nn.functional.silu
        elif config.mlp_activation == "gelu_pytorch_tanh":
            self.activation = lambda a: torch.nn.functional.gelu(a, approximate="tanh")
        self._gate_up_weight = None

    def forward(self, x):
        if self._gate_up_weight is not None:
            gate, up = torch.nn.functional.linear(x, self._gate_up_weight).chunk(2, dim=-1)
            return self.down_proj(self.activation(gate) * up)
        return self.down_proj(self.activation(self.gate_proj(x)) * self.up_proj(x))


def _fuse_mlp_gate_up_projections(root, frozen):
    frozen_weights = {id(module): weight for module, weight, _, _ in frozen}
    for module in root.modules():
        if not isinstance(module, MLP) or module._gate_up_weight is not None:
            continue
        gate_weight = module.gate_proj.weight
        up_weight = module.up_proj.weight
        projections = (module.gate_proj, module.up_proj)
        if (gate_weight.ndim != 2 or gate_weight.shape != up_weight.shape
                or gate_weight.dtype != torch.bfloat16 or up_weight.dtype != torch.bfloat16
                or not gate_weight.is_cuda or not up_weight.is_cuda
                or any(projection.bias is not None or projection.comfy_cast_weights
                       or len(getattr(projection, "weight_function", ()))
                       or len(getattr(projection, "bias_function", ()))
                       or getattr(projection, "_int8_dequant_weight_cache", None) is not None
                       for projection in projections)
                or any(getattr(frozen_weights.get(id(projection)), "is_cuda", False)
                       for projection in projections)):
            raise RuntimeError("static fused MLP requires plain resident BF16 gate/up weights")
        gate_rows = gate_weight.shape[0]
        packed = torch.cat((gate_weight, up_weight), dim=0).contiguous()
        torch._dynamo.mark_static_address(packed, guard=True)
        module._gate_up_weight = packed
        module.gate_proj._parameters["weight"] = torch.nn.Parameter(packed[:gate_rows], requires_grad=False)
        module.up_proj._parameters["weight"] = torch.nn.Parameter(packed[gate_rows:], requires_grad=False)
        storage = packed.untyped_storage().data_ptr()
        if (not module.gate_proj.weight.is_contiguous() or not module.up_proj.weight.is_contiguous()
                or module.gate_proj.weight.untyped_storage().data_ptr() != storage
                or module.up_proj.weight.untyped_storage().data_ptr() != storage):
            raise RuntimeError("static fused MLP weight aliases were not preserved")
        del gate_weight, up_weight, packed

class TransformerBlock(nn.Module):
    def __init__(self, config: Llama2Config, index, device=None, dtype=None, ops: Any = None):
        super().__init__()
        self.self_attn = Attention(config, device=device, dtype=dtype, ops=ops)
        self.mlp = MLP(config, device=device, dtype=dtype, ops=ops)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, device=device, dtype=dtype)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, device=device, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        freqs_cis: Optional[torch.Tensor] = None,
        optimized_attention=None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        # Self Attention
        residual = x
        x = self.input_layernorm(x)
        x, present_key_value = self.self_attn(
            hidden_states=x,
            attention_mask=attention_mask,
            freqs_cis=freqs_cis,
            optimized_attention=optimized_attention,
            past_key_value=past_key_value,
        )
        x = residual + x

        # MLP
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x

        return x, present_key_value

class TransformerBlockGemma2(nn.Module):
    def __init__(self, config: Llama2Config, index, device=None, dtype=None, ops: Any = None):
        super().__init__()
        self.self_attn = Attention(config, device=device, dtype=dtype, ops=ops)
        self.mlp = MLP(config, device=device, dtype=dtype, ops=ops)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, add=config.rms_norm_add, device=device, dtype=dtype)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, add=config.rms_norm_add, device=device, dtype=dtype)
        self.pre_feedforward_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, add=config.rms_norm_add, device=device, dtype=dtype)
        self.post_feedforward_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, add=config.rms_norm_add, device=device, dtype=dtype)

        if config.sliding_attention is not None:
            self.sliding_attention = config.sliding_attention[index % len(config.sliding_attention)]
        else:
            self.sliding_attention = False

        self.transformer_type = config.transformer_type

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        freqs_cis: Optional[torch.Tensor] = None,
        optimized_attention=None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        sliding_window = None
        if self.transformer_type == 'gemma3':
            if self.sliding_attention:
                sliding_window = self.sliding_attention
                if x.shape[1] > self.sliding_attention:
                    sliding_mask = torch.full((x.shape[1], x.shape[1]), torch.finfo(x.dtype).min, device=x.device, dtype=x.dtype)
                    sliding_mask.tril_(diagonal=-self.sliding_attention)
                    if attention_mask is not None:
                        attention_mask = attention_mask + sliding_mask
                    else:
                        attention_mask = sliding_mask
                freqs_cis = freqs_cis[1]
            else:
                freqs_cis = freqs_cis[0]

        # Self Attention
        residual = x
        x = self.input_layernorm(x)
        x, present_key_value = self.self_attn(
            hidden_states=x,
            attention_mask=attention_mask,
            freqs_cis=freqs_cis,
            optimized_attention=optimized_attention,
            past_key_value=past_key_value,
            sliding_window=sliding_window,
        )

        x = self.post_attention_layernorm(x)
        x = residual + x

        # MLP
        residual = x
        x = self.pre_feedforward_layernorm(x)
        x = self.mlp(x)
        x = self.post_feedforward_layernorm(x)
        x = residual + x

        return x, present_key_value

def _make_scaled_embedding(ops, vocab_size, hidden_size, scale, device, dtype):
    class ScaledEmbedding(ops.Embedding):
        def forward(self, input_ids, out_dtype=None):
            return super().forward(input_ids, out_dtype=out_dtype) * scale
    return ScaledEmbedding(vocab_size, hidden_size, device=device, dtype=dtype)


class Llama2_(nn.Module):
    def __init__(self, config, device=None, dtype=None, ops=None):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size

        if self.config.transformer_type == "gemma2" or self.config.transformer_type == "gemma3":
            transformer = TransformerBlockGemma2
            self.embed_tokens = _make_scaled_embedding(ops, config.vocab_size, config.hidden_size, config.hidden_size ** 0.5, device, dtype)
        else:
            transformer = TransformerBlock
            self.embed_tokens = ops.Embedding(config.vocab_size, config.hidden_size, device=device, dtype=dtype)

        self.layers = nn.ModuleList([
            transformer(config, index=i, device=device, dtype=dtype, ops=ops)
            for i in range(config.num_hidden_layers)
        ])

        if config.final_norm:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, add=config.rms_norm_add, device=device, dtype=dtype)
        else:
            self.norm = None

        if config.lm_head:
            self.lm_head = ops.Linear(config.hidden_size, config.vocab_size, bias=False, device=device, dtype=dtype)

    def get_past_len(self, past_key_values):
        return past_key_values[0][2]

    def compute_freqs_cis(self, position_ids, device):
        return precompute_freqs_cis(self.config.head_dim,
                                    position_ids,
                                    self.config.rope_theta,
                                    self.config.rope_scale,
                                    self.config.rope_dims,
                                    interleaved_mrope=getattr(self.config, "interleaved_mrope", False),
                                    device=device)

    def forward(self, x, attention_mask=None, embeds=None, num_tokens=None, intermediate_output=None, final_layer_norm_intermediate=True,
                dtype=None, position_ids=None, embeds_info=[], past_key_values=None, input_ids=None,deepstack_embeds=None, visual_pos_masks=None):
        if embeds is not None:
            x = embeds
        else:
            x = self.embed_tokens(x, out_dtype=dtype)

        seq_len = x.shape[1]
        past_len = 0
        if past_key_values is not None and len(past_key_values) > 0:
            past_len = self.get_past_len(past_key_values)

        if position_ids is None:
            position_ids = torch.arange(past_len, past_len + seq_len, device=x.device).unsqueeze(0)

        freqs_cis = self.compute_freqs_cis(position_ids, x.device)

        mask = None
        if attention_mask is not None:
            mask = 1.0 - attention_mask.to(x.dtype).reshape((attention_mask.shape[0], 1, -1, attention_mask.shape[-1])).expand(attention_mask.shape[0], 1, seq_len, attention_mask.shape[-1])
            mask = mask.masked_fill(mask.to(torch.bool), torch.finfo(x.dtype).min / 4)

        if seq_len > 1:
            causal_mask = torch.empty(past_len + seq_len, past_len + seq_len, dtype=x.dtype, device=x.device).fill_(torch.finfo(x.dtype).min / 4).triu_(1)
            if mask is not None:
                mask += causal_mask
            else:
                mask = causal_mask

        optimized_attention = optimized_attention_for_device(x.device, mask=mask is not None, small_input=True)

        intermediate = None
        all_intermediate = None
        only_layers = None
        if intermediate_output is not None:
            if isinstance(intermediate_output, list):
                all_intermediate = []
                only_layers = set(intermediate_output)
            elif intermediate_output == "all":
                all_intermediate = []
                intermediate_output = None
            elif intermediate_output < 0:
                intermediate_output = len(self.layers) + intermediate_output

        next_key_values = []
        for i, layer in enumerate(self.layers):
            if all_intermediate is not None:
                if only_layers is None or (i in only_layers):
                    all_intermediate.append(x.unsqueeze(1).clone())

            past_kv = None
            if past_key_values is not None:
                past_kv = past_key_values[i] if len(past_key_values) > 0 else []

            x, current_kv = layer(
                x=x,
                attention_mask=mask,
                freqs_cis=freqs_cis,
                optimized_attention=optimized_attention,
                past_key_value=past_kv,
            )

            if current_kv is not None:
                next_key_values.append(current_kv)

            # DeepStack: add per-layer visual features into the first len() decoder layers at image positions (Qwen3-VL)
            if deepstack_embeds is not None and i < len(deepstack_embeds):
                x[visual_pos_masks] = x[visual_pos_masks] + deepstack_embeds[i].to(x)

            if i == intermediate_output:
                intermediate = x.clone()

        if self.norm is not None:
            x = self.norm(x)

        if all_intermediate is not None:
            if only_layers is None or ((i + 1) in only_layers):
                all_intermediate.append(x.unsqueeze(1).clone())

        if all_intermediate is not None:
            intermediate = torch.cat(all_intermediate, dim=1)

        if intermediate is not None and final_layer_norm_intermediate and self.norm is not None:
            intermediate = self.norm(intermediate)

        if len(next_key_values) > 0:
            return x, intermediate, next_key_values
        else:
            return x, intermediate


class Gemma3MultiModalProjector(torch.nn.Module):
    def __init__(self, config, dtype, device, operations):
        super().__init__()

        self.mm_input_projection_weight = nn.Parameter(
            torch.empty(config.vision_config["hidden_size"], config.hidden_size, device=device, dtype=dtype)
        )

        self.mm_soft_emb_norm = RMSNorm(config.vision_config["hidden_size"], eps=config.rms_norm_eps, add=config.rms_norm_add, device=device, dtype=dtype)

        self.patches_per_image = int(config.vision_config["image_size"] // config.vision_config["patch_size"])
        self.tokens_per_side = int(config.mm_tokens_per_image**0.5)
        self.kernel_size = self.patches_per_image // self.tokens_per_side
        self.avg_pool = nn.AvgPool2d(kernel_size=self.kernel_size, stride=self.kernel_size)

    def forward(self, vision_outputs: torch.Tensor):
        batch_size, _, seq_length = vision_outputs.shape

        reshaped_vision_outputs = vision_outputs.transpose(1, 2)
        reshaped_vision_outputs = reshaped_vision_outputs.reshape(
            batch_size, seq_length, self.patches_per_image, self.patches_per_image
        )
        reshaped_vision_outputs = reshaped_vision_outputs.contiguous()

        pooled_vision_outputs = self.avg_pool(reshaped_vision_outputs)
        pooled_vision_outputs = pooled_vision_outputs.flatten(2)
        pooled_vision_outputs = pooled_vision_outputs.transpose(1, 2)

        normed_vision_outputs = self.mm_soft_emb_norm(pooled_vision_outputs)

        projected_vision_outputs = torch.matmul(normed_vision_outputs, comfy.model_management.cast_to_device(self.mm_input_projection_weight, device=normed_vision_outputs.device, dtype=normed_vision_outputs.dtype))
        return projected_vision_outputs.type_as(vision_outputs)


class BaseLlama:
    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, embeddings):
        self.model.embed_tokens = embeddings

    def forward(self, input_ids, *args, **kwargs):
        return self.model(input_ids, *args, **kwargs)

class BaseGenerate:
    def logits(self, x):
        input = x[:, -1:]
        if hasattr(self.model, "lm_head"):
            module = self.model.lm_head
        else:
            module = self.model.embed_tokens

        offload_stream = None
        if module.comfy_cast_weights:
            weight, _, offload_stream = comfy.ops.cast_bias_weight(module, input, offloadable=True)
        else:
            weight = self.model.embed_tokens.weight.to(x)

        params = getattr(weight, "_params", None)
        if params is not None and getattr(params, "convrot", False) and weight.is_cuda and weight.shape[0] > 32768:
            # A vocab-sized convrot table cannot be dequantized in one kernel call
            # (the CUDA dequant rejects row counts this large); dequantize once in
            # row chunks (the rotation runs along k, rows are independent) and keep
            # the compute-dtype table for every subsequent logits call.
            cache = getattr(module, "_logits_dequant_cache", None)
            key = (weight.device, input.dtype)
            if cache is None or cache[0] != key:
                from comfy_kitchen.backends.cuda import DTYPE_TO_CODE
                q, sc, gs = weight._qdata, params.scale, params.convrot_groupsize
                code = DTYPE_TO_CODE[input.dtype]
                dq = torch.cat([torch.ops.comfy_kitchen.dequantize_int8_convrot_weight_dtype(
                    q[i:i + 32768].contiguous(), sc[i:i + 32768].contiguous(), gs, code)
                    for i in range(0, q.shape[0], 32768)], dim=0)
                module._logits_dequant_cache = (key, dq)
                cache = module._logits_dequant_cache
            out = torch.nn.functional.linear(input, cache[1], None)
            comfy.ops.uncast_bias_weight(module, weight, None, offload_stream)
            return out

        x = torch.nn.functional.linear(input, weight, None)

        comfy.ops.uncast_bias_weight(module, weight, None, offload_stream)
        return x

    def init_kv_cache(self, batch, max_cache_len, device, execution_dtype):
        model_config = self.model.config
        past_key_values = []
        for x in range(model_config.num_hidden_layers):
            past_key_values.append((torch.empty([batch, model_config.num_key_value_heads, max_cache_len, model_config.head_dim], device=device, dtype=execution_dtype),
                                    torch.empty([batch, model_config.num_key_value_heads, max_cache_len, model_config.head_dim], device=device, dtype=execution_dtype), 0))
        return past_key_values

    def generate(self, embeds=None, do_sample=True, max_length=256, temperature=1.0, top_k=50, top_p=0.9, min_p=0.0, repetition_penalty=1.0, seed=42, stop_tokens=None, initial_tokens=[], execution_dtype=None, min_tokens=0, presence_penalty=0.0, initial_input_ids=None, position_ids=None, deepstack_embeds=None, visual_pos_masks=None):
        device = embeds.device

        if stop_tokens is None:
            stop_tokens = self.model.config.stop_tokens

        if execution_dtype is None:
            if comfy.model_management.should_use_bf16(device):
                execution_dtype = torch.bfloat16
            else:
                execution_dtype = torch.float32
        embeds = embeds.to(execution_dtype)

        if embeds.ndim == 2:
            embeds = embeds.unsqueeze(0)

        max_cache_len = embeds.shape[1] + max_length
        past_key_values = self.init_kv_cache(embeds.shape[0], max_cache_len, device, execution_dtype)

        generator = torch.Generator(device=device).manual_seed(seed) if do_sample else None

        generated_token_ids = []
        pbar = comfy.utils.ProgressBar(max_length)

        # MRoPE: prefill uses explicit 3D position_ids, decode continues from the last position
        next_pos = int(position_ids[:, -1].max()) + 1 if position_ids is not None else None

        # Static-KV captured decode: prefill runs the dynamic path, then caches
        # convert to fixed-shape buffers and the per-token step is compiled with
        # reduce-overhead (CUDA graphs). Requires explicit tensor position ids.
        static_kv = STATIC_KV and hasattr(self, "convert_kv_to_static") and embeds.is_cuda and position_ids is None
        prompt_len = embeds.shape[1]
        compiled_step = None
        static_pos = None
        frozen_weights = []

        # Generation loop
        current_input_ids = initial_input_ids
        try:
            return self._generate_loop(embeds, do_sample, max_length, temperature, top_k, top_p, min_p,
                                       repetition_penalty, seed, stop_tokens, initial_tokens, execution_dtype,
                                       presence_penalty, initial_input_ids, position_ids, deepstack_embeds,
                                       visual_pos_masks, past_key_values, generator, pbar, next_pos, device,
                                       static_kv, prompt_len, max_cache_len, frozen_weights)
        finally:
            if frozen_weights:
                _restore_frozen_weights(frozen_weights)

    def _generate_loop(self, embeds, do_sample, max_length, temperature, top_k, top_p, min_p,
                       repetition_penalty, seed, stop_tokens, initial_tokens, execution_dtype,
                       presence_penalty, initial_input_ids, position_ids, deepstack_embeds,
                       visual_pos_masks, past_key_values, generator, pbar, next_pos, device,
                       static_kv, prompt_len, max_cache_len, frozen_weights):
        compiled_step = None
        compiled_fused = None
        static_pos = None
        generated_token_ids = []
        # penalties need per-token history on host; without them the token chain
        # stays on device and syncs once per SYNC_EVERY steps
        batched_sync = repetition_penalty == 1.0 and not presence_penalty
        # MTP speculative decode: batch 1, no history penalties, standard positions.
        # Sampling is supported: each verify position is sampled with the normal
        # sampler and a draft is accepted only if it equals the sampled token, so the
        # output distribution is exactly the backbone sampler's. With static KV the
        # cycle runs eagerly over the fixed-shape caches (batched ring append +
        # rollback); without it, over the dynamic tuple caches. Anything else falls
        # through to the normal loop.
        if (batched_sync and position_ids is None
                and deepstack_embeds is None and embeds.shape[0] == 1):
            loader = getattr(self, "_load_mtp_drafter", None)
            drafter = loader(device, execution_dtype) if loader is not None else None
            if drafter is not None:
                return self._generate_speculative(drafter, embeds, max_length, stop_tokens,
                                                  execution_dtype, initial_input_ids,
                                                  past_key_values, pbar, device,
                                                  do_sample, temperature, top_k, top_p,
                                                  min_p, generator,
                                                  static_kv=static_kv, prompt_len=prompt_len,
                                                  max_cache_len=max_cache_len)
        device_tokens = []
        SYNC_EVERY = max_length if not stop_tokens else 8
        current_input_ids = initial_input_ids
        cur_bucket = 0
        for step in tqdm(range(max_length), desc="Generating tokens"):
            if compiled_fused is not None or compiled_step is not None:
                nb = min(-(-(prompt_len + step + 1) // 512) * 512, max_cache_len)
                if nb != cur_bucket:
                    for _kv in past_key_values:
                        if hasattr(_kv, "slots"):
                            _kv.bucket = nb
                    cur_bucket = nb
            if static_kv and step == 1:
                # convert to fixed-shape caches; step 1 runs eagerly so every
                # module-level M=1 weight cache warms at a stable address
                # outside the cudagraph pool. Caches, frozen weights and compiled
                # variants persist on self so items after the first pay zero compiles.
                runner = getattr(self, "_static_runner", None)
                rkey = (temperature, top_k, top_p, min_p, do_sample, execution_dtype, embeds.shape[0])
                if runner is not None and runner["key"] == rkey and prompt_len + max_length <= runner["max_len"]:
                    # kv-sharing layers carry empty tuples, not StaticLayerKV
                    for kv_new, kv_p in zip(past_key_values, runner["caches"]):
                        if hasattr(kv_p, "reset"):
                            kv_p.reset(kv_new)
                    past_key_values = runner["caches"]
                else:
                    plen = max(max_cache_len, 12288)
                    past_key_values = self.convert_kv_to_static(past_key_values, plen)
                    runner = {"key": rkey, "caches": past_key_values, "max_len": plen,
                              "compiled_fused": None, "compiled_step": None, "frozen": []}
                    self._static_runner = runner
                position_ids = torch.tensor([[prompt_len]], device=device)
                torch._dynamo.config.recompile_limit = 64  # one variant per attention bucket
            if static_kv and step == 2:
                runner = self._static_runner
                if not runner["frozen"]:
                    # frozen weights stay installed for the life of the runner;
                    # the finally-restore only fires if the runner was never built
                    runner["frozen"] = _freeze_resident_weights(self.model, embeds.reshape(-1)[:1])
                    prepare_static_weights = getattr(self, "_prepare_static_decode_weights", None)
                    if prepare_static_weights is not None:
                        prepare_static_weights(runner["frozen"])
                    if STATIC_KV_FUSED_MLP:
                        _fuse_mlp_gate_up_projections(self.model, runner["frozen"])
                static_pos = torch.tensor([[prompt_len + 1]], device=device)
                static_past = runner["caches"]

                if runner["compiled_step"] is None:
                    def _decode_step_fused(e, ids, pos):
                        x, _, _ = self.model.forward(None, embeds=e, attention_mask=None, past_key_values=static_past, input_ids=ids, position_ids=pos)
                        logits = self.logits(x)[:, -1]
                        tok = self._sample_in_graph(logits, temperature, top_k, top_p, min_p, do_sample)
                        e2 = self.model.embed_tokens(tok).to(execution_dtype)
                        return tok, e2
                    runner["compiled_fused"] = _compile_static_decode(_decode_step_fused)

                    def _decode_step(e, ids, pos):
                        x, _, _ = self.model.forward(None, embeds=e, attention_mask=None, past_key_values=static_past, input_ids=ids, position_ids=pos)
                        return self.logits(x)[:, -1]
                    runner["compiled_step"] = _compile_static_decode(_decode_step)

                if STATIC_KV_FUSED_SAMPLER and batched_sync:
                    if generator is not None:
                        torch.cuda.set_rng_state(generator.get_state(), device=device)
                    else:
                        torch.cuda.manual_seed(0)
                    compiled_fused = runner["compiled_fused"]
                else:
                    compiled_step = runner["compiled_step"]

            if compiled_fused is not None:
                # whole token step (forward + sample + embed) replays as one graph;
                # outputs are cloned off the graph pool before feeding the next replay
                tok, e2 = compiled_fused(embeds, current_input_ids, static_pos)
                static_pos = static_pos + 1
                tok = tok.clone()
                embeds = e2.clone()
                device_tokens.append(tok)
                current_input_ids = tok if initial_input_ids is not None else None
                if len(device_tokens) >= SYNC_EVERY or step == max_length - 1:
                    ids = torch.cat(device_tokens, dim=1)[0].tolist()
                    device_tokens.clear()
                    pbar.update(len(ids))
                    hit = next((j for j, t in enumerate(ids) if t in stop_tokens), None)
                    if hit is not None:
                        generated_token_ids.extend(ids[:hit + 1])
                        break
                    generated_token_ids.extend(ids)
                continue

            if compiled_step is not None:
                # clone: the sampler mutates logits in place and a graph output
                # buffer is rewritten by the next replay
                logits = compiled_step(embeds, current_input_ids, static_pos).clone()
                static_pos = static_pos + 1
                if batched_sync:
                    # device-side token chain: one host sync per SYNC_EVERY steps
                    # instead of per token; post-stop tokens are truncated below
                    next_token = self.sample_token(logits, temperature, top_k, top_p, min_p, repetition_penalty, initial_tokens, generator, do_sample=do_sample, presence_penalty=presence_penalty)
                    device_tokens.append(next_token)
                    embeds = self.model.embed_tokens(next_token).to(execution_dtype)
                    current_input_ids = next_token if initial_input_ids is not None else None
                    if len(device_tokens) >= SYNC_EVERY or step == max_length - 1:
                        ids = torch.cat(device_tokens, dim=1)[0].tolist()
                        device_tokens.clear()
                        pbar.update(len(ids))
                        hit = next((j for j, t in enumerate(ids) if t in stop_tokens), None)
                        if hit is not None:
                            generated_token_ids.extend(ids[:hit + 1])
                            break
                        generated_token_ids.extend(ids)
                    continue
                x = None
            else:
                # DeepStack visual features are injected on the prefill only; gemma4's forward lacks these kwargs.
                extra = {}
                if step == 0 and deepstack_embeds is not None:
                    extra["deepstack_embeds"] = deepstack_embeds
                    extra["visual_pos_masks"] = visual_pos_masks
                x, _, past_key_values = self.model.forward(None, embeds=embeds, attention_mask=None, past_key_values=past_key_values, input_ids=current_input_ids, position_ids=position_ids, **extra)
                logits = self.logits(x)[:, -1]
            next_token = self.sample_token(logits, temperature, top_k, top_p, min_p, repetition_penalty, initial_tokens + generated_token_ids, generator, do_sample=do_sample, presence_penalty=presence_penalty)
            token_id = next_token[0].item()
            generated_token_ids.append(token_id)

            embeds = self.model.embed_tokens(next_token).to(execution_dtype)
            current_input_ids = next_token if initial_input_ids is not None else None
            if next_pos is not None:  # advance MRoPE position for the next (decode) step
                position_ids = torch.tensor([[next_pos]], device=device)
                next_pos += 1
            pbar.update(1)

            if token_id in stop_tokens:
                break

        return generated_token_ids

    def _spec_sliding_margins(self, past, windows, gamma):
        # Sliding caches evict their oldest entries when a verify batch lands; if drafts
        # are then rejected, the tail rollback alone leaves the window short. Snapshot the
        # slice of pre-verify entries that any rollback could need to re-prepend.
        margins = {}
        for i, w in enumerate(windows):
            kv = past[i]
            if w is None or not kv or len(kv) != 3:
                continue
            k, v, c = kv
            lp = k.shape[2]
            if lp + gamma + 1 <= w - 1:
                continue
            ms = max(0, lp + 1 - (w - 1))
            me = max(0, lp + gamma + 1 - (w - 1))
            margins[i] = (k[:, :, ms:me].clone(), v[:, :, ms:me].clone(), lp, c, ms)
        return margins

    def _spec_rollback(self, vpast, windows, margins, gamma, acc):
        # Exact post-acceptance cache: keep the acc accepted drafts + the correction,
        # drop the gamma-acc rejected tail, and re-prepend evicted sliding entries.
        rej = gamma - acc
        out = []
        for i, w in enumerate(windows):
            kv = vpast[i]
            if not kv or len(kv) != 3:
                out.append(kv)
                continue
            k, v, c = kv
            if w is None or i not in margins:
                n = k.shape[2] - rej
                out.append((k[:, :, :n].contiguous(), v[:, :, :n].contiguous(), c - rej) if rej else kv)
                continue
            mk, mv, lp, c_pre, ms = margins[i]
            ld = min(w - 1, lp + acc + 1)
            head_start = max(0, lp + acc + 1 - ld)
            keep = k.shape[2] - rej
            nk = torch.cat([mk[:, :, head_start - ms:], k[:, :, :keep]], dim=2).contiguous()
            nv = torch.cat([mv[:, :, head_start - ms:], v[:, :, :keep]], dim=2).contiguous()
            out.append((nk, nv, c_pre + acc + 1))
        return out

    def _generate_speculative(self, drafter, embeds, max_length, stop_tokens, execution_dtype,
                              initial_input_ids, past_key_values, pbar, device,
                              do_sample=False, temperature=1.0, top_k=50, top_p=0.9,
                              min_p=0.0, generator=None, static_kv=False, prompt_len=0,
                              max_cache_len=0):
        # Speculative decode: draft gamma tokens with the MTP head, verify all of them
        # plus the last confirmed token in ONE backbone forward, emit the accepted
        # prefix and the backbone's own next token. One backbone forward per cycle.
        # Under sampling, each verify position is sampled with the normal sampler and a
        # draft survives only if it equals the sample, so outputs follow the backbone
        # sampler's distribution exactly (per-seed streams differ from the sequential
        # loop because rejected positions also consumed RNG).
        gamma = int(os.environ.get("COMFY_MTP_GAMMA", "4"))
        stats = os.environ.get("COMFY_MTP_STATS", "0") not in {"0", "false", "no"}

        def pick(logits):
            return self.sample_token(logits, temperature, top_k, top_p, min_p, 1.0, [],
                                     generator, do_sample=do_sample, presence_penalty=0.0)

        windows = [ly.sliding_attention if getattr(ly, "sliding_attention", False) else None
                   for ly in self.model.layers]
        x, _, past = self.model.forward(None, embeds=embeds, attention_mask=None,
                                        past_key_values=past_key_values, input_ids=initial_input_ids)
        h_last = x[:, -1:].clone(memory_format=torch.contiguous_format)
        last_tok = pick(self.logits(x)[:, -1])
        p = embeds.shape[1]
        statics = []
        capture = (static_kv and embeds.is_cuda
                   and os.environ.get("COMFY_MTP_CAPTURE", "0") not in {"0", "false", "no"})
        if static_kv:
            runner = getattr(self, "_mtp_cap_runner", None)
            rkey = (gamma, temperature, top_k, top_p, min_p, do_sample, execution_dtype,
                    _mtp_compile_configuration())
            if capture and runner is not None and runner["key"] == rkey and p + max_length <= runner["max_len"]:
                if os.environ.get("COMFY_MTP_STATS", "0") not in {"0", "false", "no"}:
                    print("[MTP-CAP] runner reused")
                for kv_new, kv_p in zip(past, runner["past"]):
                    if hasattr(kv_p, "reset"):
                        kv_p.reset(kv_new)
                past = runner["past"]
                statics = runner["statics"]
            else:
                past = self.convert_kv_to_static(past, max(max_cache_len, 12288))
                statics = [kv for kv in past if hasattr(kv, "rollback")]
                runner = None
        if capture and statics:
            return self._generate_speculative_captured(
                drafter, max_length, stop_tokens, execution_dtype, past, statics, pbar,
                device, do_sample, temperature, top_k, top_p, min_p, generator,
                h_last, last_tok, p, gamma, stats, runner)
        out = [int(last_tok.item())]
        pbar.update(1)
        if out[-1] in stop_tokens:
            return out
        accepted = cycles = 0
        while len(out) < max_length:
            cycles += 1
            if statics:
                # rounded-up valid region; the attention path caps it per layer at slots
                nb = -(-(p + gamma + 1) // 512) * 512
                for kv in statics:
                    kv.bucket = nb
            drafts = []
            d_tok, d_hid, d_pos = last_tok, h_last, p
            for _ in range(gamma):
                d_tok, d_hid = drafter.draft(d_tok, d_hid, past, d_pos)
                d_pos += 1
                drafts.append(d_tok)
            batch = torch.cat([last_tok] + drafts, dim=1)
            vpos = torch.arange(p, p + gamma + 1, device=device).unsqueeze(0)
            margins = None if statics else self._spec_sliding_margins(past, windows, gamma)
            vx, _, vpast = self.model.forward(None, embeds=self.model.embed_tokens(batch).to(execution_dtype),
                                              attention_mask=None, past_key_values=past,
                                              position_ids=vpos, input_ids=batch)
            vpicked = [int(pick(self.logits(vx[:, i:i + 1])[:, -1]).item()) for i in range(gamma + 1)]
            acc = gamma
            for i in range(gamma):
                if int(drafts[i].item()) != vpicked[i]:
                    acc = i
                    break
            accepted += acc
            if statics:
                for kv in statics:
                    kv.rollback(gamma - acc)
                past = vpast
            else:
                past = self._spec_rollback(vpast, windows, margins, gamma, acc)
            h_last = vx[:, acc:acc + 1]
            emit = [int(drafts[i].item()) for i in range(acc)] + [vpicked[acc]]
            last_tok = torch.tensor([[emit[-1]]], device=device)
            p += acc + 1
            for t in emit:
                out.append(t)
                pbar.update(1)
                if t in stop_tokens or len(out) >= max_length:
                    print(f"[MTP] gamma={gamma} cycles={cycles} accepted={accepted} "
                          f"drafted={cycles * gamma} tokens/forward={len(out) / (cycles + 1):.3f}")
                    return out
        print(f"[MTP] gamma={gamma} cycles={cycles} accepted={accepted} "
              f"drafted={cycles * gamma} tokens/forward={len(out) / (cycles + 1):.3f}")
        return out

    def _generate_speculative_captured(self, drafter, max_length, stop_tokens, execution_dtype,
                                       past, statics, pbar, device, do_sample, temperature,
                                       top_k, top_p, min_p, generator, h_last, last_tok, p,
                                       gamma, stats, runner=None):
        # The whole speculative cycle — drafter x gamma, one M=gamma+1 backbone verify,
        # in-graph sampling, cumprod acceptance, where-based KV commit — runs as a single
        # reduce-overhead graph over the static caches. No data-dependent control flow:
        # `acc` stays a device scalar and state (pos, hidden, token) hands off in tensors.
        # Host syncs once per SYNC_EVERY cycles to emit tokens and check stops.
        if runner is None:
            # the tied logits weight must be a plain tensor for the in-graph linear:
            # bf16 models expose the Parameter directly; quantized convrot models have
            # the chunk-dequant cache warmed by the eager prefill logits call
            mod = self.model.lm_head if hasattr(self.model, "lm_head") else self.model.embed_tokens
            cap = getattr(self.model.config, "final_logit_softcapping", None)
            for t in drafter.w.values():
                torch._dynamo.mark_static_address(t)
            # frozen weights stay installed for the life of the runner (Phase-A pattern)
            frozen = _freeze_resident_weights(self.model, h_last.reshape(-1)[:1])
            prepare_static_weights = getattr(self, "_prepare_static_decode_weights", None)
            if prepare_static_weights is not None:
                prepare_static_weights(frozen)
            if STATIC_KV_FUSED_MLP:
                _fuse_mlp_gate_up_projections(self.model, frozen)
            lw = getattr(mod, "_logits_dequant_cache", None)
            logits_w = lw[1] if lw is not None else mod.weight

            def _pick_batch(lg):
                # sampler over the top-k slice only: with top_k active this is exactly
                # _sample_in_graph (topk output is sorted, so the top-p cumsum order
                # matches), and it avoids the vocab-wide scan inductor cannot codegen
                # at these row counts
                if not do_sample or temperature == 0.0:
                    return lg.argmax(-1)
                if temperature != 1.0:
                    lg = lg / temperature
                if top_k <= 0:
                    return self._sample_in_graph(lg, 1.0, 0, top_p, min_p, do_sample).view(-1)
                vals, idx = torch.topk(lg, min(top_k, lg.shape[-1]))
                if min_p > 0.0:
                    probs_bf = torch.softmax(vals, dim=-1)
                    vals = vals.masked_fill(probs_bf < min_p * probs_bf[..., :1], float("-inf"))
                if top_p < 1.0:
                    cum = torch.cumsum(torch.softmax(vals, dim=-1), dim=-1)
                    rm = cum > top_p
                    rm[..., 0] = False
                    vals = vals.masked_fill(rm, float("-inf"))
                # gumbel-argmax instead of multinomial: rand_like is philox-managed
                # under cudagraph replay, multinomial's generator state is not
                u = torch.rand_like(vals).clamp_min(1e-20)
                g = -torch.log((-torch.log(u)).clamp_min(1e-20))
                ch = (torch.log_softmax(vals, dim=-1) + g).argmax(-1, keepdim=True)
                return idx.gather(-1, ch).view(-1)

            def _cycle(lt, hl, pos_t):
                vpos = pos_t + torch.arange(gamma + 1, device=device).view(1, -1)
                drafts = []
                d_tok, d_hid = lt, hl
                for j in range(gamma):
                    d_tok, d_hid = drafter.draft(d_tok, d_hid, past, pos_t + j)
                    drafts.append(d_tok)
                drafts_t = torch.cat(drafts, dim=1)
                batch = torch.cat([lt, drafts_t], dim=1)
                e = self.model.embed_tokens(batch).to(execution_dtype)
                vx, _, _ = self.model.forward(None, embeds=e, attention_mask=None,
                                              past_key_values=past, position_ids=vpos, input_ids=batch)
                logits = torch.nn.functional.linear(vx, logits_w)
                if cap:
                    logits = cap * torch.tanh(logits / cap)
                picked = _pick_batch(logits[0].float())
                acc = (drafts_t[0] == picked[:gamma]).long().cumprod(0).sum()
                for kv in statics:
                    kv.commit(acc)
                new_tok = picked.index_select(0, acc.view(1)).view(1, 1)
                new_h = vx.index_select(1, acc.view(1))
                return drafts_t, picked, acc, new_tok, new_h, pos_t + acc + 1

            # per-graph memory pools: cudagraph trees' shared-pool liveness checks
            # (check_memory_pool) fail under the full ComfyUI runtime allocator,
            # which holds cross-item allocations the tree does not know about
            torch._inductor.config.triton.cudagraph_trees = False
            runner = {"key": (gamma, temperature, top_k, top_p, min_p, do_sample, execution_dtype,
                              _mtp_compile_configuration()),
                      "past": past, "statics": statics, "max_len": statics[0].slots if statics else 0,
                      "cycle": _cycle, "compiled": None, "warm": False, "frozen": frozen}
            for kv in statics:
                if kv.window is None:
                    runner["max_len"] = kv.slots
            self._mtp_cap_runner = runner

        out = [int(last_tok.item())]
        pbar.update(1)
        if out[-1] in stop_tokens:
            return out
        torch.cuda.manual_seed(generator.initial_seed() if generator is not None else 0)
        torch._dynamo.config.recompile_limit = 64
        pos_t = torch.tensor([[p]], device=device, dtype=torch.long)
        p_upper = p
        records = []
        accepted = 0
        cycles = 0
        if not runner["warm"]:
            # one eager cycle first: allocates the per-cache save buffers and warms
            # module caches at stable addresses before capture
            nb0 = -(-(p + gamma + 1) // 512) * 512
            for kv in statics:
                kv.bucket = nb0
            dts, pk, acc, last_tok, h_last, pos_t = [t.clone() for t in runner["cycle"](last_tok, h_last, pos_t)]
            records.append((dts, pk, acc))
            # default compile mode: cudagraph capture of the partitioned cycle aborts
            # (CUDA invalid argument) under the ComfyUI server's greenlet execution,
            # while plain inductor runs everywhere at ~85% of the cudagraph rate.
            # reduce-overhead stays available for standalone benchmarking.
            _mode = os.environ.get("COMFY_MTP_COMPILE_MODE", "default")
            if stats:
                print(f"[MTP-CAP] compile config={_mtp_compile_configuration()} "
                      f"stream={torch.cuda.current_stream(device)}")
            if _mode == "base":
                runner["compiled"] = _compile_static_decode(runner["cycle"])
            else:
                runner["compiled"] = torch.compile(
                    runner["cycle"], mode=_mode if _mode != "none" else None, dynamic=False
                )
            runner["warm"] = True
            p_upper = p + gamma + 1
            cycles = 1
        compiled = runner["compiled"]
        SYNC_EVERY = max_length if not stop_tokens else 8
        while True:
            pending = len(records) * (gamma + 1)
            if len(out) + pending >= max_length or len(records) >= SYNC_EVERY:
                for dts, pk, a in records:
                    a = int(a.item())
                    accepted += a
                    emit = dts[0].tolist()[:a] + [int(pk[a].item())]
                    for t in emit:
                        out.append(t)
                        pbar.update(1)
                        if t in stop_tokens or len(out) >= max_length:
                            print(f"[MTP-CAP] gamma={gamma} cycles={cycles} accepted={accepted} "
                                  f"drafted={cycles * gamma} "
                                  f"tokens/forward={len(out) / (cycles + 1):.3f}")
                            return out
                records.clear()
            cycles += 1
            nb = -(-(p_upper + gamma + 1) // 512) * 512
            for kv in statics:
                kv.bucket = nb
            if stats and cycles % 16 == 0:
                torch.cuda.synchronize()
                _t0 = _time.time()
            dts, pk, acc, nt, nh, np_ = compiled(last_tok, h_last, pos_t)
            if stats and cycles % 16 == 0:
                torch.cuda.synchronize()
                print(f"[MTP-CAP] cycle {cycles}: {(_time.time() - _t0) * 1000:.1f} ms")
            last_tok, h_last, pos_t = nt.clone(), nh.clone(), np_.clone()
            records.append((dts.clone(), pk.clone(), acc.clone()))
            # drop the pool-aliasing output refs before the next call: a bucket change
            # triggers a fresh capture, and cudagraph trees require no live allocations
            # from the pool at capture time (check_memory_pool fails otherwise)
            del dts, pk, acc, nt, nh, np_
            p_upper += gamma + 1

    def _sample_in_graph(self, logits, temperature, top_k, top_p, min_p, do_sample):
        # sample_token minus history penalties and the Generator object, so the whole
        # filter+sample chain stays capturable (philox RNG); seeded per item by the caller
        if not do_sample or temperature == 0.0:
            return torch.argmax(logits, dim=-1, keepdim=True)
        if temperature != 1.0:
            logits = logits / temperature
        if top_k > 0:
            logits, top_indices = torch.topk(logits, min(top_k, logits.shape[-1]))
            if min_p > 0.0:
                probs_before_filter = torch.nn.functional.softmax(logits, dim=-1)
                top_probs, _ = probs_before_filter.max(dim=-1, keepdim=True)
                indices_to_remove = probs_before_filter < min_p * top_probs
                logits = logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)
            if top_p < 1.0:
                cumulative_probs = torch.cumsum(torch.nn.functional.softmax(logits, dim=-1), dim=-1)
                indices_to_remove = cumulative_probs > top_p
                indices_to_remove[..., 0] = False
                logits = logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)
            probs = torch.nn.functional.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            return top_indices.gather(1, next_token)
        if min_p > 0.0:
            probs_before_filter = torch.nn.functional.softmax(logits, dim=-1)
            top_probs, _ = probs_before_filter.max(dim=-1, keepdim=True)
            indices_to_remove = probs_before_filter < min_p * top_probs
            logits = logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 0] = False
            indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
            indices_to_remove.scatter_(1, sorted_indices, sorted_indices_to_remove)
            logits = logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    def sample_token(self, logits, temperature, top_k, top_p, min_p, repetition_penalty, token_history, generator, do_sample=True, presence_penalty=0.0):

        if not do_sample or temperature == 0.0:
            return torch.argmax(logits, dim=-1, keepdim=True)

        # Sampling mode
        if repetition_penalty != 1.0:
            for i in range(logits.shape[0]):
                for token_id in set(token_history):
                    logits[i, token_id] *= repetition_penalty if logits[i, token_id] < 0 else 1/repetition_penalty

        if presence_penalty is not None and presence_penalty != 0.0:
            for i in range(logits.shape[0]):
                for token_id in set(token_history):
                    logits[i, token_id] -= presence_penalty

        if temperature != 1.0:
            logits = logits / temperature

        if top_k > 0:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = torch.finfo(logits.dtype).min

        if min_p > 0.0:
            probs_before_filter = torch.nn.functional.softmax(logits, dim=-1)
            top_probs, _ = probs_before_filter.max(dim=-1, keepdim=True)
            min_threshold = min_p * top_probs
            indices_to_remove = probs_before_filter < min_threshold
            logits[indices_to_remove] = torch.finfo(logits.dtype).min

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 0] = False
            indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
            indices_to_remove.scatter_(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = torch.finfo(logits.dtype).min

        probs = torch.nn.functional.softmax(logits, dim=-1)

        return torch.multinomial(probs, num_samples=1, generator=generator)

class BaseQwen3:
    def logits(self, x):
        input = x[:, -1:]
        if self.model.config.lm_head:
            return self.model.lm_head(input)

        module = self.model.embed_tokens

        offload_stream = None
        if module.comfy_cast_weights:
            weight, _, offload_stream = comfy.ops.cast_bias_weight(module, input, offloadable=True)
        else:
            weight = self.model.embed_tokens.weight.to(x)

        x = torch.nn.functional.linear(input, weight, None)

        comfy.ops.uncast_bias_weight(module, weight, None, offload_stream)
        return x

class Llama2(BaseLlama, torch.nn.Module):
    def __init__(self, config_dict, dtype, device, operations):
        super().__init__()
        config = Llama2Config(**config_dict)
        self.num_layers = config.num_hidden_layers

        self.model = Llama2_(config, device=device, dtype=dtype, ops=operations)
        self.dtype = dtype

class Mistral3Small24B(BaseLlama, torch.nn.Module):
    def __init__(self, config_dict, dtype, device, operations):
        super().__init__()
        config = Mistral3Small24BConfig(**config_dict)
        self.num_layers = config.num_hidden_layers

        self.model = Llama2_(config, device=device, dtype=dtype, ops=operations)
        self.dtype = dtype

class Ministral3_3B(BaseLlama, BaseQwen3, BaseGenerate, torch.nn.Module):
    def __init__(self, config_dict, dtype, device, operations):
        super().__init__()
        config = Ministral3_3BConfig(**config_dict)
        self.num_layers = config.num_hidden_layers

        self.model = Llama2_(config, device=device, dtype=dtype, ops=operations)
        self.dtype = dtype

class Qwen25_3B(BaseLlama, torch.nn.Module):
    def __init__(self, config_dict, dtype, device, operations):
        super().__init__()
        config = Qwen25_3BConfig(**config_dict)
        self.num_layers = config.num_hidden_layers

        self.model = Llama2_(config, device=device, dtype=dtype, ops=operations)
        self.dtype = dtype

class Qwen3_06B(BaseLlama, BaseQwen3, BaseGenerate, torch.nn.Module):
    def __init__(self, config_dict, dtype, device, operations):
        super().__init__()
        config = Qwen3_06BConfig(**config_dict)
        self.num_layers = config.num_hidden_layers

        self.model = Llama2_(config, device=device, dtype=dtype, ops=operations)
        self.dtype = dtype

class Qwen3_06B_ACE15(BaseLlama, BaseQwen3, torch.nn.Module):
    def __init__(self, config_dict, dtype, device, operations):
        super().__init__()
        config = Qwen3_06B_ACE15_Config(**config_dict)
        self.num_layers = config.num_hidden_layers

        self.model = Llama2_(config, device=device, dtype=dtype, ops=operations)
        self.dtype = dtype

class Qwen3_2B_ACE15_lm(BaseLlama, BaseQwen3, torch.nn.Module):
    def __init__(self, config_dict, dtype, device, operations):
        super().__init__()
        config = Qwen3_2B_ACE15_lm_Config(**config_dict)
        self.num_layers = config.num_hidden_layers

        self.model = Llama2_(config, device=device, dtype=dtype, ops=operations)
        self.dtype = dtype

class Qwen3_4B(BaseLlama, BaseQwen3, BaseGenerate, torch.nn.Module):
    def __init__(self, config_dict, dtype, device, operations):
        super().__init__()
        config = Qwen3_4BConfig(**config_dict)
        self.num_layers = config.num_hidden_layers

        self.model = Llama2_(config, device=device, dtype=dtype, ops=operations)
        self.dtype = dtype

class Qwen3_4B_ACE15_lm(BaseLlama, BaseQwen3, torch.nn.Module):
    def __init__(self, config_dict, dtype, device, operations):
        super().__init__()
        config = Qwen3_4B_ACE15_lm_Config(**config_dict)
        self.num_layers = config.num_hidden_layers

        self.model = Llama2_(config, device=device, dtype=dtype, ops=operations)
        self.dtype = dtype

class Qwen3_8B(BaseLlama, BaseQwen3, BaseGenerate, torch.nn.Module):
    def __init__(self, config_dict, dtype, device, operations):
        super().__init__()
        config = Qwen3_8BConfig(**config_dict)
        self.num_layers = config.num_hidden_layers

        self.model = Llama2_(config, device=device, dtype=dtype, ops=operations)
        self.dtype = dtype

class Ovis25_2B(BaseLlama, torch.nn.Module):
    def __init__(self, config_dict, dtype, device, operations):
        super().__init__()
        config = Ovis25_2BConfig(**config_dict)
        self.num_layers = config.num_hidden_layers

        self.model = Llama2_(config, device=device, dtype=dtype, ops=operations)
        self.dtype = dtype

class Qwen25_7BVLI(BaseLlama, BaseGenerate, torch.nn.Module):
    def __init__(self, config_dict, dtype, device, operations):
        super().__init__()
        config = Qwen25_7BVLI_Config(**config_dict)
        self.num_layers = config.num_hidden_layers

        self.model = Llama2_(config, device=device, dtype=dtype, ops=operations)
        self.visual = qwen_vl.Qwen2VLVisionTransformer(hidden_size=1280, output_hidden_size=config.hidden_size, device=device, dtype=dtype, ops=operations)
        self.dtype = dtype

        # todo: should this be tied or not?
        #self.lm_head = operations.Linear(config.hidden_size, config.vocab_size, bias=False, device=device, dtype=dtype)

    def preprocess_embed(self, embed, device):
        if embed["type"] == "image":
            image, grid = qwen_vl.process_qwen2vl_images(embed["data"])
            return self.visual(image.to(device, dtype=torch.float32), grid), grid
        return None, None

    def forward(self, x, attention_mask=None, embeds=None, num_tokens=None, intermediate_output=None, final_layer_norm_intermediate=True, dtype=None, embeds_info=[]):
        grid = None
        position_ids = None
        offset = 0
        for e in embeds_info:
            if e.get("type") == "image":
                grid = e.get("extra", None)
                start = e.get("index")
                if position_ids is None:
                    position_ids = torch.ones((3, embeds.shape[1]), device=embeds.device, dtype=torch.long)
                    position_ids[:, :start] = torch.arange(0, start, device=embeds.device)
                end = e.get("size") + start
                len_max = int(grid.max()) // 2
                start_next = len_max + start
                if attention_mask is not None:
                    # Assign compact sequential positions to attended tokens only,
                    # skipping over padding so post-padding tokens aren't inflated.
                    after_mask = attention_mask[0, end:]
                    text_positions = after_mask.cumsum(0) - 1 + start_next + offset
                    position_ids[:, end:] = torch.where(after_mask.bool(), text_positions, position_ids[0, end:])
                else:
                    position_ids[:, end:] = torch.arange(start_next + offset, start_next + (embeds.shape[1] - end) + offset, device=embeds.device)
                position_ids[0, start:end] = start + offset
                max_d = int(grid[0][1]) // 2
                position_ids[1, start:end] = torch.arange(start + offset, start + max_d + offset, device=embeds.device).unsqueeze(1).repeat(1, math.ceil((end - start) / max_d)).flatten(0)[:end - start]
                max_d = int(grid[0][2]) // 2
                position_ids[2, start:end] = torch.arange(start + offset, start + max_d + offset, device=embeds.device).unsqueeze(0).repeat(math.ceil((end - start) / max_d), 1).flatten(0)[:end - start]
                offset += len_max - (end - start)

        if grid is None:
            position_ids = None

        return super().forward(x, attention_mask=attention_mask, embeds=embeds, num_tokens=num_tokens, intermediate_output=intermediate_output, final_layer_norm_intermediate=final_layer_norm_intermediate, dtype=dtype, position_ids=position_ids)

class Gemma2_2B(BaseLlama, BaseGenerate, torch.nn.Module):
    def __init__(self, config_dict, dtype, device, operations):
        super().__init__()
        config = Gemma2_2B_Config(**config_dict)
        self.num_layers = config.num_hidden_layers

        self.model = Llama2_(config, device=device, dtype=dtype, ops=operations)
        self.dtype = dtype

class Gemma3_4B(BaseLlama, BaseGenerate, torch.nn.Module):
    def __init__(self, config_dict, dtype, device, operations):
        super().__init__()
        config = Gemma3_4B_Config(**config_dict)
        self.num_layers = config.num_hidden_layers

        self.model = Llama2_(config, device=device, dtype=dtype, ops=operations)
        self.dtype = dtype

class Gemma3_4B_Vision(BaseLlama, BaseGenerate, torch.nn.Module):
    def __init__(self, config_dict, dtype, device, operations):
        super().__init__()
        config = Gemma3_4B_Vision_Config(**config_dict)
        self.num_layers = config.num_hidden_layers

        self.model = Llama2_(config, device=device, dtype=dtype, ops=operations)
        self.dtype = dtype
        self.multi_modal_projector = Gemma3MultiModalProjector(config, dtype, device, operations)
        self.vision_model = comfy.clip_model.CLIPVision(config.vision_config, dtype, device, operations)
        self.image_size = config.vision_config["image_size"]

    def preprocess_embed(self, embed, device):
        if embed["type"] == "image":
            image = comfy.clip_model.clip_preprocess(embed["data"], size=self.image_size, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], crop=True)
            return self.multi_modal_projector(self.vision_model(image.to(device, dtype=torch.float32))[0]), None
        return None, None

class Gemma3_12B(BaseLlama, BaseGenerate, torch.nn.Module):
    def __init__(self, config_dict, dtype, device, operations):
        super().__init__()
        config = Gemma3_12B_Config(**config_dict)
        self.num_layers = config.num_hidden_layers

        self.model = Llama2_(config, device=device, dtype=dtype, ops=operations)
        self.multi_modal_projector = Gemma3MultiModalProjector(config, dtype, device, operations)
        self.vision_model = comfy.clip_model.CLIPVision(config.vision_config, dtype, device, operations)
        self.dtype = dtype
        self.image_size = config.vision_config["image_size"]

    def preprocess_embed(self, embed, device):
        if embed["type"] == "image":
            image = comfy.clip_model.clip_preprocess(embed["data"], size=self.image_size, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], crop=True)
            return self.multi_modal_projector(self.vision_model(image.to(device, dtype=torch.float32))[0]), None
        return None, None
