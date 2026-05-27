import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

import comfy.ldm.modules.attention as attention  # noqa: E402


def _inputs():
    heads = 2
    head_dim = 4
    total = 6
    q = torch.randn(total, heads, head_dim)
    k = torch.randn(total, heads, head_dim)
    v = torch.randn(total, heads, head_dim)
    cu = torch.tensor([0, 3, 6], dtype=torch.int32)
    return q, k, v, heads, cu


def test_var_attention_registry_contains_always_available_entries():
    assert attention.REGISTERED_ATTENTION_FUNCTIONS["var_attention_pytorch"] is attention.var_attention_pytorch
    assert attention.REGISTERED_ATTENTION_FUNCTIONS["var_attention_sub_quad"] is attention.var_attention_sub_quad
    assert attention.REGISTERED_ATTENTION_FUNCTIONS["var_attention_split"] is attention.var_attention_split


def test_var_attention_sage_uses_cu_seqlens_contract(monkeypatch):
    q, k, v, heads, cu = _inputs()
    captured = {}

    def fake_sageattn_varlen(q, k, v, cu_q, cu_k, max_q, max_k, is_causal, sm_scale):
        captured.update(cu_q=cu_q, cu_k=cu_k, max_q=max_q, max_k=max_k, is_causal=is_causal)
        return torch.zeros_like(q)

    monkeypatch.setattr(attention, "SAGE_ATTENTION_VARLEN_IS_AVAILABLE", True)
    monkeypatch.setattr(attention, "sageattn_varlen", fake_sageattn_varlen, raising=False)

    out = attention.var_attention_sage(q, k, v, heads, cu, cu, skip_reshape=True, skip_output_reshape=True)

    assert tuple(out.shape) == tuple(q.shape)
    assert torch.equal(captured["cu_q"], cu)
    assert torch.equal(captured["cu_k"], cu)
    assert captured["max_q"] == 3
    assert captured["max_k"] == 3
    assert captured["is_causal"] is False


def test_var_attention_pytorch_split_normalizes_split_indices_to_cpu(monkeypatch):
    q, k, v, heads, cu = _inputs()
    captured_devices = []
    real_tensor_split = torch.tensor_split

    def capture_tensor_split(input, indices_or_sections, dim=0):
        if isinstance(indices_or_sections, torch.Tensor):
            captured_devices.append(indices_or_sections.device.type)
        return real_tensor_split(input, indices_or_sections, dim=dim)

    monkeypatch.setattr(torch, "tensor_split", capture_tensor_split)

    out = attention.var_attention_pytorch_split(q, k, v, heads, cu, cu, skip_reshape=True, skip_output_reshape=True)

    assert tuple(out.shape) == tuple(q.shape)
    assert captured_devices == ["cpu", "cpu", "cpu"]
