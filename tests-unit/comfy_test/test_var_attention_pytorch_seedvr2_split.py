"""Regression tests for the current SeedVR2 PyTorch var-attention path.

``var_attention_pytorch`` intentionally delegates to the split SDPA
implementation so the path has no runtime dependency on PyTorch's prototype
``torch.nested`` jagged tensor API.
"""

from comfy.cli_args import args
import torch
import torch.nn.functional as F

if not torch.cuda.is_available():
    args.cpu = True

import comfy.ldm.modules.attention as attention  # noqa: E402


def _inputs():
    heads, head_dim, total_tokens = 2, 8, 6
    embed_dim = heads * head_dim
    generator = torch.Generator().manual_seed(0)
    q = torch.randn(total_tokens, embed_dim, generator=generator)
    k = torch.randn(total_tokens, embed_dim, generator=generator)
    v = torch.randn(total_tokens, embed_dim, generator=generator)
    cu = torch.tensor([0, 3, 6], dtype=torch.int32)
    return q, k, v, heads, cu, cu


def _reference(q, k, v, heads, cu_q, cu_k):
    q = q.view(q.shape[0], heads, -1)
    k = k.view(k.shape[0], heads, -1)
    v = v.view(v.shape[0], heads, -1)
    out = []
    for i in range(cu_q.numel() - 1):
        qs = slice(cu_q[i].item(), cu_q[i + 1].item())
        ks = slice(cu_k[i].item(), cu_k[i + 1].item())
        q_i = q[qs].permute(1, 0, 2).unsqueeze(0)
        k_i = k[ks].permute(1, 0, 2).unsqueeze(0)
        v_i = v[ks].permute(1, 0, 2).unsqueeze(0)
        out_i = F.scaled_dot_product_attention(q_i, k_i, v_i, dropout_p=0.0)
        out.append(out_i.squeeze(0).permute(1, 0, 2))
    return torch.cat(out, dim=0).reshape(q.shape[0], -1)


def test_missing_nested_api_still_runs_split_sdpa(monkeypatch):
    monkeypatch.delattr(torch.nested, "nested_tensor_from_jagged", raising=False)
    q, k, v, heads, cu_q, cu_k = _inputs()

    out = attention.var_attention_pytorch(q, k, v, heads, cu_q, cu_k)

    torch.testing.assert_close(out, _reference(q, k, v, heads, cu_q, cu_k))


def test_missing_nested_namespace_still_runs_split_sdpa(monkeypatch):
    monkeypatch.delattr(torch, "nested", raising=False)
    q, k, v, heads, cu_q, cu_k = _inputs()

    out = attention.var_attention_pytorch(q, k, v, heads, cu_q, cu_k)

    torch.testing.assert_close(out, _reference(q, k, v, heads, cu_q, cu_k))


def test_var_attention_pytorch_normalizes_offsets_before_split(monkeypatch):
    captured = {}

    class _Offsets:
        def __init__(self, name):
            self.name = name

        def cpu(self):
            return f"{self.name}_cpu"

    def fake_split(q, k, v, heads, cu_seqlens_q, cu_seqlens_k, skip_reshape=False, skip_output_reshape=False):
        captured.update(
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            skip_reshape=skip_reshape,
            skip_output_reshape=skip_output_reshape,
        )
        return "ok"

    monkeypatch.setattr(attention, "var_attention_pytorch_split", fake_split)

    out = attention.var_attention_pytorch(
        "q",
        "k",
        "v",
        2,
        _Offsets("q"),
        _Offsets("k"),
        skip_reshape=True,
        skip_output_reshape=True,
    )

    assert out == "ok"
    assert captured == {
        "cu_seqlens_q": "q_cpu",
        "cu_seqlens_k": "k_cpu",
        "skip_reshape": True,
        "skip_output_reshape": True,
    }


def test_malformed_offsets_fail_loudly_from_split_validator():
    q, k, v, heads, _, cu_k = _inputs()
    cu_q_bad = torch.tensor([0, 3, 7], dtype=torch.int32)

    try:
        attention.var_attention_pytorch(q, k, v, heads, cu_q_bad, cu_k)
    except ValueError as exc:
        msg = str(exc)
    else:
        raise AssertionError("malformed offsets were accepted")

    assert "cu_seqlens_q does not match token count" in msg
    assert "nested_tensor_from_jagged" not in msg
