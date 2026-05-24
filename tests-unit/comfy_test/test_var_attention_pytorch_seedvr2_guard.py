"""Regression tests for the SeedVR2 varlen fallback inside
``comfy.ldm.modules.attention.var_attention_pytorch``.

Contract:

  * If ``torch.nested.nested_tensor_from_jagged`` is unavailable on the
    installed PyTorch build, ``var_attention_pytorch`` must fall back to
    the codebase-local split implementation instead of raising an
    operator-facing backend error.
  * If the API is present, the present-API path must produce the
    canonical SeedVR2-inference output shape ``(total_tokens,
    heads * head_dim)``.
  * If the caller passes malformed offsets (off-end / non-monotonic /
    size-mismatched), the underlying varlen implementation error
    propagates unchanged: missing-backend fallback must not substitute a
    SeedVR2-specific error onto per-call shape errors.

Each cell additionally pins the production lookup through source
inspection so direct ``torch.nested.nested_tensor_from_jagged`` access
cannot come back silently.
"""

from comfy.cli_args import args
import torch

if not torch.cuda.is_available():
    args.cpu = True

import inspect  # noqa: E402
import logging  # noqa: E402
import warnings  # noqa: E402

import pytest  # noqa: E402

from comfy.ldm.modules.attention import (  # noqa: E402
    var_attention_pytorch,
    var_attention_pytorch_split,
)


def _inputs():
    """Canonical 2-D ``(q, k, v, heads, cu_seqlens_q, cu_seqlens_k,
    total_tokens, embed_dim)`` matching the live shape from GPT-3:
    two segments of 3 tokens each, ``embed_dim = heads * head_dim =
    2 * 8 = 16``.
    """
    heads, head_dim, total_tokens = 2, 8, 6
    embed_dim = heads * head_dim
    q = torch.randn(total_tokens, embed_dim)
    k = torch.randn(total_tokens, embed_dim)
    v = torch.randn(total_tokens, embed_dim)
    cu = torch.tensor([0, 3, 6], dtype=torch.int32)
    return q, k, v, heads, cu, cu, total_tokens, embed_dim


def _assert_no_direct_nested_lookup():
    src = inspect.getsource(var_attention_pytorch)
    assert "torch.nested.nested_tensor_from_jagged" not in src


def test_missing_api_uses_split_fallback(monkeypatch):
    monkeypatch.delattr(torch.nested, "nested_tensor_from_jagged", raising=False)
    q, k, v, heads, cu_q, cu_k, _, _ = _inputs()

    out = var_attention_pytorch(q, k, v, heads, cu_q, cu_k)
    expected = var_attention_pytorch_split(q, k, v, heads, cu_q, cu_k)

    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    _assert_no_direct_nested_lookup()


def test_missing_namespace_uses_split_fallback(monkeypatch):
    monkeypatch.delattr(torch, "nested", raising=False)
    q, k, v, heads, cu_q, cu_k, _, _ = _inputs()

    out = var_attention_pytorch(q, k, v, heads, cu_q, cu_k)
    expected = var_attention_pytorch_split(q, k, v, heads, cu_q, cu_k)

    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    _assert_no_direct_nested_lookup()


def test_present_api_returns_expected_shape():
    q, k, v, heads, cu_q, cu_k, total_tokens, embed_dim = _inputs()

    torch_fx_logger = logging.getLogger("torch.fx._symbolic_trace")
    old_torch_fx_level = torch_fx_logger.level
    torch_fx_logger.setLevel(logging.ERROR)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="The PyTorch API of nested tensors is in prototype stage.*",
                category=UserWarning,
            )
            out = var_attention_pytorch(q, k, v, heads, cu_q, cu_k)
    finally:
        torch_fx_logger.setLevel(old_torch_fx_level)

    assert tuple(out.shape) == (total_tokens, embed_dim), (
        f"expected ({total_tokens}, {embed_dim}); got {tuple(out.shape)}"
    )

    _assert_no_direct_nested_lookup()


def test_malformed_offsets_propagates_torch_runtime_error():
    q, k, v, heads, _, _, _, _ = _inputs()
    cu_q_bad = torch.tensor([0, 3, 7], dtype=torch.int32)
    cu_k_ok = torch.tensor([0, 3, 6], dtype=torch.int32)

    with pytest.raises((RuntimeError, ValueError)) as exc_info:
        var_attention_pytorch(q, k, v, heads, cu_q_bad, cu_k_ok)

    msg = str(exc_info.value)
    assert "SeedVR2" not in msg, (
        f"SeedVR2-context substring must not be substituted onto torch's "
        f"per-call shape error; got: {msg!r}"
    )

    _assert_no_direct_nested_lookup()
