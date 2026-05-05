"""Regression tests for the SeedVR2-named guard inside
``comfy.ldm.modules.attention.var_attention_pytorch``.

Contract:

  * If ``torch.nested.nested_tensor_from_jagged`` is unavailable on the
    installed PyTorch build, ``var_attention_pytorch`` must raise
    ``RuntimeError`` whose message contains both ``SeedVR2`` and
    ``nested_tensor_from_jagged`` so the operator can identify the
    failing attention path. A bare ``AttributeError`` from the
    ``torch.nested`` lookup is non-conformant.
  * If the API is present, the present-API path must produce the
    canonical SeedVR2-inference output shape ``(total_tokens,
    heads * head_dim)``.
  * If the caller passes malformed offsets (off-end / non-monotonic /
    size-mismatched), torch's own per-call ``RuntimeError`` propagates
    unchanged: the SeedVR2-context guard fires only on the missing-API
    path, never on torch's per-call shape errors.

Each cell additionally pins the production guard at the source level
via ``inspect.getsource(var_attention_pytorch)`` so every AC fails
diagnostically on an unguarded base.
"""

from comfy.cli_args import args
import torch

if not torch.cuda.is_available():
    args.cpu = True

import inspect  # noqa: E402

import pytest  # noqa: E402

from comfy.ldm.modules.attention import var_attention_pytorch  # noqa: E402


def _inputs():
    """Canonical 2-D ``(q, k, v, heads, cu_seqlens_q, cu_seqlens_k,
    total_tokens, embed_dim)`` matching the live shape from GTP-3:
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


def _assert_guard_source_pin():
    src = inspect.getsource(var_attention_pytorch)
    assert "raise RuntimeError(" in src, (
        "var_attention_pytorch source has no `raise RuntimeError(` substring; "
        "the SeedVR2-named guard is missing.\n"
        f"--- source ---\n{src}"
    )
    raise_idx = src.index("raise RuntimeError(")
    call_idx = src.index("nested_tensor_from_jagged")
    assert raise_idx < call_idx, (
        "`raise RuntimeError(` appears at index "
        f"{raise_idx} but the first `nested_tensor_from_jagged` substring is "
        f"at index {call_idx}; the guard must precede the unguarded lookup.\n"
        f"--- source ---\n{src}"
    )


def test_missing_api_raises_seedvr2_runtime_error(monkeypatch):
    monkeypatch.delattr(torch.nested, "nested_tensor_from_jagged")
    q, k, v, heads, cu_q, cu_k, _, _ = _inputs()

    with pytest.raises(RuntimeError, match=r"SeedVR2.*nested_tensor_from_jagged"):
        var_attention_pytorch(q, k, v, heads, cu_q, cu_k)

    _assert_guard_source_pin()


def test_present_api_returns_expected_shape():
    q, k, v, heads, cu_q, cu_k, total_tokens, embed_dim = _inputs()

    out = var_attention_pytorch(q, k, v, heads, cu_q, cu_k)

    assert tuple(out.shape) == (total_tokens, embed_dim), (
        f"expected ({total_tokens}, {embed_dim}); got {tuple(out.shape)}"
    )

    _assert_guard_source_pin()


def test_malformed_offsets_propagates_torch_runtime_error():
    q, k, v, heads, _, _, _, _ = _inputs()
    cu_q_bad = torch.tensor([0, 3, 7], dtype=torch.int32)
    cu_k_ok = torch.tensor([0, 3, 6], dtype=torch.int32)

    with pytest.raises(RuntimeError) as exc_info:
        var_attention_pytorch(q, k, v, heads, cu_q_bad, cu_k_ok)

    msg = str(exc_info.value)
    assert "split_with_sizes" in msg, (
        f"expected torch's `split_with_sizes` error to propagate; got: {msg!r}"
    )
    assert "SeedVR2" not in msg, (
        f"SeedVR2-context substring must not be substituted onto torch's "
        f"per-call shape error; got: {msg!r}"
    )

    _assert_guard_source_pin()
