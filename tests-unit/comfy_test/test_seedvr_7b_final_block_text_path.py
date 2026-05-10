from __future__ import annotations

import torch
from torch import nn

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

import comfy.ldm.seedvr.model as seedvr_model  # noqa: E402


class _StubModule(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()


def _capture_last_layer_flags(monkeypatch, vid_dim: int, txt_in_dim: int) -> list[bool]:
    flags = []

    class _Block(_StubModule):
        def __init__(self, *args, **kwargs):
            flags.append(kwargs["is_last_layer"])
            super().__init__()

    monkeypatch.setattr(seedvr_model, "NaPatchIn", _StubModule)
    monkeypatch.setattr(seedvr_model, "NaPatchOut", _StubModule)
    monkeypatch.setattr(seedvr_model, "TimeEmbedding", _StubModule)
    monkeypatch.setattr(seedvr_model, "NaMMSRTransformerBlock", _Block)

    seedvr_model.NaDiT(
        norm_eps=1e-5,
        qk_rope=None,
        num_layers=4,
        mlp_type="normal",
        vid_dim=vid_dim,
        txt_in_dim=txt_in_dim,
        heads=24,
        mm_layers=3,
    )

    return flags


def test_seedvr2_7b_keeps_final_block_text_path(monkeypatch):
    assert _capture_last_layer_flags(monkeypatch, vid_dim=3072, txt_in_dim=3072) == [
        False,
        False,
        False,
        False,
    ]


def test_seedvr2_3b_keeps_final_block_vid_only_path(monkeypatch):
    assert _capture_last_layer_flags(monkeypatch, vid_dim=2560, txt_in_dim=2560) == [
        False,
        False,
        False,
        True,
    ]


def test_seedvr2_7b_rope3d_matches_checkpoint_buffer_shape():
    rope = seedvr_model.get_na_rope("rope3d", dim=64)

    assert isinstance(rope, seedvr_model.NaRotaryEmbedding3d)
    assert tuple(rope.rope.freqs.shape) == (10,)


def test_seedvr2_7b_rope3d_preserves_qk_shape():
    rope = seedvr_model.get_na_rope("rope3d", dim=64)
    q = torch.randn(4, 2, 128)
    k = torch.randn(4, 2, 128)
    shape = torch.tensor([[1, 2, 2]], dtype=torch.long)

    q_out, k_out = rope(q, k, shape, seedvr_model.Cache(disable=True))

    assert q_out.shape == q.shape
    assert k_out.shape == k.shape


def test_seedvr2_7b_rope3d_matches_wrapper_oracle():
    rope = seedvr_model.get_na_rope("rope3d", dim=64)
    generator = torch.Generator(device="cpu").manual_seed(0)
    q = torch.randn(4, 2, 128, generator=generator)
    k = torch.randn(4, 2, 128, generator=generator)
    shape = torch.tensor([[1, 2, 2]], dtype=torch.long)
    freqs = rope.get_axial_freqs(1, 2, 2).reshape(4, -1)

    expected_q = seedvr_model.apply_rotary_emb(
        freqs,
        q.permute(1, 0, 2).float(),
    ).to(q.dtype).permute(1, 0, 2)
    expected_k = seedvr_model.apply_rotary_emb(
        freqs,
        k.permute(1, 0, 2).float(),
    ).to(k.dtype).permute(1, 0, 2)

    actual_q, actual_k = rope(q.clone(), k.clone(), shape, seedvr_model.Cache(disable=True))

    torch.testing.assert_close(actual_q, expected_q, rtol=0, atol=0)
    torch.testing.assert_close(actual_k, expected_k, rtol=0, atol=0)
