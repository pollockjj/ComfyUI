from unittest.mock import patch

import torch

import comfy.ldm.seedvr.vae as seedvr_vae


def _identity_linear(linear):
    with torch.no_grad():
        linear.weight.copy_(torch.eye(linear.weight.shape[0], linear.weight.shape[1]))
        if linear.bias is not None:
            linear.bias.zero_()


def test_seedvr_vae_4d_self_attention_uses_vae_attention_with_channel_first_layout():
    calls = {}

    def vae_attention_spy(q, k, v):
        calls["q"] = q.detach().clone()
        calls["k"] = k.detach().clone()
        calls["v"] = v.detach().clone()
        return q

    def global_attention_forbidden(*args, **kwargs):
        raise AssertionError("SeedVR2 VAE self-attention must not use global optimized_attention")

    with patch.object(seedvr_vae, "vae_attention", return_value=vae_attention_spy):
        attention = seedvr_vae.Attention(query_dim=4, heads=1, dim_head=4)

    _identity_linear(attention.to_q)
    _identity_linear(attention.to_k)
    _identity_linear(attention.to_v)
    _identity_linear(attention.to_out[0])

    hidden_states = torch.arange(24, dtype=torch.float32).reshape(1, 4, 2, 3)

    with patch.object(seedvr_vae, "optimized_attention", global_attention_forbidden):
        output = attention(hidden_states)

    assert torch.equal(calls["q"], hidden_states)
    assert torch.equal(calls["k"], hidden_states)
    assert torch.equal(calls["v"], hidden_states)
    assert torch.equal(output, hidden_states)
