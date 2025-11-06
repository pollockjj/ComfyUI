"""Raylight's exact single block forward method."""

import torch
from torch import Tensor

from comfy.ldm.flux.layers import apply_mod
from comfy.parallel_attention.raylight_attention import attention


def raylight_single_stream_forward(
    self,
    x: Tensor,
    vec: Tensor,
    pe: Tensor,
    attn_mask=None,
    modulation_dims=None,
    **kwargs,
) -> Tensor:
    """Raylight's exact single block forward.

    This is a direct copy of Raylight's usp_single_stream_forward().
    """
    mod, _ = self.modulation(vec)

    qkv, mlp = torch.split(
        self.linear1(
            apply_mod(
                self.pre_norm(x),
                (1 + mod.scale),
                mod.shift,
                modulation_dims,
            )
        ),
        [3 * self.hidden_size, self.mlp_hidden_dim],
        dim=-1,
    )

    q, k, v = qkv.view(
        qkv.shape[0],
        qkv.shape[1],
        3,
        self.num_heads,
        -1,
    ).permute(2, 0, 3, 1, 4)

    q, k = self.norm(q, k, v)

    attn = attention(q, k, v, pe=pe, mask=attn_mask)

    output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
    x += apply_mod(output, mod.gate, None, modulation_dims)

    if x.dtype == torch.float16:
        x = torch.nan_to_num(x, nan=0.0, posinf=65504, neginf=-65504)

    return x
