import inspect
import json

import pytest
import torch

from comfy.ldm.flux.math import apply_rope1
from comfy.ldm.seedvr.model import apply_rotary_emb


@pytest.fixture(scope="session", autouse=True)
def _print_apply_rotary_emb_params(pytestconfig):
    names = list(inspect.signature(apply_rotary_emb).parameters)
    line = f"params: {json.dumps(names)}"
    tr = pytestconfig.pluginmanager.get_plugin("terminalreporter")
    if tr is not None:
        tr.write_line(line)
    else:
        print(line)
    yield


_TOL = {
    torch.float32: 1e-6,
    torch.float16: 1e-3,
    torch.bfloat16: 1e-2,
}


_CASES = [
    pytest.param("cpu", torch.float32, (1, 8, 16), id="cpu-float32-1x8x16"),
    pytest.param("cpu", torch.float16, (1, 8, 16), id="cpu-float16-1x8x16"),
    pytest.param("cpu", torch.bfloat16, (1, 8, 16), id="cpu-bfloat16-1x8x16"),
    pytest.param("cpu", torch.float32, (2, 16, 32), id="cpu-float32-2x16x32"),
    pytest.param(
        "cuda",
        torch.float16,
        (1, 8, 16),
        id="cuda-float16-1x8x16",
        marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="no cuda"),
    ),
]


@pytest.mark.parametrize("device,dtype,shape", _CASES)
def test_apply_rotary_emb_delegates_to_apply_rope1(device, dtype, shape):
    torch.manual_seed(0)
    t = torch.randn(*shape, dtype=dtype, device=device)
    freqs = torch.randn(shape[-2], shape[-1], dtype=dtype, device=device)

    wrapper_out = apply_rotary_emb(freqs, t)

    rot_feats = freqs.shape[-1]
    t_middle = t[..., 0:rot_feats]
    angles = freqs.to(t_middle.device)[..., ::2]
    cos = torch.cos(angles) * 1.0
    sin = torch.sin(angles) * 1.0
    col0 = torch.stack([cos, sin], dim=-1)
    col1 = torch.stack([-sin, cos], dim=-1)
    freqs_mat = torch.stack([col0, col1], dim=-1)
    direct_out = apply_rope1(t_middle, freqs_mat)

    tol = _TOL[dtype]
    assert torch.allclose(wrapper_out, direct_out, atol=tol)
