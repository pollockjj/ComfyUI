import pytest
import torch

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

from comfy.ldm.modules.diffusionmodules.model import get_timestep_embedding  # noqa: E402


def test_get_timestep_embedding_rejects_invalid_downscale_freq_shift():
    with pytest.raises(ValueError, match="downscale_freq_shift"):
        get_timestep_embedding(torch.tensor([1.0]), embedding_dim=2, downscale_freq_shift=1)


def test_get_timestep_embedding_accepts_valid_downscale_freq_shift():
    out = get_timestep_embedding(torch.tensor([1.0]), embedding_dim=4, downscale_freq_shift=1)

    assert tuple(out.shape) == (1, 4)
