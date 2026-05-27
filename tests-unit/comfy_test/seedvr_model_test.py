"""Regression tests for SeedVR2 conditioning split hardening."""

from comfy.cli_args import args
import torch

if not torch.cuda.is_available():
    args.cpu = True

from comfy.ldm.seedvr.model import NaDiT  # noqa: E402


def _make_standin(positive_conditioning):
    class _StandIn(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer(
                "positive_conditioning", positive_conditioning
            )

        _resolve_text_conditioning = NaDiT._resolve_text_conditioning
        _swap_pos_neg_halves = NaDiT._swap_pos_neg_halves

    return _StandIn()


def test_missing_context_falls_back_to_positive_buffer():
    """AC: ``context is None`` falls back to the registered
    ``positive_conditioning`` buffer and runs to completion — no
    silent zero substitution, no raised exception.
    """
    pos_buffer = torch.full((58, 5120), 7.0)
    standin = _make_standin(pos_buffer)
    txt, txt_shape = standin._resolve_text_conditioning(None)
    assert txt.shape == (58, 5120)
    assert (txt == 7.0).all(), (
        "fallback path must use the positive_conditioning buffer "
        "verbatim, not a zero tensor"
    )
    assert txt_shape.shape == (1, 1)
    assert txt_shape[0, 0].item() == 58


def test_output_side_swaps_pos_neg_halves():
    """AC complement: ``_swap_pos_neg_halves`` reorders the post-network
    output so the first half (positive) and second half (negative) trade
    places. For a 2-batch tensor with distinguishable halves, the
    returned tensor must be the swap — first half becomes negative,
    second half becomes positive — matching the original
    ``torch.cat([neg, pos])`` semantics from the pre-fix forward path.
    """
    pos_buffer = torch.zeros((58, 5120))
    standin = _make_standin(pos_buffer)
    pos_half = torch.full((1, 4, 8, 8), 1.0)
    neg_half = torch.full((1, 4, 8, 8), -1.0)
    out = torch.cat([pos_half, neg_half], dim=0)
    swapped = standin._swap_pos_neg_halves(out)
    assert swapped.shape == out.shape
    assert (swapped[0] == -1.0).all(), "first half of swapped output must be the original negative half"
    assert (swapped[1] == 1.0).all(), "second half of swapped output must be the original positive half"
