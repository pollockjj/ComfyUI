"""Regression: ``comfy.ldm.seedvr.vae.VideoAutoencoderKL.forward`` must
honor the actual tensor return contract of ``encode()`` and
``decode_()`` and must NOT dereference diffusers-style ``.latent_dist``
or ``.sample`` attributes on those returns.

Source: CodeRabbit review on Comfy-Org/ComfyUI#11294 thread
https://github.com/Comfy-Org/ComfyUI/pull/11294#discussion_r2959796348 .
Tracker: pollockjj/mydevelopment#190.

The pre-fix body raised ``AttributeError: 'Tensor' object has no
attribute 'latent_dist'`` for ``mode in {"encode", "all"}`` and
``AttributeError: 'VideoAutoencoderKL' object has no attribute 'decode'``
for ``mode == "decode"`` (the class only defines ``decode_`` with a
trailing underscore). The post-fix body returns ``encode()`` /
``decode_()`` outputs directly. ``forward`` preserves the pre-fix
``**kwargs`` call surface (Copilot 3188197480 / 3188197524: scope the
fix to the actual return-contract bug, not signature narrowing) and
extracts ``return_dict`` explicitly so ``return_dict=False`` is
propagated end-to-end as a single-element tuple — no silent discarding
of the tuple wrapper. Other kwargs are absorbed at the ``forward``
boundary and NOT propagated into ``encode()`` / ``decode_()`` (Copilot
3188128381 cycle 7 — leakage into helpers raises ``TypeError`` from
inside them rather than at the entry point). An unsupported ``mode``
value raises ``ValueError`` instead of silently running the ``"all"``
path.

Tests construct a stub subclass of ``VideoAutoencoderKL`` that bypasses
the heavy ``__init__`` via ``torch.nn.Module.__init__(self)`` and
overrides ``encode``/``decode_`` with sentinel-valued tensors so the
contract can be probed without loading any real VAE weights.
``decode_`` records its input so ``mode="all"`` can pin the
encode->decode composition.
"""

import ast
import inspect
import textwrap

import torch
import torch.nn as nn

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

from comfy.ldm.seedvr.vae import VideoAutoencoderKL  # noqa: E402


_LATENT_SHAPE = (1, 16, 2, 2, 2)
_DECODED_SHAPE = (1, 3, 5, 16, 16)
_INPUT_ENCODE_SHAPE = (1, 3, 5, 16, 16)
_INPUT_DECODE_SHAPE = (1, 16, 2, 2, 2)

# Sentinel scalar values let tests verify that ``decode_`` received the
# encode output and not the original ``x`` (or any other unintended tensor).
_ENCODE_SENTINEL = 0.4242
_DECODE_SENTINEL = 0.7373


class _StubVAE(VideoAutoencoderKL):
    def __init__(self):
        nn.Module.__init__(self)
        self._encode_out = torch.full(_LATENT_SHAPE, _ENCODE_SENTINEL)
        self._decode_out = torch.full(_DECODED_SHAPE, _DECODE_SENTINEL)
        self.encode_calls = []
        self.decode_calls = []

    def encode(self, x, return_dict=True):
        self.encode_calls.append({"x": x, "return_dict": return_dict})
        if not return_dict:
            return (self._encode_out,)
        return self._encode_out

    def decode_(self, z, return_dict=True):
        self.decode_calls.append({"z": z, "return_dict": return_dict})
        if not return_dict:
            return (self._decode_out,)
        return self._decode_out


def test_forward_encode_returns_tensor():
    vae = _StubVAE()
    x = torch.zeros(*_INPUT_ENCODE_SHAPE)
    result = vae.forward(x, mode="encode")
    assert type(result) is torch.Tensor
    assert result.shape == torch.Size(_LATENT_SHAPE)
    assert torch.equal(result, vae._encode_out)
    assert len(vae.encode_calls) == 1
    assert vae.encode_calls[0]["return_dict"] is True
    assert len(vae.decode_calls) == 0


def test_forward_decode_returns_tensor():
    vae = _StubVAE()
    z = torch.zeros(*_INPUT_DECODE_SHAPE)
    result = vae.forward(z, mode="decode")
    assert type(result) is torch.Tensor
    assert result.shape == torch.Size(_DECODED_SHAPE)
    assert torch.equal(result, vae._decode_out)
    assert len(vae.encode_calls) == 0
    assert len(vae.decode_calls) == 1
    assert vae.decode_calls[0]["return_dict"] is True


def test_forward_all_returns_tensor():
    vae = _StubVAE()
    x = torch.zeros(*_INPUT_ENCODE_SHAPE)
    result = vae.forward(x, mode="all")
    assert type(result) is torch.Tensor
    assert result.shape == torch.Size(_DECODED_SHAPE)
    assert torch.equal(result, vae._decode_out)


def test_forward_all_pins_encode_then_decode_composition():
    """``mode='all'`` must call ``encode(x)`` and then feed that output
    into ``decode_``. A regression that decoded the original ``x`` (or any
    other tensor) instead of the encode output must fail this test.
    """
    vae = _StubVAE()
    x = torch.zeros(*_INPUT_ENCODE_SHAPE)
    vae.forward(x, mode="all")
    assert len(vae.encode_calls) == 1
    assert len(vae.decode_calls) == 1
    assert torch.equal(vae.encode_calls[0]["x"], x)
    decode_input = vae.decode_calls[0]["z"]
    assert decode_input.shape == torch.Size(_LATENT_SHAPE)
    assert torch.equal(decode_input, vae._encode_out)


def test_forward_encode_propagates_return_dict_false():
    """``forward(x, mode="encode", return_dict=False)`` must reach
    ``encode()`` with ``return_dict=False`` and must return the
    single-element tuple ``(tensor,)`` exactly as ``encode()`` itself
    does. ``forward`` extracts ``return_dict`` from its ``**kwargs``
    surface and propagates only that (Copilot 3188128381 cycle 7:
    propagating arbitrary kwargs leaks into helpers as ``TypeError``;
    Copilot 3188197480 cycle 8: removing the ``**kwargs`` surface is a
    breaking API change). This test pins the positive propagation
    contract for the supported kwarg.
    """
    vae = _StubVAE()
    x = torch.zeros(*_INPUT_ENCODE_SHAPE)
    result = vae.forward(x, mode="encode", return_dict=False)
    assert type(result) is tuple
    assert len(result) == 1
    assert type(result[0]) is torch.Tensor
    assert torch.equal(result[0], vae._encode_out)
    assert len(vae.encode_calls) == 1
    assert vae.encode_calls[0]["return_dict"] is False
    assert len(vae.decode_calls) == 0


def test_forward_decode_propagates_return_dict_false():
    """``forward(z, mode="decode", return_dict=False)`` must reach
    ``decode_()`` with ``return_dict=False`` and return the
    single-element tuple ``decode_()`` produces.
    """
    vae = _StubVAE()
    z = torch.zeros(*_INPUT_DECODE_SHAPE)
    result = vae.forward(z, mode="decode", return_dict=False)
    assert type(result) is tuple
    assert len(result) == 1
    assert type(result[0]) is torch.Tensor
    assert torch.equal(result[0], vae._decode_out)
    assert len(vae.encode_calls) == 0
    assert len(vae.decode_calls) == 1
    assert vae.decode_calls[0]["return_dict"] is False


def test_forward_all_propagates_return_dict_false_to_terminal_decode_only():
    """For ``mode="all"`` the internal ``encode`` MUST run with the
    default ``return_dict=True`` so the latent it returns is a tensor
    (the only shape ``decode_`` accepts as input). The user's
    ``return_dict=False`` must reach only the terminal ``decode_`` call,
    which then yields the single-element tuple. This asymmetry is
    deliberate — it keeps the ``encode→decode_`` composition working
    while still honoring the user's requested final return shape.
    """
    vae = _StubVAE()
    x = torch.zeros(*_INPUT_ENCODE_SHAPE)
    result = vae.forward(x, mode="all", return_dict=False)
    assert type(result) is tuple
    assert len(result) == 1
    assert type(result[0]) is torch.Tensor
    assert torch.equal(result[0], vae._decode_out)
    assert len(vae.encode_calls) == 1
    assert vae.encode_calls[0]["return_dict"] is True
    assert len(vae.decode_calls) == 1
    assert vae.decode_calls[0]["return_dict"] is False
    decode_input = vae.decode_calls[0]["z"]
    assert torch.equal(decode_input, vae._encode_out)


def test_forward_source_has_no_diffusers_attr_access():
    """AST-based pin: walk ``VideoAutoencoderKL.forward``'s body and
    assert no ``Attribute`` node accesses ``.latent_dist`` or ``.sample``
    on any expression, and no ``Call`` invokes ``self.decode(...)`` (the
    method is named ``decode_`` with a trailing underscore on this
    class). This pins Copilot inline comment 3187882551: source-text
    matching is brittle because a future docstring or explanatory
    comment containing those tokens would false-fail the test, while
    AST traversal sees only real attribute access and call nodes.
    """
    src = textwrap.dedent(inspect.getsource(VideoAutoencoderKL.forward))
    tree = ast.parse(src)

    bad_attrs = []
    bad_self_decode_calls = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            if node.attr in ("latent_dist", "sample"):
                bad_attrs.append((node.lineno, node.attr))
        if isinstance(node, ast.Call):
            f = node.func
            if (
                isinstance(f, ast.Attribute)
                and f.attr == "decode"
                and isinstance(f.value, ast.Name)
                and f.value.id == "self"
            ):
                bad_self_decode_calls.append(node.lineno)

    assert bad_attrs == [], (
        f"diffusers-style attribute access detected in forward(): {bad_attrs}"
    )
    assert bad_self_decode_calls == [], (
        f"self.decode(...) call detected in forward() at lines "
        f"{bad_self_decode_calls}; this class only defines decode_ "
        f"(trailing underscore)."
    )


def test_forward_unknown_kwarg_absorbed_at_forward_boundary():
    """Pins the joint contract from Copilot 3188128381 (cycle 7) and
    Copilot 3188197480 / 3188197524 (cycle 8): ``forward`` exposes a
    ``**kwargs`` surface so it remains call-compatible with the pre-fix
    signature, AND unsupported kwargs are absorbed at the forward
    boundary instead of being propagated into ``encode()`` / ``decode_()``
    where they would surface as ``TypeError`` from inside the helpers.
    The forward call must succeed; the helper call list shows no
    ``unsupported_kwarg`` was leaked.
    """
    vae = _StubVAE()
    x = torch.zeros(*_INPUT_ENCODE_SHAPE)
    result = vae.forward(x, mode="encode", unsupported_kwarg=True)
    assert type(result) is torch.Tensor
    assert torch.equal(result, vae._encode_out)
    assert len(vae.encode_calls) == 1
    encode_call = vae.encode_calls[0]
    assert "unsupported_kwarg" not in encode_call, (
        f"unsupported_kwarg must not be propagated into encode(); "
        f"recorded call kwargs: {sorted(encode_call.keys())}"
    )
    assert encode_call["return_dict"] is True
    assert len(vae.decode_calls) == 0


def test_forward_unsupported_mode_raises_value_error():
    """Pins Copilot inline comment 3188128442: ``Literal[...]`` is only
    a type hint; an unsupported ``mode`` value at runtime must fail
    fast with ``ValueError`` instead of silently running the ``"all"``
    encode/decode round-trip.
    """
    vae = _StubVAE()
    x = torch.zeros(*_INPUT_ENCODE_SHAPE)
    raised = None
    try:
        vae.forward(x, mode="bogus_mode")
    except ValueError as exc:
        raised = exc
    assert raised is not None, (
        "forward(x, mode='bogus_mode') must raise ValueError; "
        "the else branch must not silently run the encode/decode path."
    )
    assert "bogus_mode" in str(raised), (
        f"ValueError must echo the rejected mode; got: {raised!r}"
    )
    assert len(vae.encode_calls) == 0
    assert len(vae.decode_calls) == 0
