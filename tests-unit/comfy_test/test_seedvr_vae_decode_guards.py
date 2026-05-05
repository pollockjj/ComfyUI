"""Regression tests for SeedVR2 ``VideoAutoencoderKLWrapper.decode`` input guards.

Tracks pollockjj/mydevelopment#194 (parent #101). CodeRabbit thread
``r2962219551`` on upstream PR Comfy-Org/ComfyUI#11294 flagged that
``decode()`` accessed ``self.original_image_video`` and ``self.img_dims``
without validation, raising opaque torch errors deep in ``rearrange()`` /
attribute unpacking when a workflow wired the wrapper directly without
``SeedVR2InputProcessing.execute`` populating those attributes. This PR
adds two presence guards plus shape/arity validation at the top of
``decode()``, raising ``RuntimeError`` with a SeedVR2-specific message
identifying the missing or malformed attribute, and initialises
``self.img_dims = None`` in ``__init__`` so the missing-state branch is
reachable from a default-constructed instance.

PR review round 8 (Copilot comment_id=3188027986) extended the same
defect class to ``self.tiled_args``: ``decode()`` unconditionally calls
``self.tiled_args.get("enable_tiling", False)`` but ``__init__`` did
not initialise the attribute, so a default-constructed wrapper (or any
instance not populated by ``SeedVR2InputProcessing.execute``) raised
an opaque ``AttributeError`` after passing the original_image_video /
img_dims guards. The fix adds ``self.tiled_args = None`` to ``__init__``
and a presence + type guard in ``decode()`` ahead of the ``.get()`` call.
"""

import ast
import inspect
import textwrap
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

import comfy.ldm.seedvr.vae as vae_mod  # noqa: E402


class _PostGuardReached(Exception):
    """Sentinel raised from the patched module-level ``tiled_vae`` to prove
    every guard at the top of ``VideoAutoencoderKLWrapper.decode`` passed
    and control reached the post-guard region without standing up the
    full decode pipeline.
    """


@pytest.fixture
def _bypass_super_decode(monkeypatch):
    """Replace ``VideoAutoencoderKL.decode_`` with a stub returning a 5-D
    tensor of shape ``(B, 3, 1, 16, 16)`` matching the standin's expected
    post-decode shape. The standin built by ``_make_standin`` does NOT run
    the real ``__init__`` (which would allocate the full VAE weight set),
    so attributes the parent ``decode_`` requires (``use_slicing``,
    ``slicing_decode`` infra) are absent. On the pre-fix base SHA (where
    ``decode`` has no guards) control would otherwise die with an
    ``AttributeError`` on ``use_slicing`` BEFORE reaching the unguarded
    ``rearrange(self.original_image_video, ...)`` / ``o_h, o_w =
    self.img_dims`` sites this test class targets. On the post-fix branch
    the guards fire ahead of ``super().decode_()`` and this stub is never
    invoked, so the patch is a no-op.
    """
    def _stub(self, z, return_dict=True):
        b = z.shape[0]
        return torch.zeros(b, 3, 1, 16, 16, device=z.device)

    monkeypatch.setattr(vae_mod.VideoAutoencoderKL, "decode_", _stub)


_TILED_ARGS_DEFAULT = object()


def _make_standin(
    *,
    original_image_video,
    img_dims,
    set_img_dims=True,
    enable_tiling=False,
    tiled_args=_TILED_ARGS_DEFAULT,
    set_tiled_args=True,
):
    """Build a ``VideoAutoencoderKLWrapper`` instance via ``__new__`` so
    the real ``__init__`` (which would allocate the full VAE weight set)
    does not run. ``nn.Module.__init__`` is invoked manually so that
    ``super()`` resolution inside the borrowed ``decode`` body still
    behaves correctly under ``VideoAutoencoderKLWrapper``'s MRO.

    ``tiled_args`` defaults to ``{"enable_tiling": <enable_tiling>}`` for
    parity with the original signature. Passing an explicit value (any
    object including ``None``) overrides that default, used by the AC11
    type-guard cases. Passing ``set_tiled_args=False`` leaves the
    attribute genuinely unset on the instance, used by the AC10
    attribute-unset cell to assert the ``getattr`` fallback in
    ``decode``'s presence guard fires.
    """
    wrapper = vae_mod.VideoAutoencoderKLWrapper.__new__(
        vae_mod.VideoAutoencoderKLWrapper
    )
    nn.Module.__init__(wrapper)
    if set_tiled_args:
        if tiled_args is _TILED_ARGS_DEFAULT:
            wrapper.tiled_args = {"enable_tiling": enable_tiling}
        else:
            wrapper.tiled_args = tiled_args
    wrapper.original_image_video = original_image_video
    if set_img_dims:
        wrapper.img_dims = img_dims
    return wrapper


def test_init_initializes_img_dims_to_none():
    """AC6: ``VideoAutoencoderKLWrapper.__init__`` body contains the literal
    ``self.img_dims = None`` initialiser so the missing-state branch is
    reachable from a default-constructed instance. Source introspection
    is the contract because executing ``__init__`` would allocate the
    real VAE weight set.
    """
    src = inspect.getsource(vae_mod.VideoAutoencoderKLWrapper.__init__)
    assert "self.img_dims = None" in src, (
        "AC6: VideoAutoencoderKLWrapper.__init__ must initialise "
        "`self.img_dims = None`; missing initialiser leaves the guard's "
        "missing-state branch unreachable from a default-constructed instance.\n"
        f"--- __init__ source ---\n{src}"
    )


def test_none_original_image_video_raises_seedvr2_runtime_error(_bypass_super_decode):
    """AC1: ``original_image_video`` is ``None`` and ``img_dims`` is a valid
    ``(H, W)`` 2-tuple. ``decode(z)`` raises ``RuntimeError`` whose message
    matches ``SeedVR2.*original_image_video``. On the pre-fix base SHA the
    unguarded ``rearrange(self.original_image_video, ...)`` call instead
    raises an einops error of the form ``Tensor type unknown to einops
    <class 'NoneType'>`` — the wrong exception class, so the matcher
    misses and pytest.raises(RuntimeError) fails.
    """
    wrapper = _make_standin(original_image_video=None, img_dims=(16, 16))
    z = torch.zeros(1, 16, 2, 2)

    with pytest.raises(RuntimeError, match=r"SeedVR2.*original_image_video"):
        wrapper.decode(z)


@pytest.mark.parametrize(
    "set_img_dims",
    [True, False],
    ids=["explicit-None", "attribute-unset"],
)
def test_none_img_dims_raises_seedvr2_runtime_error(set_img_dims, _bypass_super_decode):
    """AC2: ``original_image_video`` is a valid 5-D tensor and ``img_dims``
    is either ``None`` or genuinely unset on the instance. ``decode(z)``
    raises ``RuntimeError`` whose message matches ``SeedVR2.*img_dims``.
    On the pre-fix base SHA the unguarded ``o_h, o_w = self.img_dims``
    unpack raises ``TypeError('cannot unpack non-iterable NoneType
    object')`` — the wrong exception class, so the matcher misses and
    pytest.raises(RuntimeError) fails.
    """
    oiv = torch.zeros(1, 3, 1, 16, 16)
    wrapper = _make_standin(
        original_image_video=oiv,
        img_dims=None,
        set_img_dims=set_img_dims,
    )
    if not set_img_dims:
        assert not hasattr(wrapper, "img_dims"), (
            "Standin must have no `img_dims` attribute for the unset cell"
        )
    z = torch.zeros(1, 16, 2, 2)

    with pytest.raises(RuntimeError, match=r"SeedVR2.*img_dims"):
        wrapper.decode(z)


def test_wrong_rank_original_image_video_raises_seedvr2_runtime_error(_bypass_super_decode):
    """AC3: ``original_image_video`` is a non-5-D tensor (3-D here) and
    ``img_dims`` is a valid ``(H, W)`` 2-tuple. ``decode(z)`` raises
    ``RuntimeError`` whose message matches ``SeedVR2.*original_image_video``.
    On the pre-fix base SHA the unguarded ``rearrange(...)`` instead
    raises ``einops.EinopsError`` complaining about rank mismatch — the
    wrong exception class, so the matcher misses and
    pytest.raises(RuntimeError) fails.
    """
    oiv = torch.zeros(3, 16, 16)
    wrapper = _make_standin(original_image_video=oiv, img_dims=(16, 16))
    z = torch.zeros(1, 16, 2, 2)

    with pytest.raises(RuntimeError, match=r"SeedVR2.*original_image_video"):
        wrapper.decode(z)


@pytest.mark.parametrize(
    "non_tensor",
    ["not-a-tensor", 42, {"key": "value"}, [1, 2, 3]],
    ids=["str", "int", "dict", "list"],
)
def test_non_tensor_original_image_video_raises_seedvr2_runtime_error(non_tensor, _bypass_super_decode):
    """AC7 (Copilot review round 1, comment_id=3187775620): the rank
    guard ``self.original_image_video.ndim != 5`` assumes ``.ndim``
    exists on the attribute. When a workflow assigns a non-tensor
    sentinel (str / int / dict / list — exactly the misuse the guard
    matrix is meant to harden against), the unguarded ``.ndim`` access
    raises ``AttributeError`` (or ``TypeError`` for sequences with no
    ``.ndim``) before reaching the intended SeedVR2 ``RuntimeError``.
    The fix inserts a ``torch.is_tensor`` presence check ahead of the
    rank guard so the wrong-type cell still surfaces a SeedVR2-context
    ``RuntimeError`` whose message identifies ``original_image_video``.
    """
    wrapper = _make_standin(original_image_video=non_tensor, img_dims=(16, 16))
    z = torch.zeros(1, 16, 2, 2)

    with pytest.raises(RuntimeError, match=r"SeedVR2.*original_image_video"):
        wrapper.decode(z)


@pytest.mark.parametrize(
    "img_dims",
    [(16,), (16, 16, 16)],
    ids=["1-tuple", "3-tuple"],
)
def test_wrong_arity_img_dims_raises_seedvr2_runtime_error(img_dims, _bypass_super_decode):
    """AC4: ``original_image_video`` is a valid 5-D tensor and ``img_dims``
    is a 1-tuple or 3-tuple (wrong arity). ``decode(z)`` raises
    ``RuntimeError`` whose message matches ``SeedVR2.*img_dims``. On the
    pre-fix base SHA the unguarded ``o_h, o_w = self.img_dims`` unpack
    raises ``ValueError`` (``not enough values to unpack`` for the 1-tuple,
    ``too many values to unpack`` for the 3-tuple) — the wrong exception
    class, so the matcher misses and pytest.raises(RuntimeError) fails
    for both parametrized cells.
    """
    oiv = torch.zeros(1, 3, 1, 16, 16)
    wrapper = _make_standin(original_image_video=oiv, img_dims=img_dims)
    z = torch.zeros(1, 16, 2, 2)

    with pytest.raises(RuntimeError, match=r"SeedVR2.*img_dims"):
        wrapper.decode(z)


@pytest.mark.parametrize(
    "non_sized",
    [42, 3.14, object()],
    ids=["int", "float", "bare-object"],
)
def test_non_sized_img_dims_raises_seedvr2_runtime_error(non_sized, _bypass_super_decode):
    """AC8 (Copilot review round 1, comment_id=3187775673): the arity
    guard ``len(img_dims) != 2`` assumes ``img_dims`` is a sized
    container. When a workflow assigns a non-sized sentinel
    (int / float / bare object — exactly the misuse the guard matrix is
    meant to harden against), the unguarded ``len()`` call raises
    ``TypeError: object of type 'X' has no len()`` before reaching the
    intended SeedVR2 ``RuntimeError``. The fix inserts an
    ``isinstance(img_dims, (tuple, list))`` presence check ahead of the
    arity guard so the wrong-type cell still surfaces a SeedVR2-context
    ``RuntimeError`` whose message identifies ``img_dims``.
    """
    oiv = torch.zeros(1, 3, 1, 16, 16)
    wrapper = _make_standin(original_image_video=oiv, img_dims=non_sized)
    z = torch.zeros(1, 16, 2, 2)

    with pytest.raises(RuntimeError, match=r"SeedVR2.*img_dims"):
        wrapper.decode(z)


def test_valid_state_passes_through_guards_to_tiled_vae():
    """AC5: ``original_image_video`` is a valid 5-D tensor and ``img_dims``
    is a valid ``(H, W)`` 2-tuple. With ``tiled_args = {"enable_tiling":
    True}`` the post-guard region selects the ``tiled_vae(...)`` branch.
    The module-level ``comfy.ldm.seedvr.vae.tiled_vae`` is patched with
    a sentinel side-effect; reaching it proves every guard passed.

    Two assertions are required by the contract:

    * ``mock_tiled.call_count == 1`` — the patched module-level
      ``tiled_vae`` was reached exactly once via ``_PostGuardReached``,
      proving valid input flows past the guards into the post-guard
      region.
    * AST-walked source pin (PR review round 43, Copilot
      ``comment_id=3189256300``): the function-body statement index of
      the first top-level statement transitively containing a
      ``raise RuntimeError(...)`` must precede the body-statement index
      of the ``b, tc, h, w = z.shape`` tuple-unpack assignment, proving
      the guards are placed at the top of ``decode()`` ahead of the
      existing first body line. AST-walked rather than substring-matched
      so that ``raise RuntimeError(`` appearing in a docstring or
      comment does not false-positive, and so that whitespace /
      formatting refactors that preserve the structural ordering do
      not false-negative — pattern-matching the precedent established
      by ``seedvr_model_test.py::test_no_bare_except_in_forward_path``.
      On the pre-fix base SHA this assertion FAILS because the unguarded
      ``decode()`` body contains no ``Raise(Call(Name('RuntimeError')))``
      node before the ``z.shape`` unpack assignment.
    """
    oiv = torch.zeros(1, 3, 1, 16, 16)
    wrapper = _make_standin(
        original_image_video=oiv,
        img_dims=(16, 16),
        enable_tiling=True,
    )
    z = torch.zeros(1, 16, 2, 2)

    mock_tiled = MagicMock(
        side_effect=_PostGuardReached("post-guard region reached: tiled_vae called"),
    )
    with patch.object(vae_mod, "tiled_vae", mock_tiled):
        with pytest.raises(_PostGuardReached):
            wrapper.decode(z)

    assert mock_tiled.call_count == 1, (
        "AC5 (a): patched module-level `comfy.ldm.seedvr.vae.tiled_vae` "
        f"must be called exactly once; observed call_count={mock_tiled.call_count}."
    )

    src = inspect.getsource(vae_mod.VideoAutoencoderKLWrapper.decode)
    tree = ast.parse(textwrap.dedent(src))
    func_def = tree.body[0]
    assert isinstance(func_def, ast.FunctionDef), (
        "AC5 (b): expected `inspect.getsource(VideoAutoencoderKLWrapper.decode)` "
        f"to parse to a single ast.FunctionDef; got {type(func_def).__name__}.\n"
        f"--- decode source ---\n{src}"
    )

    def _is_z_shape_unpack(stmt):
        if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
            return False
        target = stmt.targets[0]
        if not isinstance(target, ast.Tuple):
            return False
        target_names = [elt.id for elt in target.elts if isinstance(elt, ast.Name)]
        if target_names != ["b", "tc", "h", "w"]:
            return False
        value = stmt.value
        return (
            isinstance(value, ast.Attribute)
            and value.attr == "shape"
            and isinstance(value.value, ast.Name)
            and value.value.id == "z"
        )

    def _contains_raise_runtime_error(stmt):
        for node in ast.walk(stmt):
            if not isinstance(node, ast.Raise):
                continue
            exc = node.exc
            if (
                isinstance(exc, ast.Call)
                and isinstance(exc.func, ast.Name)
                and exc.func.id == "RuntimeError"
            ):
                return True
        return False

    z_shape_idx = None
    first_raise_idx = None
    for idx, stmt in enumerate(func_def.body):
        if z_shape_idx is None and _is_z_shape_unpack(stmt):
            z_shape_idx = idx
        if first_raise_idx is None and _contains_raise_runtime_error(stmt):
            first_raise_idx = idx

    assert z_shape_idx is not None, (
        "AC5 (b): VideoAutoencoderKLWrapper.decode body must contain a "
        "`b, tc, h, w = z.shape` tuple-unpack assignment (an ast.Assign with "
        "Tuple target ('b','tc','h','w') and Attribute value `z.shape`); "
        f"not found.\n--- decode source ---\n{src}"
    )
    assert first_raise_idx is not None, (
        "AC5 (b): VideoAutoencoderKLWrapper.decode body must contain at least one "
        "`raise RuntimeError(...)` (an ast.Raise whose exc is a Call to a Name "
        f"'RuntimeError'); not found.\n--- decode source ---\n{src}"
    )
    assert first_raise_idx < z_shape_idx, (
        "AC5 (b): the first body-statement transitively containing a "
        "`raise RuntimeError(...)` must appear at a position BEFORE the "
        "`b, tc, h, w = z.shape` tuple-unpack assignment in "
        "VideoAutoencoderKLWrapper.decode (proving the guards are placed at "
        "the top of decode() ahead of the existing first body line); got "
        f"first_raise_idx={first_raise_idx}, z_shape_idx={z_shape_idx}.\n"
        f"--- decode source ---\n{src}"
    )


def test_init_initializes_tiled_args_to_none():
    """AC9 (Copilot review round 8, comment_id=3188027986): the same
    missing-state failure mode the AC1/AC2 guards close also applies to
    ``self.tiled_args``. ``decode()`` unconditionally calls
    ``self.tiled_args.get("enable_tiling", False)``, but ``__init__``
    did not initialise the attribute. A default-constructed wrapper
    therefore raised an opaque ``AttributeError: 'VideoAutoencoderKLWrapper'
    object has no attribute 'tiled_args'`` after passing the
    ``original_image_video`` and ``img_dims`` guards. The fix initialises
    ``self.tiled_args = None`` in ``__init__`` so the missing-state
    branch (AC10) is reachable from a default-constructed instance.
    Source introspection is the contract because executing ``__init__``
    would allocate the real VAE weight set.
    """
    src = inspect.getsource(vae_mod.VideoAutoencoderKLWrapper.__init__)
    assert "self.tiled_args = None" in src, (
        "AC9: VideoAutoencoderKLWrapper.__init__ must initialise "
        "`self.tiled_args = None`; missing initialiser leaves the guard's "
        "missing-state branch unreachable from a default-constructed instance.\n"
        f"--- __init__ source ---\n{src}"
    )


@pytest.mark.parametrize(
    "set_tiled_args",
    [True, False],
    ids=["explicit-None", "attribute-unset"],
)
def test_none_tiled_args_raises_seedvr2_runtime_error(set_tiled_args, _bypass_super_decode):
    """AC10 (Copilot review round 8, comment_id=3188027986):
    ``original_image_video`` and ``img_dims`` are valid but ``tiled_args``
    is either ``None`` or genuinely unset on the instance. ``decode(z)``
    raises ``RuntimeError`` whose message matches ``SeedVR2.*tiled_args``.
    On the pre-fix base SHA the unguarded ``self.tiled_args.get(...)``
    call raises ``AttributeError`` (for the unset branch) or
    ``AttributeError: 'NoneType' object has no attribute 'get'`` (for
    the explicit-None branch) — the wrong exception class, so the
    matcher misses and pytest.raises(RuntimeError) fails.
    """
    oiv = torch.zeros(1, 3, 1, 16, 16)
    wrapper = _make_standin(
        original_image_video=oiv,
        img_dims=(16, 16),
        tiled_args=None,
        set_tiled_args=set_tiled_args,
    )
    if not set_tiled_args:
        assert not hasattr(wrapper, "tiled_args"), (
            "Standin must have no `tiled_args` attribute for the unset cell"
        )
    z = torch.zeros(1, 16, 2, 2)

    with pytest.raises(RuntimeError, match=r"SeedVR2.*tiled_args"):
        wrapper.decode(z)


@pytest.mark.parametrize(
    "non_dict",
    ["not-a-dict", 42, [1, 2, 3], (1, 2)],
    ids=["str", "int", "list", "tuple"],
)
def test_non_dict_tiled_args_raises_seedvr2_runtime_error(non_dict, _bypass_super_decode):
    """AC11 (Copilot review round 8, comment_id=3188027986): the
    ``self.tiled_args.get("enable_tiling", False)`` call assumes
    ``tiled_args`` is a dict. When a workflow assigns a non-dict
    sentinel (str / int / list / tuple — exactly the misuse the guard
    matrix is meant to harden against), the unguarded ``.get()`` access
    raises ``AttributeError`` (for str / int) or returns ``None`` /
    raises ``TypeError`` (for list / tuple) before reaching the
    intended SeedVR2 ``RuntimeError``. The fix inserts an
    ``isinstance(tiled_args, dict)`` type check ahead of the ``.get()``
    call so the wrong-type cell still surfaces a SeedVR2-context
    ``RuntimeError`` whose message identifies ``tiled_args``.
    """
    oiv = torch.zeros(1, 3, 1, 16, 16)
    wrapper = _make_standin(
        original_image_video=oiv,
        img_dims=(16, 16),
        tiled_args=non_dict,
    )
    z = torch.zeros(1, 16, 2, 2)

    with pytest.raises(RuntimeError, match=r"SeedVR2.*tiled_args"):
        wrapper.decode(z)
