"""Regression tests for SeedVR2 forward-path device discipline.

Tracks pollockjj/mydevelopment#185 (parent #101). Kosinkadink flagged
five line ranges in PR #11294 where ``forward`` paths in
``comfy/ldm/seedvr/model.py`` called
``comfy.model_management.get_torch_device()`` and used ``.to(device)``
to override Comfy's model-management placement. Yousef removed four;
``MMModule.forward`` retained the pattern (lines 624-625 at upstream
commit ``553f71aa``):

    device = comfy.model_management.get_torch_device()
    vid = vid.to(device)

These two lines were the residual. They are now removed: ``vid``
flows through ``vid_module`` on the device chosen by Comfy's
``load_device``, and ``txt`` follows ``vid.device`` on the existing
line below the removed cast — the Comfy-native pattern already used
by ``NaMMRotaryEmbedding3d.forward`` (which calls
``vid_freqs.to(target_device)`` where ``target_device = vid_q.device``).

The two regression layers below pin the fix:

  1. **Source-level invariant** (AST-walked, not substring): no
     ``forward`` method body in ``comfy/ldm/seedvr/model.py`` may
     call ``comfy.model_management.get_torch_device()``. Locking
     this at AST level means a future regression that moves the
     call into a helper still trips, while ``get_torch_device``
     references in docstrings / comments do not false-positive.

  2. **Smoke forward** (stub-weight, no real SeedVR2 checkpoint):
     a small ``MMModule(nn.Linear, ...)`` runs forward on tensors
     placed on a deliberately non-default device choice. The output
     stays on the input device — proving the forward path no longer
     yanks tensors to ``get_torch_device()``. The ``seedvr_model_test.py``
     stand-in pattern is the precedent for stubbing ``MMModule``
     without standing up a full SeedVR2 transformer.
"""

from comfy.cli_args import args
import torch

if not torch.cuda.is_available():
    args.cpu = True

import ast  # noqa: E402
import inspect  # noqa: E402
import textwrap  # noqa: E402

import pytest  # noqa: E402
from torch import nn  # noqa: E402

from comfy.ldm.seedvr import model as seedvr_model  # noqa: E402
from comfy.ldm.seedvr.model import MMModule  # noqa: E402


def _iter_forward_method_sources():
    """Yield (qualname, source) for every ``def forward(...)`` defined
    inside ``comfy/ldm/seedvr/model.py``.

    Walks the module's AST so that nested classes / methods are caught
    deterministically, independent of class-discovery order.
    """
    module_src = inspect.getsource(seedvr_model)
    tree = ast.parse(module_src)
    for cls in ast.walk(tree):
        if not isinstance(cls, ast.ClassDef):
            continue
        for item in cls.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == "forward":
                qualname = f"{cls.name}.forward"
                yield qualname, ast.get_source_segment(module_src, item)


def _calls_get_torch_device(src):
    """Return True if the AST of ``src`` contains a call to
    ``comfy.model_management.get_torch_device`` or
    ``model_management.get_torch_device`` — including bare
    ``get_torch_device()`` after a local rebinding.
    """
    tree = ast.parse(textwrap.dedent(src))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "get_torch_device":
            return True
        if isinstance(func, ast.Name) and func.id == "get_torch_device":
            return True
    return False


def test_no_get_torch_device_in_any_seedvr_forward():
    """AC: No ``forward`` method in ``comfy/ldm/seedvr/model.py``
    calls ``comfy.model_management.get_torch_device()``. Phase 0
    audit identified one site (``MMModule.forward`` at lines
    624-625 of upstream ``553f71aa``); this test pins the
    repository-level invariant at zero residual sites.

    AST-walked rather than grep'd so that occurrences inside
    docstrings / comments do not false-positive.
    """
    offenders = []
    for qualname, src in _iter_forward_method_sources():
        if _calls_get_torch_device(src):
            offenders.append(qualname)
    assert offenders == [], (
        "comfy/ldm/seedvr/model.py forward methods must not call "
        "comfy.model_management.get_torch_device(); offenders: "
        f"{offenders}"
    )


def test_mmmodule_source_no_longer_imports_model_management():
    """AC complement: with the cast removed, ``comfy.model_management``
    is no longer referenced anywhere in ``comfy/ldm/seedvr/model.py``.
    Pinning the import-level state catches regressions that re-import
    the module for a fresh forward-path cast.
    """
    src = inspect.getsource(seedvr_model)
    assert "comfy.model_management" not in src, (
        "comfy/ldm/seedvr/model.py must not reference "
        "comfy.model_management after the forward-path cast removal"
    )


def _device_for_smoke():
    """Pick a real device for the smoke forward.

    On CUDA hosts use cuda:0 — distinct from CPU so that the test
    actually exercises a non-trivial device choice. On CPU-only hosts
    fall back to CPU; the source-level AC still pins the invariant.
    """
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def test_mmmodule_forward_preserves_caller_device():
    """AC: ``MMModule.forward`` runs without manual cast and the
    output stays on the input tensor's device — proving the forward
    path no longer relies on ``get_torch_device()`` for placement.

    Stub: ``MMModule(nn.Linear, in_features, out_features,
    shared_weights=False)`` builds two real ``nn.Linear`` modules
    (one for ``vid``, one for ``txt``). No SeedVR2 weights required;
    this is the same stand-in shape used by ``seedvr_model_test.py``.
    """
    device = _device_for_smoke()
    in_features, out_features = 16, 8
    mm = MMModule(nn.Linear, in_features, out_features, shared_weights=False).to(device)

    vid = torch.randn(4, in_features, device=device)
    txt = torch.randn(3, in_features, device=device)

    vid_out, txt_out = mm(vid, txt)

    assert vid_out.device == device, (
        f"vid output must stay on input device {device}, got {vid_out.device}"
    )
    assert txt_out.device == device, (
        f"txt output must stay on input device {device}, got {txt_out.device}"
    )
    assert vid_out.shape == (4, out_features)
    assert txt_out.shape == (3, out_features)


def test_mmmodule_forward_vid_only_path():
    """AC complement: the ``vid_only=True`` branch (used by
    ``mlp_norm`` on the last layer; ``ada`` on the last layer) skips
    the ``txt`` half entirely. ``txt`` arrives but is returned
    unchanged. The smoke forward must work on this branch without
    any device cast.
    """
    device = _device_for_smoke()
    in_features, out_features = 12, 6
    mm = MMModule(
        nn.Linear,
        in_features,
        out_features,
        shared_weights=False,
        vid_only=True,
    ).to(device)

    vid = torch.randn(2, in_features, device=device)
    txt = torch.randn(2, in_features, device=device)
    txt_in = txt.clone()

    vid_out, txt_out = mm(vid, txt)

    assert vid_out.device == device
    assert vid_out.shape == (2, out_features)
    assert torch.equal(txt_out, txt_in), (
        "vid_only branch must return txt unchanged"
    )


def test_mmmodule_forward_shared_weights_path():
    """AC complement: ``shared_weights=True`` collapses ``vid`` and
    ``txt`` into a single ``self.all`` module. The smoke forward
    must work on this branch without any device cast.
    """
    device = _device_for_smoke()
    in_features, out_features = 10, 5
    mm = MMModule(
        nn.Linear,
        in_features,
        out_features,
        shared_weights=True,
    ).to(device)

    vid = torch.randn(3, in_features, device=device)
    txt = torch.randn(2, in_features, device=device)

    vid_out, txt_out = mm(vid, txt)

    assert vid_out.device == device
    assert txt_out.device == device
    assert vid_out.shape == (3, out_features)
    assert txt_out.shape == (2, out_features)


def test_mmmodule_forward_txt_follows_vid_device():
    """AC: when ``vid`` and ``txt`` enter on the same device, the
    surviving ``txt = txt.to(device=vid.device, dtype=vid.dtype)``
    line on the post-cast forward path is a no-op cast. The output
    must be bit-identical to a direct ``txt_module(txt)`` call.

    This pins the Comfy-native pattern (txt follows vid) as the
    intended discipline — distinct from the removed
    ``get_torch_device()`` global override.
    """
    device = _device_for_smoke()
    in_features, out_features = 8, 4
    torch.manual_seed(0)
    mm = MMModule(nn.Linear, in_features, out_features, shared_weights=False).to(device)

    vid = torch.randn(2, in_features, device=device)
    txt = torch.randn(2, in_features, device=device)

    _, txt_out = mm(vid, txt)
    direct = mm.txt(txt)
    assert torch.equal(txt_out, direct), (
        "txt half must match a direct txt_module(txt) call when vid "
        "and txt already share device/dtype"
    )


def test_mmmodule_forward_does_not_import_model_management_at_call_time():
    """AC complement: confirm ``MMModule.forward`` runs without
    ``comfy.model_management`` being importable from inside the
    forward closure. Asserts the source of the method does not
    name ``comfy`` at all — defensive belt-and-suspenders pin
    against a future ``comfy.model_management`` re-introduction.
    """
    src = inspect.getsource(MMModule.forward)
    tree = ast.parse(textwrap.dedent(src))
    names = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name)
    }
    assert "comfy" not in names, (
        "MMModule.forward must not reference the 'comfy' top-level "
        f"namespace; names found: {sorted(names)}"
    )
