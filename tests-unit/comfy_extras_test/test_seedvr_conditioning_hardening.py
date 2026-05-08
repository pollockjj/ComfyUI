"""Regression tests for SeedVR2 conditioning model resolution and RoPE
frequency cast. Anchored by pollockjj/mydevelopment#183 and the CodeRabbit
r2962219538 finding on Comfy-Org/ComfyUI#11294.

Pin two behaviors:

  1. ``_resolve_seedvr2_diffusion_model`` returns the inner diffusion-model
     for the expected ``model.model.diffusion_model`` shape and fails loud
     with a ``RuntimeError`` whose message begins with
     ``_SEEDVR2_INVALID_MODEL_MSG_PREFIX`` for any other shape, including
     the four distinct missing-vs-None subcases of the chain.
  2. ``_apply_rope_freqs_float32_cast`` is idempotent **per-tensor by
     dtype check**, NOT per-instance by sentinel attribute. Every call
     walks the diffusion-model module tree and invokes ``.to(float32)``
     only on tensors whose dtype is not already ``float32``. The cache-by-
     attribute approach was rejected on PR pollockjj/ComfyUI#32 because
     the sentinel survives Comfy's dynamic model unload/reload cycle while
     ``rope.freqs`` itself is restored to the archived dtype, so the next
     call would short-circuit and leave RoPE running in fp16/bf16 — the
     exact failure the helper is supposed to prevent. The dtype check is
     self-correcting against any weight-restore lifecycle event.

Import isolation: ``comfy.model_management`` is stubbed via direct
``sys.modules`` assignment so importing ``comfy_extras.nodes_seedvr`` does
not trigger GPU/server-side initialization. ``patch.dict`` is intentionally
NOT used here because its snapshot/restore semantics evict transitively
imported third-party modules (e.g. ``torchvision``) on exit, which causes
``torch``'s global op-library Meta-key registrations to double-register on
re-import. Module-level cached import + scoped restore of the four mocked
entries avoids that hazard. See ``_import_nodes_seedvr_isolated``.
"""

import importlib
import sys
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn


_SENTINEL = object()


def _import_nodes_seedvr_isolated():
    """Stub ``comfy.model_management``, import (or reuse a cached import of)
    ``comfy_extras.nodes_seedvr``, and return ``(module, restore)``.

    ``restore()`` snapshots and restores **three** in-process import-state
    surfaces (Copilot review on PR pollockjj/ComfyUI#32):

      1. ``sys.modules["comfy.model_management"]`` — the stubbed module.
      2. ``sys.modules["comfy_extras.nodes_seedvr"]`` — the imported test
         target. If we leave this in ``sys.modules`` after the test, a
         later test importing the real ``comfy_extras.nodes_seedvr`` will
         get our stubbed-``comfy.model_management`` cached version, which
         does not re-resolve against the real ``comfy.model_management``.
      3. ``comfy_extras.nodes_seedvr`` package attribute on the
         ``comfy_extras`` package, mirroring the existing
         ``comfy.model_management`` attribute restore. Mirrors the
         pattern in ``test_seedvr_node_signature.py``.

    All three are restored verbatim if previously set; deleted on exit
    if previously unset. No global state leaks into later tests.
    """
    prior_comfy_mm = sys.modules.get("comfy.model_management", _SENTINEL)
    prior_comfy_mm_attr = _SENTINEL
    comfy_pkg = sys.modules.get("comfy")
    if comfy_pkg is not None:
        prior_comfy_mm_attr = getattr(comfy_pkg, "model_management", _SENTINEL)
    prior_nodes_seedvr_module = sys.modules.get(
        "comfy_extras.nodes_seedvr", _SENTINEL,
    )
    prior_nodes_seedvr_attr = _SENTINEL
    comfy_extras_pkg = sys.modules.get("comfy_extras")
    if comfy_extras_pkg is not None:
        prior_nodes_seedvr_attr = getattr(
            comfy_extras_pkg, "nodes_seedvr", _SENTINEL,
        )

    sys.modules["comfy.model_management"] = MagicMock()
    if "comfy_extras.nodes_seedvr" in sys.modules:
        nodes_seedvr = sys.modules["comfy_extras.nodes_seedvr"]
    else:
        nodes_seedvr = importlib.import_module("comfy_extras.nodes_seedvr")

    def _restore():
        # 1. comfy.model_management sys.modules entry
        if prior_comfy_mm is _SENTINEL:
            sys.modules.pop("comfy.model_management", None)
        else:
            sys.modules["comfy.model_management"] = prior_comfy_mm
        # 2. comfy.model_management package attribute on comfy
        comfy_pkg_now = sys.modules.get("comfy")
        if comfy_pkg_now is not None:
            if prior_comfy_mm_attr is _SENTINEL:
                if hasattr(comfy_pkg_now, "model_management"):
                    delattr(comfy_pkg_now, "model_management")
            else:
                setattr(comfy_pkg_now, "model_management", prior_comfy_mm_attr)
        # 3. comfy_extras.nodes_seedvr sys.modules entry
        if prior_nodes_seedvr_module is _SENTINEL:
            sys.modules.pop("comfy_extras.nodes_seedvr", None)
        else:
            sys.modules["comfy_extras.nodes_seedvr"] = prior_nodes_seedvr_module
        # 4. comfy_extras.nodes_seedvr package attribute on comfy_extras
        comfy_extras_pkg_now = sys.modules.get("comfy_extras")
        if comfy_extras_pkg_now is not None:
            if prior_nodes_seedvr_attr is _SENTINEL:
                if hasattr(comfy_extras_pkg_now, "nodes_seedvr"):
                    delattr(comfy_extras_pkg_now, "nodes_seedvr")
            else:
                setattr(
                    comfy_extras_pkg_now, "nodes_seedvr",
                    prior_nodes_seedvr_attr,
                )

    return nodes_seedvr, _restore


class _Rope(nn.Module):
    def __init__(self):
        super().__init__()
        self.freqs = nn.Parameter(torch.zeros(4))


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.rope = _Rope()


class _DiffusionModel(nn.Module):
    def __init__(self, n_blocks=3):
        super().__init__()
        self.blocks = nn.ModuleList([_Block() for _ in range(n_blocks)])
        self.register_buffer("positive_conditioning", torch.ones((2, 4)))
        self.register_buffer("negative_conditioning", torch.zeros((3, 4)))


class _ModelInner:
    def __init__(self, diffusion_model):
        self.diffusion_model = diffusion_model


class _ModelPatcher:
    def __init__(self, diffusion_model):
        self.model = _ModelInner(diffusion_model)
        self.disable_cfg1_optimization_calls = 0

    def disable_model_cfg1_optimization(self):
        self.disable_cfg1_optimization_calls += 1


def test_resolve_seedvr2_diffusion_model_returns_inner_when_valid():
    nodes_seedvr, restore = _import_nodes_seedvr_isolated()
    try:
        diffusion_model = _DiffusionModel()
        patcher = _ModelPatcher(diffusion_model)
        resolved = nodes_seedvr._resolve_seedvr2_diffusion_model(patcher)
        assert resolved is diffusion_model
    finally:
        restore()


def test_seedvr2_conditioning_disables_cfg1_optimization():
    """SeedVR2's native model consumes paired negative/positive text
    context. Comfy's CFG=1 fast path drops the negative branch, so the
    conditioning node must disable that optimization on the model patcher.
    """
    nodes_seedvr, restore = _import_nodes_seedvr_isolated()
    try:
        diffusion_model = _DiffusionModel()
        patcher = _ModelPatcher(diffusion_model)
        vae_conditioning = {"samples": torch.zeros((1, 1, 1, 1, 2))}

        positive, negative, latent = nodes_seedvr.SeedVR2Conditioning.execute(
            vae_conditioning,
            patcher,
            0.0,
        )

        assert patcher.disable_cfg1_optimization_calls == 1
        assert positive[0][0].shape == (1, 3, 4)
        assert negative[0][0].shape == (1, 3, 4)
        assert latent["samples"].shape == (1, 2, 1, 1)
    finally:
        restore()


def test_resolve_seedvr2_diffusion_model_raises_runtime_error_with_specific_prefix():
    """Pin all four failure modes of the resolver chain to the same error
    prefix and to message text that distinguishes 'attribute missing'
    from 'attribute present but None' (Copilot review on PR
    pollockjj/ComfyUI#32). The four modes:

      mode 1: input has no 'model' attribute
      mode 2: input.model is None
      mode 3: 'model.model' has no 'diffusion_model' attribute
      mode 4: 'model.model.diffusion_model' is None
    """
    nodes_seedvr, restore = _import_nodes_seedvr_isolated()
    try:
        # Mode 1: model has no 'model' attribute at all.
        class _NoModelAttr:
            pass

        with pytest.raises(RuntimeError) as excinfo:
            nodes_seedvr._resolve_seedvr2_diffusion_model(_NoModelAttr())
        msg = str(excinfo.value)
        assert msg.startswith(nodes_seedvr._SEEDVR2_INVALID_MODEL_MSG_PREFIX)
        assert "no 'model' attribute" in msg

        # Mode 2: model.model exists but is None (must not be conflated
        # with "no 'model' attribute").
        class _ModelIsNone:
            def __init__(self):
                self.model = None

        with pytest.raises(RuntimeError) as excinfo:
            nodes_seedvr._resolve_seedvr2_diffusion_model(_ModelIsNone())
        msg = str(excinfo.value)
        assert msg.startswith(nodes_seedvr._SEEDVR2_INVALID_MODEL_MSG_PREFIX)
        assert "input.model is None" in msg

        # Mode 3: model.model exists, has no 'diffusion_model' attribute.
        class _NoDiffusionAttr:
            def __init__(self):
                self.model = object()

        with pytest.raises(RuntimeError) as excinfo:
            nodes_seedvr._resolve_seedvr2_diffusion_model(_NoDiffusionAttr())
        msg = str(excinfo.value)
        assert msg.startswith(nodes_seedvr._SEEDVR2_INVALID_MODEL_MSG_PREFIX)
        assert "no 'diffusion_model' attribute" in msg

        # Mode 4: model.model.diffusion_model exists but is None (must not
        # be conflated with "no 'diffusion_model' attribute").
        class _DiffusionIsNoneInner:
            def __init__(self):
                self.diffusion_model = None

        class _DiffusionIsNone:
            def __init__(self):
                self.model = _DiffusionIsNoneInner()

        with pytest.raises(RuntimeError) as excinfo:
            nodes_seedvr._resolve_seedvr2_diffusion_model(_DiffusionIsNone())
        msg = str(excinfo.value)
        assert msg.startswith(nodes_seedvr._SEEDVR2_INVALID_MODEL_MSG_PREFIX)
        assert "'model.model.diffusion_model' is None" in msg
    finally:
        restore()


def test_apply_rope_freqs_float32_cast_idempotent_on_unchanged_dtype():
    """Calling the helper twice on a model whose rope.freqs is already
    float32 must NOT mutate the tensor identity or contents — the dtype
    check on every nested module short-circuits the .to() call when the
    tensor is already in float32.
    """
    nodes_seedvr, restore = _import_nodes_seedvr_isolated()
    try:
        diffusion_model = _DiffusionModel()

        # Starting dtype is non-float32 so the first call has work to do.
        for module in diffusion_model.modules():
            if hasattr(module, "rope") and hasattr(module.rope, "freqs"):
                module.rope.freqs.data = module.rope.freqs.data.to(torch.float64)

        nodes_seedvr._apply_rope_freqs_float32_cast(diffusion_model)
        first_call_data_ids = []
        for module in diffusion_model.modules():
            if hasattr(module, "rope") and hasattr(module.rope, "freqs"):
                assert module.rope.freqs.data.dtype == torch.float32
                first_call_data_ids.append(id(module.rope.freqs.data))

        # Second call on the same already-float32 model: every per-tensor
        # dtype check sees float32 and skips the .to() call. Tensor data
        # identity must be preserved (no re-allocation).
        nodes_seedvr._apply_rope_freqs_float32_cast(diffusion_model)
        for module, prior_id in zip(
            (m for m in diffusion_model.modules()
             if hasattr(m, "rope") and hasattr(m.rope, "freqs")),
            first_call_data_ids,
            strict=True,
        ):
            assert module.rope.freqs.data.dtype == torch.float32
            assert id(module.rope.freqs.data) == prior_id, (
                "Already-float32 rope.freqs must not be re-allocated on "
                "subsequent calls; the per-tensor dtype check must skip the "
                ".to(float32) call when the tensor is already in float32."
            )
    finally:
        restore()


def test_apply_rope_freqs_float32_cast_recovers_after_dtype_reset():
    """After a model unload/reload that restores rope.freqs from an
    archived non-float32 dtype, the next call must re-cast to float32 —
    the original bool-sentinel implementation would short-circuit here
    and leave RoPE running in fp16/bf16 (codex P1 finding on PR #32).
    """
    nodes_seedvr, restore = _import_nodes_seedvr_isolated()
    try:
        diffusion_model = _DiffusionModel()
        for module in diffusion_model.modules():
            if hasattr(module, "rope") and hasattr(module.rope, "freqs"):
                module.rope.freqs.data = module.rope.freqs.data.to(torch.float64)

        # First call casts to float32.
        nodes_seedvr._apply_rope_freqs_float32_cast(diffusion_model)
        for module in diffusion_model.modules():
            if hasattr(module, "rope") and hasattr(module.rope, "freqs"):
                assert module.rope.freqs.data.dtype == torch.float32

        # Simulate a Comfy dynamic unload/reload that restores rope.freqs
        # to the archived (non-float32) dtype.
        for module in diffusion_model.modules():
            if hasattr(module, "rope") and hasattr(module.rope, "freqs"):
                module.rope.freqs.data = module.rope.freqs.data.to(torch.float64)

        # Second call must detect the dtype regression and re-cast.
        nodes_seedvr._apply_rope_freqs_float32_cast(diffusion_model)
        for module in diffusion_model.modules():
            if hasattr(module, "rope") and hasattr(module.rope, "freqs"):
                assert module.rope.freqs.data.dtype == torch.float32, (
                    "After a model unload/reload that resets rope.freqs to "
                    "non-float32, the next _apply_rope_freqs_float32_cast "
                    "call MUST re-cast to float32. A bool-sentinel cache "
                    "would have short-circuited here."
                )
    finally:
        restore()

