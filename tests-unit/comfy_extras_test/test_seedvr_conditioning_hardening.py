"""Regression tests for SeedVR2 conditioning model resolution and RoPE
frequency-cast caching. Anchored by pollockjj/mydevelopment#183 and the
CodeRabbit r2962219538 finding on Comfy-Org/ComfyUI#11294.

Pin two behaviors:
  1. ``_resolve_seedvr2_diffusion_model`` returns the inner diffusion-model
     for the expected ``model.model.diffusion_model`` shape and fails loud
     with a ``RuntimeError`` whose message begins with
     ``_SEEDVR2_INVALID_MODEL_MSG_PREFIX`` for any other shape.
  2. ``_apply_rope_freqs_float32_cast`` iterates the diffusion-model module
     tree exactly once per resolved instance; subsequent calls against the
     same instance short-circuit.

Import isolation: ``comfy.model_management`` is stubbed via direct
``sys.modules`` assignment so importing ``comfy_extras.nodes_seedvr`` does
not trigger GPU/server-side initialization. ``patch.dict`` is intentionally
NOT used here because its snapshot/restore semantics evict transitively
imported third-party modules (e.g. ``torchvision``) on exit, which causes
``torch``'s global op-library Meta-key registrations to double-register on
re-import. Module-level cached import + scoped restore of the single
mocked entry avoids that hazard.
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
    ``restore()`` undoes only the single ``comfy.model_management``
    sys.modules entry that this helper introduced.
    """
    prior_comfy_mm = sys.modules.get("comfy.model_management", _SENTINEL)
    prior_comfy_mm_attr = _SENTINEL
    comfy_pkg = sys.modules.get("comfy")
    if comfy_pkg is not None:
        prior_comfy_mm_attr = getattr(comfy_pkg, "model_management", _SENTINEL)

    sys.modules["comfy.model_management"] = MagicMock()
    if "comfy_extras.nodes_seedvr" in sys.modules:
        nodes_seedvr = sys.modules["comfy_extras.nodes_seedvr"]
    else:
        nodes_seedvr = importlib.import_module("comfy_extras.nodes_seedvr")

    def _restore():
        if prior_comfy_mm is _SENTINEL:
            sys.modules.pop("comfy.model_management", None)
        else:
            sys.modules["comfy.model_management"] = prior_comfy_mm
        comfy_pkg_now = sys.modules.get("comfy")
        if comfy_pkg_now is not None:
            if prior_comfy_mm_attr is _SENTINEL:
                if hasattr(comfy_pkg_now, "model_management"):
                    delattr(comfy_pkg_now, "model_management")
            else:
                setattr(comfy_pkg_now, "model_management", prior_comfy_mm_attr)

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


class _ModelInner:
    def __init__(self, diffusion_model):
        self.diffusion_model = diffusion_model


class _ModelPatcher:
    def __init__(self, diffusion_model):
        self.model = _ModelInner(diffusion_model)


def test_resolve_seedvr2_diffusion_model_returns_inner_when_valid():
    nodes_seedvr, restore = _import_nodes_seedvr_isolated()
    try:
        diffusion_model = _DiffusionModel()
        patcher = _ModelPatcher(diffusion_model)
        resolved = nodes_seedvr._resolve_seedvr2_diffusion_model(patcher)
        assert resolved is diffusion_model
    finally:
        restore()


def test_resolve_seedvr2_diffusion_model_raises_runtime_error_with_specific_prefix():
    nodes_seedvr, restore = _import_nodes_seedvr_isolated()
    try:
        class _NoModelAttr:
            pass

        with pytest.raises(RuntimeError) as excinfo:
            nodes_seedvr._resolve_seedvr2_diffusion_model(_NoModelAttr())
        assert str(excinfo.value).startswith(
            nodes_seedvr._SEEDVR2_INVALID_MODEL_MSG_PREFIX
        )

        class _NoDiffusionAttr:
            def __init__(self):
                self.model = object()

        with pytest.raises(RuntimeError) as excinfo2:
            nodes_seedvr._resolve_seedvr2_diffusion_model(_NoDiffusionAttr())
        assert str(excinfo2.value).startswith(
            nodes_seedvr._SEEDVR2_INVALID_MODEL_MSG_PREFIX
        )
    finally:
        restore()


def test_apply_rope_freqs_float32_cast_iterates_once_then_caches():
    nodes_seedvr, restore = _import_nodes_seedvr_isolated()
    try:
        diffusion_model = _DiffusionModel()

        # Ensure starting dtype is non-float32 so the cast is observable.
        for module in diffusion_model.modules():
            if hasattr(module, "rope") and hasattr(module.rope, "freqs"):
                module.rope.freqs.data = module.rope.freqs.data.to(torch.float64)

        # First call: should iterate the module tree and cast every
        # rope.freqs to float32.
        nodes_seedvr._apply_rope_freqs_float32_cast(diffusion_model)
        for module in diffusion_model.modules():
            if hasattr(module, "rope") and hasattr(module.rope, "freqs"):
                assert module.rope.freqs.data.dtype == torch.float32

        # Force every rope.freqs back to float64 manually. A naive
        # (uncached) implementation would re-cast on the next call.
        for module in diffusion_model.modules():
            if hasattr(module, "rope") and hasattr(module.rope, "freqs"):
                module.rope.freqs.data = module.rope.freqs.data.to(torch.float64)

        # Second call: cache hit must short-circuit; rope.freqs must NOT
        # be re-cast and must remain float64.
        nodes_seedvr._apply_rope_freqs_float32_cast(diffusion_model)
        for module in diffusion_model.modules():
            if hasattr(module, "rope") and hasattr(module.rope, "freqs"):
                assert module.rope.freqs.data.dtype == torch.float64, (
                    "RoPE float32 cast must be cached after the first call: "
                    "the second invocation must short-circuit and not iterate "
                    "the module tree."
                )
    finally:
        restore()
