"""Containment firewall: isolation-specific model-management behavior must never
leak into the standard (non-isolation) path. Each test toggles
``_isolation_mode_enabled`` and asserts the standard branch matches upstream while
the isolation branch keeps its cleanup semantics. Guards the gates in
``comfy/model_management.py`` and ``comfy/hooks.py``.
"""
from types import SimpleNamespace
from unittest.mock import Mock

import comfy.model_management as mm
import comfy.hooks
import comfy.lora


def _stub_caches(monkeypatch):
    monkeypatch.setattr(mm, "soft_empty_cache", lambda *a, **k: None)
    monkeypatch.setattr(mm, "reset_cast_buffers", lambda *a, **k: None)


def test_is_dead_keeps_base_and_semantics():
    obj = object()
    # base contract: dead iff the real model is still alive AND the patcher ref is gone
    assert mm.LoadedModel.is_dead(SimpleNamespace(real_model=lambda: obj, model=None)) is True
    assert mm.LoadedModel.is_dead(SimpleNamespace(real_model=lambda: obj, model=obj)) is False
    # real model collected but patcher present must NOT be dead (the OR-flip would wrongly say True)
    assert mm.LoadedModel.is_dead(SimpleNamespace(real_model=lambda: None, model=None)) is False


def test_cleanup_models_gc_standard_does_not_pop(monkeypatch):
    _stub_caches(monkeypatch)
    monkeypatch.setattr(mm, "_isolation_mode_enabled", lambda: False)
    dead = SimpleNamespace(is_dead=lambda: True, real_model=lambda: object(),
                           model=SimpleNamespace(cleanup=Mock()))
    monkeypatch.setattr(mm, "current_loaded_models", [dead])
    mm.cleanup_models_gc()
    assert dead in mm.current_loaded_models            # base: warn-only, never popped
    assert not dead.model.cleanup.called               # base: no cleanup() on gc


def test_cleanup_models_gc_isolation_pops_and_cleans(monkeypatch):
    _stub_caches(monkeypatch)
    monkeypatch.setattr(mm, "_isolation_mode_enabled", lambda: True)
    spy = Mock()
    stale = SimpleNamespace(dead_state=lambda: (True, False), model=SimpleNamespace(cleanup=spy))
    monkeypatch.setattr(mm, "current_loaded_models", [stale])
    mm.cleanup_models_gc()
    assert stale not in mm.current_loaded_models        # isolation: stale entry popped
    assert spy.called                                   # isolation: cleanup() fired


def test_cleanup_models_standard_pops_without_cleanup(monkeypatch):
    monkeypatch.setattr(mm, "_isolation_mode_enabled", lambda: False)
    spy = Mock()
    dead = SimpleNamespace(real_model=lambda: None, model=SimpleNamespace(cleanup=spy))
    monkeypatch.setattr(mm, "current_loaded_models", [dead])
    mm.cleanup_models()
    assert dead not in mm.current_loaded_models         # base detection still pops it
    assert not spy.called                               # base: never calls cleanup()


def test_cleanup_models_isolation_calls_cleanup(monkeypatch):
    monkeypatch.setattr(mm, "_isolation_mode_enabled", lambda: True)
    spy = Mock()
    dead = SimpleNamespace(real_model=lambda: None, model=SimpleNamespace(cleanup=spy))
    monkeypatch.setattr(mm, "current_loaded_models", [dead])
    mm.cleanup_models()
    assert dead not in mm.current_loaded_models
    assert spy.called                                   # isolation: cleanup() fired


def test_weighthook_weights_none_guard_is_isolation_only(monkeypatch):
    monkeypatch.setattr(comfy.lora, "model_lora_keys_unet", lambda *a, **k: {})
    monkeypatch.setattr(comfy.lora, "load_lora", lambda *a, **k: {})
    monkeypatch.delenv("PYISOLATE_CHILD", raising=False)
    model = SimpleNamespace(model=object(), add_hook_patches=lambda **k: None)
    target_dict = {"target": None}

    def make_hook():
        h = object.__new__(comfy.hooks.WeightHook)
        h.weights = None
        h.need_weight_init = True
        h._strength_model = 1.0
        h._strength_clip = 1.0
        h.should_register = lambda *a, **k: True
        return h

    monkeypatch.setattr(comfy.hooks.args, "use_process_isolation", False, raising=False)
    off = make_hook()
    off.add_hook_patches(model, {}, target_dict, Mock())
    assert off.weights is None                          # base: weights stays None

    monkeypatch.setattr(comfy.hooks.args, "use_process_isolation", True, raising=False)
    on = make_hook()
    on.add_hook_patches(model, {}, target_dict, Mock())
    assert on.weights == {}                             # isolation: coerced to {}
