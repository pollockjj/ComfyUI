"""Parity guard for the ModelPatcher RPC proxy (class 3).

The boundary-crossing multigpu / checkpoint-save additions on ModelPatcher must be
relayed by *class-defined* methods on ModelPatcherProxy. If one falls through to the
closed ``__getattr__`` whitelist it raises AttributeError the moment an isolated
MultiGPU node or CheckpointSave node touches it. This pins the three multigpu/save
relays (deepclone_multigpu, register_load_device, state_dict_for_saving) plus
match_multigpu_clones and proves each relays to a matching registry handler.
"""
from unittest.mock import Mock

from comfy.isolation.model_patcher_proxy import ModelPatcherProxy
from comfy.isolation.model_patcher_proxy_registry import ModelPatcherRegistry

BOUNDARY_CROSSING = (
    "deepclone_multigpu",
    "match_multigpu_clones",
    "register_load_device",
    "state_dict_for_saving",
)


def test_boundary_crossing_methods_are_class_defined_relays():
    for name in BOUNDARY_CROSSING:
        assert name in vars(ModelPatcherProxy), (
            f"{name} is not class-defined on ModelPatcherProxy; it would hit the "
            f"__getattr__ whitelist and AttributeError under isolation"
        )
        assert callable(getattr(ModelPatcherRegistry, name, None)), (
            f"ModelPatcherRegistry has no host handler for {name}"
        )


def test_proxy_relays_to_matching_registry_method_names(monkeypatch):
    proxy = ModelPatcherProxy("inst-0", Mock())
    calls = []
    monkeypatch.setattr(proxy, "_call_rpc",
                        lambda name, *a, **k: (calls.append((name, a)), "child-id")[1])
    monkeypatch.setattr(proxy, "_spawn_related_proxy", lambda new_id: ("spawned", new_id))

    proxy.register_load_device("cuda:1")
    proxy.match_multigpu_clones()
    saved = proxy.state_dict_for_saving("clip", "vae", "clipvision")
    cloned = proxy.deepclone_multigpu(new_load_device="cuda:1")

    assert [c[0] for c in calls] == [
        "register_load_device", "match_multigpu_clones",
        "state_dict_for_saving", "deepclone_multigpu",
    ]
    assert saved == "child-id"                     # value relay returns the host result
    assert cloned == ("spawned", "child-id")       # deepclone wraps the new patcher id in a proxy
    assert calls[0][1] == ("cuda:1",)              # device arg relayed verbatim
    assert calls[3][1] == ("cuda:1",)              # deepclone relays only new_load_device


def test_clip_proxy_relays_state_dict_for_saving():
    # CheckpointSave on an isolated CLIP relays clip.state_dict_for_saving (sd.py:2089).
    from comfy.isolation.clip_proxy import CLIPProxy, CLIPRegistry

    assert "state_dict_for_saving" in vars(CLIPProxy)
    assert callable(getattr(CLIPRegistry, "state_dict_for_saving", None))


def test_vae_proxy_relays_compression_ratio_methods():
    # Tiling/controlnet math runs these on an isolated VAE before decode/encode
    # (nodes.py:336/344, controlnet.py:275).
    from comfy.isolation.vae_proxy import VAEProxy, VAERegistry

    for name in ("spacial_compression_decode", "spacial_compression_encode", "temporal_compression_decode"):
        assert name in vars(VAEProxy), f"VAEProxy missing relay for {name}"
        assert callable(getattr(VAERegistry, name, None)), f"VAERegistry missing handler for {name}"
