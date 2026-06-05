from __future__ import annotations

import ast
from pathlib import Path
from types import MethodType

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def _class_node(relative_path: str, class_name: str) -> ast.ClassDef:
    tree = ast.parse((REPO_ROOT / relative_path).read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return node
    raise AssertionError(f"{class_name} not found in {relative_path}")


def _public_methods(class_node: ast.ClassDef) -> set[str]:
    methods = set()
    for node in class_node.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name.startswith("_"):
            continue
        decorators = {
            getattr(decorator, "id", None) or getattr(decorator, "attr", None)
            for decorator in node.decorator_list
        }
        if "property" in decorators or "setter" in decorators:
            continue
        methods.add(node.name)
    return methods


def _public_properties(class_node: ast.ClassDef) -> set[str]:
    properties = set()
    for node in class_node.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name.startswith("_"):
            continue
        decorators = {
            getattr(decorator, "id", None) or getattr(decorator, "attr", None)
            for decorator in node.decorator_list
        }
        if "property" in decorators:
            properties.add(node.name)
    return properties


def _assigned_public_attrs(class_node: ast.ClassDef) -> set[str]:
    attrs = set()
    for node in ast.walk(class_node):
        targets = []
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if hasattr(node, "targets") else [node.target]
        for target in targets:
            if not isinstance(target, ast.Attribute):
                continue
            if not isinstance(target.value, ast.Name) or target.value.id != "self":
                continue
            if not target.attr.startswith("_"):
                attrs.add(target.attr)
        if isinstance(node, ast.Call):
            func = node.func
            if not (
                isinstance(func, ast.Attribute)
                and func.attr == "register_buffer"
                and isinstance(func.value, ast.Name)
                and func.value.id == "self"
            ):
                continue
            if not node.args:
                continue
            name = node.args[0]
            if isinstance(name, ast.Constant) and isinstance(name.value, str):
                if not name.value.startswith("_"):
                    attrs.add(name.value)
    return attrs


def _literal_string_set(function_node: ast.FunctionDef, variable_name: str) -> set[str]:
    for node in ast.walk(function_node):
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == variable_name
            for target in node.targets
        ):
            continue
        if isinstance(node.value, ast.Set):
            return {
                item.value
                for item in node.value.elts
                if isinstance(item, ast.Constant) and isinstance(item.value, str)
            }
    return set()


def _forwarded_getattr_attrs(class_node: ast.ClassDef) -> set[str]:
    attrs = set()
    for node in class_node.body:
        if isinstance(node, ast.FunctionDef) and node.name == "__getattr__":
            attrs.update(_literal_string_set(node, "_whitelisted_attrs"))
    return attrs


def _capabilities(relative_path: str, class_names: str | tuple[str, ...]) -> set[str]:
    if isinstance(class_names, str):
        class_names = (class_names,)
    capabilities = set()
    for class_name in class_names:
        node = _class_node(relative_path, class_name)
        capabilities.update(
            _public_methods(node)
            | _public_properties(node)
            | _assigned_public_attrs(node)
            | _forwarded_getattr_attrs(node)
        )
    return capabilities


OBJECT_PROXY_PAIRS = (
    (
        "ModelPatcherProxy",
        "comfy/model_patcher.py",
        "ModelPatcher",
        "comfy/isolation/model_patcher_proxy.py",
        "ModelPatcherProxy",
        {
            "apply_model",
            "device",
            "get_non_dynamic_delegate",
            "get_operation_state",
            "get_ram_usage",
            "load_lora",
            "model_lowvram",
            "model_loaded_weight_memory",
            "model_mmap_residency",
            "wait_for_idle",
        },
    ),
    (
        "CLIPProxy",
        "comfy/sd.py",
        "CLIP",
        "comfy/isolation/clip_proxy.py",
        "CLIPProxy",
        {"get_ram_usage"},
    ),
    (
        "VAEProxy",
        "comfy/sd.py",
        "VAE",
        "comfy/isolation/vae_proxy.py",
        "VAEProxy",
        set(),
    ),
    (
        "ModelSamplingProxy",
        "comfy/model_sampling.py",
        (
            "EPS",
            "V_PREDICTION",
            "V_PREDICTION_DDPM",
            "EDM",
            "CONST",
            "X0",
            "IMG_TO_IMG",
            "IMG_TO_IMG_FLOW",
            "COSMOS_RFLOW",
            "ModelSamplingDiscrete",
            "ModelSamplingDiscreteEDM",
            "ModelSamplingContinuousEDM",
            "ModelSamplingContinuousV",
            "ModelSamplingDiscreteFlow",
            "StableCascadeSampling",
            "ModelSamplingFlux",
            "ModelSamplingCosmosRFlow",
        ),
        "comfy/isolation/model_sampling_proxy.py",
        "ModelSamplingProxy",
        set(),
    ),
)


@pytest.mark.parametrize(
    "name,target_path,target_class,proxy_path,proxy_class,allowed_extensions",
    OBJECT_PROXY_PAIRS,
)
def test_object_proxy_public_capabilities_match_current_targets(
    name,
    target_path,
    target_class,
    proxy_path,
    proxy_class,
    allowed_extensions,
):
    target_capabilities = _capabilities(target_path, target_class)
    proxy_capabilities = _capabilities(proxy_path, proxy_class)

    missing = sorted(target_capabilities - proxy_capabilities)
    stale = sorted(proxy_capabilities - target_capabilities - allowed_extensions)

    assert missing == [], f"{name} missing proxied capabilities: {missing}"
    assert stale == [], f"{name} stale proxy implementations: {stale}"


def _recording_proxy(proxy_cls):
    proxy = object.__new__(proxy_cls)
    proxy.calls = []

    def call_rpc(self, method_name, *args, **kwargs):
        self.calls.append((method_name, args, kwargs))
        return f"{method_name}-result"

    proxy._call_rpc = MethodType(call_rpc, proxy)
    return proxy


def test_model_patcher_new_public_forwarders_smoke():
    from comfy.isolation.model_patcher_proxy import ModelPatcherProxy

    proxy = _recording_proxy(ModelPatcherProxy)
    proxy._spawn_related_proxy = lambda instance_id: ("spawned", instance_id)

    assert proxy.get_clone_model_override() == "get_clone_model_override-result"
    assert proxy.deepclone_multigpu("cuda:1") == ("spawned", "deepclone_multigpu-result")
    proxy.set_model_middle_block_after_patch("patch")
    proxy.model_patches_call_function("cleanup", {"x": 1})
    assert proxy.loaded_ram_size() == "loaded_ram_size-result"
    assert proxy.get_callbacks("call", "key") == "get_callbacks-result"
    assert proxy.get_wrappers("wrap", "key") == "get_wrappers-result"
    proxy.patch_hooks("hooks")
    proxy.patch_cached_hook_weights({"k": "v"}, "k", "counter")
    proxy.patch_hook_weight_to_device("hooks", {"k": "v"}, "k", {"k": "w"}, "counter")
    assert proxy.model_state_dict_for_saving("model", "prefix") == "model_state_dict_for_saving-result"
    assert proxy.state_dict_for_saving("clip", "vae", "vision") == "state_dict_for_saving-result"

    assert [call[0] for call in proxy.calls] == [
        "get_clone_model_override",
        "deepclone_multigpu",
        "set_model_patch",
        "model_patches_call_function",
        "loaded_ram_size",
        "get_callbacks",
        "get_wrappers",
        "patch_hooks",
        "patch_cached_hook_weights",
        "patch_hook_weight_to_device",
        "model_state_dict_for_saving",
        "state_dict_for_saving",
    ]


def test_clip_new_public_forwarders_smoke():
    from comfy.isolation.clip_proxy import CLIPProxy

    proxy = _recording_proxy(CLIPProxy)

    assert proxy.add_hooks_to_dict({"pooled": True}) == "add_hooks_to_dict-result"
    assert proxy.state_dict_for_saving() == "state_dict_for_saving-result"
    assert proxy.generate({"tokens": []}, seed=1) == "generate-result"
    assert proxy.decode([1, 2, 3]) == "decode-result"

    assert [call[0] for call in proxy.calls] == [
        "add_hooks_to_dict",
        "state_dict_for_saving",
        "generate",
        "decode",
    ]


def test_vae_new_public_forwarders_smoke():
    from comfy.isolation.vae_proxy import VAEProxy

    proxy = _recording_proxy(VAEProxy)

    assert proxy.model_size() == "model_size-result"
    proxy.throw_exception_if_invalid()
    assert proxy.vae_encode_crop_pixels("pixels") == "vae_encode_crop_pixels-result"
    assert proxy.vae_output_dtype() == "vae_output_dtype-result"
    assert proxy.decode_tiled_("samples") == "decode_tiled_-result"
    assert proxy.decode_tiled_1d("samples") == "decode_tiled_1d-result"
    assert proxy.decode_tiled_3d("samples") == "decode_tiled_3d-result"
    assert proxy.encode_tiled_("pixels") == "encode_tiled_-result"
    assert proxy.encode_tiled_1d("samples") == "encode_tiled_1d-result"
    assert proxy.encode_tiled_3d("samples") == "encode_tiled_3d-result"
    assert proxy.spacial_compression_decode() == "spacial_compression_decode-result"
    assert proxy.spacial_compression_encode() == "spacial_compression_encode-result"
    assert proxy.temporal_compression_decode() == "temporal_compression_decode-result"

    assert [call[0] for call in proxy.calls] == [
        "model_size",
        "throw_exception_if_invalid",
        "vae_encode_crop_pixels",
        "vae_output_dtype",
        "decode_tiled_",
        "decode_tiled_1d",
        "decode_tiled_3d",
        "encode_tiled_",
        "encode_tiled_1d",
        "encode_tiled_3d",
        "spacial_compression_decode",
        "spacial_compression_encode",
        "temporal_compression_decode",
    ]


def test_model_sampling_new_public_forwarders_smoke():
    from comfy.isolation.model_sampling_proxy import ModelSamplingProxy

    proxy = _recording_proxy(ModelSamplingProxy)
    proxy._call = proxy._call_rpc

    assert proxy.cosine_s == "get_property-result"
    assert proxy.linear_end == "get_property-result"
    assert proxy.linear_start == "get_property-result"
    assert proxy.log_sigmas == "get_property-result"
    assert proxy.multiplier == "get_property-result"
    assert proxy.noise_scale == "get_property-result"
    assert proxy.num_timesteps == "get_property-result"
    assert proxy.shift == "get_property-result"
    assert proxy.zsnr == "get_property-result"
    assert proxy.set_parameters("shift") == "set_parameters-result"
    assert proxy.set_noise_scale(1.5) == "set_noise_scale-result"

    assert [call[0] for call in proxy.calls] == [
        "get_property",
        "get_property",
        "get_property",
        "get_property",
        "get_property",
        "get_property",
        "get_property",
        "get_property",
        "get_property",
        "set_parameters",
        "set_noise_scale",
    ]
