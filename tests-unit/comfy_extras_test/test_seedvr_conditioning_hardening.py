"""Regression tests for SeedVR2 conditioning model resolution and RoPE
frequency cast.
"""

import importlib
import sys
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn


_SENTINEL = object()


def _import_nodes_seedvr_isolated():
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

    mock_mm = MagicMock()
    mock_mm.xformers_enabled.return_value = False
    mock_mm.xformers_enabled_vae.return_value = False
    mock_mm.pytorch_attention_enabled.return_value = False
    mock_mm.pytorch_attention_enabled_vae.return_value = False
    mock_mm.sage_attention_enabled.return_value = False
    mock_mm.flash_attention_enabled.return_value = False
    torch_version_parts = torch.version.__version__.split(".")
    mock_mm.torch_version_numeric = (
        int(torch_version_parts[0]),
        int(torch_version_parts[1]),
    )
    mock_mm.WINDOWS = False
    mock_mm.is_intel_xpu.return_value = False
    sys.modules["comfy.model_management"] = mock_mm
    if comfy_pkg is None:
        import comfy as _comfy_pkg  # noqa: F401
        comfy_pkg = sys.modules.get("comfy")
    if comfy_pkg is not None:
        setattr(comfy_pkg, "model_management", mock_mm)
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
        if prior_nodes_seedvr_module is _SENTINEL:
            sys.modules.pop("comfy_extras.nodes_seedvr", None)
        else:
            sys.modules["comfy_extras.nodes_seedvr"] = prior_nodes_seedvr_module
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
    def __init__(
        self,
        n_blocks=3,
        zero_conditioning=False,
        conditioning_dtype=torch.float32,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([_Block() for _ in range(n_blocks)])
        if zero_conditioning:
            self.register_buffer(
                "positive_conditioning",
                torch.zeros((2, 4), dtype=conditioning_dtype),
            )
            self.register_buffer(
                "negative_conditioning",
                torch.zeros((3, 4), dtype=conditioning_dtype),
            )
        else:
            self.register_buffer(
                "positive_conditioning",
                torch.ones((2, 4), dtype=conditioning_dtype),
            )
            self.register_buffer(
                "negative_conditioning",
                torch.zeros((3, 4), dtype=conditioning_dtype),
            )


class _ModelInner:
    def __init__(self, diffusion_model):
        self.diffusion_model = diffusion_model


class _ModelPatcher:
    def __init__(self, diffusion_model):
        self.model = _ModelInner(diffusion_model)


def test_seedvr2_conditioning_schema_exposes_model_passthrough_output():
    nodes_seedvr, restore = _import_nodes_seedvr_isolated()
    try:
        schema = nodes_seedvr.SeedVR2Conditioning.define_schema()
        assert [input_item.id for input_item in schema.inputs] == [
            "model",
            "vae_conditioning",
        ]
        assert schema.inputs[1].display_name == "LATENT"
        assert [output.display_name for output in schema.outputs] == [
            "model",
            "positive",
            "negative",
            "latent",
        ]
    finally:
        restore()


def test_seedvr2_conditioning_returns_packed_input_latent_deterministically():
    nodes_seedvr, restore = _import_nodes_seedvr_isolated()
    try:
        diffusion_model = _DiffusionModel()
        patcher = _ModelPatcher(diffusion_model)
        samples = torch.arange(1, 25, dtype=torch.float32).reshape(1, 2, 3, 2, 2)
        vae_conditioning = {"samples": samples}

        _, first_positive, first_negative, first_latent = (
            nodes_seedvr.SeedVR2Conditioning.execute(
                patcher,
                vae_conditioning,
            )
        )
        _, second_positive, second_negative, second_latent = (
            nodes_seedvr.SeedVR2Conditioning.execute(
                patcher,
                vae_conditioning,
            )
        )

        expected_latent = samples.reshape(1, 6, 2, 2)
        channel_last = samples.movedim(1, -1).contiguous()
        expected_condition = torch.cat(
            [
                channel_last,
                torch.ones((*channel_last.shape[:-1], 1)),
            ],
            dim=-1,
        ).movedim(-1, 1).reshape(1, 9, 2, 2)

        assert torch.equal(first_latent["samples"], expected_latent)
        assert torch.equal(second_latent["samples"], expected_latent)
        assert torch.equal(
            first_positive[0][1]["condition"],
            expected_condition,
        )
        assert torch.equal(
            second_positive[0][1]["condition"],
            expected_condition,
        )
        assert torch.equal(
            first_negative[0][1]["condition"],
            expected_condition,
        )
        assert torch.equal(
            second_negative[0][1]["condition"],
            expected_condition,
        )
    finally:
        restore()


def test_resolve_seedvr2_diffusion_model_raises_runtime_error_with_specific_prefix():
    nodes_seedvr, restore = _import_nodes_seedvr_isolated()
    try:
        class _NoModelAttr:
            pass

        with pytest.raises(RuntimeError) as excinfo:
            nodes_seedvr._resolve_seedvr2_diffusion_model(_NoModelAttr())
        msg = str(excinfo.value)
        assert msg.startswith(nodes_seedvr._SEEDVR2_INVALID_MODEL_MSG_PREFIX)
        assert "no 'model' attribute" in msg

        class _ModelIsNone:
            def __init__(self):
                self.model = None

        with pytest.raises(RuntimeError) as excinfo:
            nodes_seedvr._resolve_seedvr2_diffusion_model(_ModelIsNone())
        msg = str(excinfo.value)
        assert msg.startswith(nodes_seedvr._SEEDVR2_INVALID_MODEL_MSG_PREFIX)
        assert "input.model is None" in msg

        class _NoDiffusionAttr:
            def __init__(self):
                self.model = object()

        with pytest.raises(RuntimeError) as excinfo:
            nodes_seedvr._resolve_seedvr2_diffusion_model(_NoDiffusionAttr())
        msg = str(excinfo.value)
        assert msg.startswith(nodes_seedvr._SEEDVR2_INVALID_MODEL_MSG_PREFIX)
        assert "no 'diffusion_model' attribute" in msg

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
    nodes_seedvr, restore = _import_nodes_seedvr_isolated()
    try:
        diffusion_model = _DiffusionModel()

        for module in diffusion_model.modules():
            if hasattr(module, "rope") and hasattr(module.rope, "freqs"):
                module.rope.freqs.data = module.rope.freqs.data.to(torch.float64)

        nodes_seedvr._apply_rope_freqs_float32_cast(diffusion_model)
        first_call_data_ids = []
        for module in diffusion_model.modules():
            if hasattr(module, "rope") and hasattr(module.rope, "freqs"):
                assert module.rope.freqs.data.dtype == torch.float32
                first_call_data_ids.append(id(module.rope.freqs.data))

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


def test_seedvr2_conditioning_fails_loud_on_zero_buffers():
    nodes_seedvr, restore = _import_nodes_seedvr_isolated()
    try:
        diffusion_model = _DiffusionModel(zero_conditioning=True)
        patcher = _ModelPatcher(diffusion_model)
        vae_conditioning = {"samples": torch.zeros((1, 2, 1, 1, 1))}

        with pytest.raises(RuntimeError) as excinfo:
            nodes_seedvr.SeedVR2Conditioning.execute(
                patcher, vae_conditioning,
            )

        message = str(excinfo.value)
        assert message.startswith(
            nodes_seedvr._SEEDVR2_INVALID_MODEL_MSG_PREFIX
        ), (
            "Fail-loud message must use the standard "
            "_SEEDVR2_INVALID_MODEL_MSG_PREFIX so callers/log scrapers "
            f"can match it. Got: {message!r}"
        )
        assert "positive_conditioning" in message
        assert "negative_conditioning" in message
    finally:
        restore()
