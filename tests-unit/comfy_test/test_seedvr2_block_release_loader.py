from comfy.cli_args import args

args.cpu = True

import comfy.sd as sd  # noqa: E402


class _SeedVR2Config:
    unet_config = {"image_model": "seedvr2"}


class _OtherConfig:
    unet_config = {"image_model": "other"}


class _StaticPatcher:
    pass


class _DynamicPatcher:
    pass


def test_seedvr2_block_release_loader_uses_static_model_patcher(monkeypatch):
    monkeypatch.setattr(sd.comfy.model_patcher, "ModelPatcher", _StaticPatcher)
    monkeypatch.setattr(sd.comfy.model_patcher, "CoreModelPatcher", _DynamicPatcher)

    assert sd._select_model_patcher_for_diffusion_model(
        _SeedVR2Config(),
        {"seedvr2_block_release": True},
        False,
    ) is _StaticPatcher
    assert sd._select_model_patcher_for_diffusion_model(
        _SeedVR2Config(),
        {"transformer_options": {"seedvr2_block_release": True}},
        False,
    ) is _StaticPatcher
    assert sd._select_model_patcher_for_diffusion_model(
        _SeedVR2Config(),
        {},
        False,
    ) is _DynamicPatcher
    assert sd._select_model_patcher_for_diffusion_model(
        _OtherConfig(),
        {"seedvr2_block_release": True},
        False,
    ) is _DynamicPatcher
