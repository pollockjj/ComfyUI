import importlib
import sys
import types

import torch

import comfy.utils


def test_tiled_scale_multidim_multigpu_clips_edge_tiles(monkeypatch):
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)

    scale = 1.1

    def upscale(a):
        return torch.ones((a.shape[0], 1, round(a.shape[-1] * scale)), dtype=a.dtype, device=a.device)

    samples = torch.ones((1, 1, 11))
    devices = [torch.device("cpu:0"), torch.device("cpu:1")]

    actual = comfy.utils.tiled_scale_multidim_multigpu(
        samples,
        {device: upscale for device in devices},
        tile=(5,),
        overlap=2,
        upscale_amount=scale,
        out_channels=1,
        output_device="cpu",
    )
    expected = comfy.utils.tiled_scale_multidim(
        samples,
        upscale,
        tile=(5,),
        overlap=2,
        upscale_amount=scale,
        out_channels=1,
        output_device="cpu",
    )

    assert actual.shape == expected.shape == (1, 1, 12)
    torch.testing.assert_close(actual, expected)


def test_upscale_model_deepclone_does_not_copy_existing_clone_graph(monkeypatch):
    class FakeModel:
        def __init__(self):
            self.param = torch.nn.Parameter(torch.ones(1))

        def eval(self):
            return self

        def parameters(self):
            return [self.param]

    class FakeDescriptor:
        def __init__(self):
            self.model = FakeModel()
            self.device = None

        def to(self, device):
            self.device = device
            return self

    first_device = torch.device("cpu:0")
    second_device = torch.device("cpu:1")
    existing_clone = FakeDescriptor()
    source = FakeDescriptor()
    source.multigpu_clones = {first_device: existing_clone}
    fake_model_management = types.ModuleType("comfy.model_management")
    fake_model_management.get_all_torch_devices = lambda exclude_current=True: [first_device, second_device]
    monkeypatch.setitem(sys.modules, "comfy.model_management", fake_model_management)
    import comfy
    monkeypatch.setattr(comfy, "model_management", fake_model_management, raising=False)
    import comfy.multigpu
    importlib.reload(comfy.multigpu)

    cloned = comfy.multigpu.create_upscale_model_multigpu_deepclones(source, max_gpus=3)

    assert cloned is not source
    assert cloned.multigpu_clones[first_device] is existing_clone
    assert second_device in cloned.multigpu_clones
    assert not hasattr(cloned.multigpu_clones[second_device], "multigpu_clones")
    assert cloned.multigpu_clones[second_device].device == "cpu"
    assert not cloned.multigpu_clones[second_device].model.param.requires_grad
