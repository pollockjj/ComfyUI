from comfy.cli_args import args
import torch

if not torch.cuda.is_available():
    args.cpu = True

import ast  # noqa: E402
import inspect  # noqa: E402
import importlib  # noqa: E402

from torch import nn  # noqa: E402

import comfy  # noqa: E402

_model_management = importlib.import_module("comfy.model_management")

from comfy.ldm.seedvr import model as seedvr_model  # noqa: E402
from comfy.ldm.seedvr.model import MMModule  # noqa: E402


def test_no_get_torch_device_in_forward_methods():
    src = inspect.getsource(seedvr_model)
    tree = ast.parse(src)
    offenders = []
    for cls in ast.walk(tree):
        if not isinstance(cls, ast.ClassDef):
            continue
        for item in cls.body:
            if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if item.name != "forward":
                continue
            for node in ast.walk(item):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                if isinstance(func, ast.Attribute) and func.attr == "get_torch_device":
                    offenders.append((item.lineno, node.lineno))
                    break
                if isinstance(func, ast.Name) and func.id == "get_torch_device":
                    offenders.append((item.lineno, node.lineno))
                    break
    assert not offenders, f"found: {offenders}"


def test_mmmodule_forward_succeeds_without_get_torch_device_lookup(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("MMModule.forward called get_torch_device()")

    monkeypatch.setattr(comfy.model_management, "get_torch_device", boom)

    in_features, out_features = 16, 8
    mm = MMModule(nn.Linear, in_features, out_features, shared_weights=False)

    vid = torch.randn(4, in_features)
    txt = torch.randn(3, in_features)

    vid_out, txt_out = mm(vid, txt)

    assert vid_out.shape == (4, out_features)
    assert txt_out.shape == (3, out_features)
