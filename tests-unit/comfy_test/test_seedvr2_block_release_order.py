from comfy.cli_args import args

args.cpu = True

import torch  # noqa: E402

import comfy.ldm.seedvr.model as seedvr_model  # noqa: E402
from comfy.ldm.seedvr.model import NaDiT  # noqa: E402


class _RecordingBlock:
    def __init__(self, block_index, events):
        self.block_index = block_index
        self.events = events
        self.device = "cuda"

    def to(self, device):
        target = str(device)
        if target == "meta":
            event = "offload" if self.device == "cpu" else "preoffload"
            self.events.append(f"{event} block {self.block_index}")
        else:
            self.events.append(f"enter block {self.block_index}")
        self.device = "cpu" if target != "meta" else "meta"
        return self

    def __call__(self, *, vid, txt, vid_shape, txt_shape, emb, cache):
        self.events.append(f"run block {self.block_index}")
        return vid, txt, vid_shape, txt_shape

class _SeedVR2StandIn:
    _seedvr2_call_block = NaDiT._seedvr2_call_block
    _seedvr2_run_blocks = NaDiT._seedvr2_run_blocks


def _make_standin(events):
    model = _SeedVR2StandIn()
    model.blocks = [_RecordingBlock(0, events), _RecordingBlock(1, events)]
    model._seedvr2_synchronize_after_block = lambda device: events.append("synchronize")
    return model


def _run_block_loop(model, transformer_options):
    vid = torch.zeros(1, 1)
    txt = torch.ones(1, 1)
    vid_shape = torch.tensor([[1]])
    txt_shape = torch.tensor([[1]])
    emb = torch.full((1, 1), 2.0)
    cache = seedvr_model.Cache()
    return model._seedvr2_run_blocks(
        vid,
        txt,
        vid_shape,
        txt_shape,
        emb,
        cache,
        transformer_options,
        block_release=True,
    )


def test_seedvr2_block_release_moves_one_block_at_a_time(monkeypatch):
    events = []
    model = _make_standin(events)
    monkeypatch.setattr(seedvr_model.comfy.model_management, "unet_offload_device", lambda: torch.device("meta"))
    monkeypatch.setattr(seedvr_model.comfy.model_management, "soft_empty_cache", lambda: events.append("soft_empty_cache"))

    _run_block_loop(model, {})

    assert events[:9] == [
        "preoffload block 0",
        "preoffload block 1",
        "synchronize",
        "soft_empty_cache",
        "enter block 0",
        "run block 0",
        "synchronize",
        "offload block 0",
        "soft_empty_cache",
    ]
    assert events[9:14] == [
        "enter block 1",
        "run block 1",
        "synchronize",
        "offload block 1",
        "soft_empty_cache",
    ]


def test_seedvr2_block_release_preserves_replacement_block_path(monkeypatch):
    events = []
    model = _make_standin(events)
    monkeypatch.setattr(seedvr_model.comfy.model_management, "unet_offload_device", lambda: torch.device("meta"))
    monkeypatch.setattr(seedvr_model.comfy.model_management, "soft_empty_cache", lambda: events.append("soft_empty_cache"))
    seen_original_keys = []

    def replacement(args, controls):
        out = controls["original_block"](args)
        seen_original_keys.append(tuple(out.keys()))
        return out

    transformer_options = {
        "patches_replace": {
            "dit": {
                ("block", 0): replacement,
            },
        },
    }

    vid, txt, vid_shape, txt_shape = _run_block_loop(model, transformer_options)

    assert seen_original_keys == [("vid", "txt", "vid_shape", "txt_shape")]
    assert vid.shape == (1, 1)
    assert txt.shape == (1, 1)
    assert vid_shape.tolist() == [[1]]
    assert txt_shape.tolist() == [[1]]
