import json
import asyncio
import inspect

import torch

import comfy.supported_models
import comfy.model_management
import comfy.ldm.seedvr.model as seedvr_model
import comfy.ldm.modules.attention as attention_module
import execution as execution_module
import nodes


def test_seedvr2_fp16_manual_cast_only_for_bf16_device(monkeypatch):
    bf16_device = object()
    fp16_device = object()

    monkeypatch.setattr(
        comfy.supported_models.comfy.model_management,
        "should_use_bf16",
        lambda device=None: device is bf16_device,
    )

    bf16_config = comfy.supported_models.SeedVR2({"image_model": "seedvr2"})
    bf16_config.set_inference_dtype(torch.float16, None, device=bf16_device)
    assert bf16_config.manual_cast_dtype is torch.bfloat16

    fp16_config = comfy.supported_models.SeedVR2({"image_model": "seedvr2"})
    fp16_config.set_inference_dtype(torch.float16, None, device=fp16_device)
    assert fp16_config.manual_cast_dtype is None


def test_apply_rope1_partial_preserves_full_rotation_input_dtype(monkeypatch):
    def fake_apply_rope1(t, freqs_cis):
        return t.float() + 1.0

    monkeypatch.setattr(seedvr_model, "apply_rope1", fake_apply_rope1)

    t = torch.arange(8, dtype=torch.float16).reshape(1, 2, 4)
    freqs_cis = torch.zeros(1, 2, 2, 2)

    out = seedvr_model._apply_rope1_partial(t, freqs_cis)

    assert out.dtype is torch.float16
    torch.testing.assert_close(out, (t.float() + 1.0).to(torch.float16))


def test_apply_rope1_partial_preserves_partial_rotation_input_dtype(monkeypatch):
    def fake_apply_rope1(t, freqs_cis):
        return t.float() + 1.0

    monkeypatch.setattr(seedvr_model, "apply_rope1", fake_apply_rope1)

    t = torch.arange(12, dtype=torch.float16).reshape(1, 2, 6)
    freqs_cis = torch.zeros(1, 2, 2, 2)

    out = seedvr_model._apply_rope1_partial(t, freqs_cis)

    assert out.dtype is torch.float16
    torch.testing.assert_close(
        out[..., :4],
        (t[..., :4].float() + 1.0).to(torch.float16),
    )
    torch.testing.assert_close(out[..., 4:], t[..., 4:])


def test_var_attention_boundary_telemetry_emits_contract_fields(monkeypatch, tmp_path):
    telemetry_path = tmp_path / "telemetry.jsonl"
    monkeypatch.setattr(attention_module, "_VAR_ATTENTION_TELEMETRY_CALL_INDEX", 0)
    monkeypatch.setenv("COMFY_VAR_ATTENTION_TELEMETRY_PATH", str(telemetry_path))
    monkeypatch.setenv("COMFY_VAR_ATTENTION_TELEMETRY_WORKFLOW", "workflow.json")
    monkeypatch.setenv("COMFY_VAR_ATTENTION_TELEMETRY_FLOW_ID", "native")
    monkeypatch.setenv("COMFY_VAR_ATTENTION_TELEMETRY_BLOCK_INDEX", "7")

    def fake_attention(q, k, v, heads, cu_seqlens_q, cu_seqlens_k, *args, **kwargs):
        return q + k + v

    wrapped = attention_module._instrument_var_attention(fake_attention)
    q = torch.ones(2, 1, 2)
    k = torch.full((2, 1, 2), 2.0)
    v = torch.full((2, 1, 2), 3.0)
    cu_seqlens = torch.tensor([0, 2])

    out = wrapped(
        q,
        k,
        v,
        1,
        cu_seqlens,
        cu_seqlens,
        skip_reshape=True,
        skip_output_reshape=True,
    )

    torch.testing.assert_close(out, torch.full((2, 1, 2), 6.0))
    rows = [json.loads(line) for line in telemetry_path.read_text().splitlines()]
    assert [row["event"] for row in rows] == ["entry", "exit"]
    assert rows[-1]["status"] == "pass"
    assert set(attention_module._VAR_ATTENTION_TELEMETRY_FIELDS).issubset(rows[-1])
    assert rows[-1]["workflow"] == "workflow.json"
    assert rows[-1]["flow_id"] == "native"
    assert rows[-1]["block_index"] == "7"
    assert rows[-1]["call_index"] == 0
    assert rows[-1]["attention_backend"] == "fake_attention"


def test_var_attention_boundary_telemetry_records_oom(monkeypatch, tmp_path):
    telemetry_path = tmp_path / "telemetry.jsonl"
    monkeypatch.setattr(attention_module, "_VAR_ATTENTION_TELEMETRY_CALL_INDEX", 0)
    monkeypatch.setenv("COMFY_VAR_ATTENTION_TELEMETRY_PATH", str(telemetry_path))
    monkeypatch.setenv("COMFY_VAR_ATTENTION_TELEMETRY_BLOCK_INDEX", "2")

    def fake_attention(q, k, v, heads, cu_seqlens_q, cu_seqlens_k, *args, **kwargs):
        raise torch.cuda.OutOfMemoryError("Tried to allocate 1.24 GiB")

    wrapped = attention_module._instrument_var_attention(fake_attention)
    q = torch.ones(2, 1, 2)
    cu_seqlens = torch.tensor([0, 2])

    try:
        wrapped(q, q, q, 1, cu_seqlens, cu_seqlens)
    except torch.cuda.OutOfMemoryError:
        pass
    else:
        raise AssertionError("expected OutOfMemoryError")

    rows = [json.loads(line) for line in telemetry_path.read_text().splitlines()]
    assert [row["event"] for row in rows] == ["entry", "exception"]
    assert rows[-1]["status"] == "oom"
    assert rows[-1]["requested_allocation_gib"] == 1.24
    assert rows[-1]["block_index"] == "2"
    assert rows[-1]["call_index"] == 0


def test_var_attention_boundary_telemetry_does_not_synthesize_block_index(monkeypatch, tmp_path):
    telemetry_path = tmp_path / "telemetry.jsonl"
    monkeypatch.setattr(attention_module, "_VAR_ATTENTION_TELEMETRY_CALL_INDEX", 0)
    monkeypatch.setenv("COMFY_VAR_ATTENTION_TELEMETRY_PATH", str(telemetry_path))

    def fake_attention(q, k, v, heads, cu_seqlens_q, cu_seqlens_k, *args, **kwargs):
        return q

    wrapped = attention_module._instrument_var_attention(fake_attention)
    q = torch.ones(2, 1, 2)
    cu_seqlens = torch.tensor([0, 2])
    wrapped(q, q, q, 1, cu_seqlens, cu_seqlens)

    rows = [json.loads(line) for line in telemetry_path.read_text().splitlines()]
    assert rows[-1]["block_index"] is None
    assert isinstance(rows[-1]["call_index"], int)


def test_seedvr2_attention_caller_shape_is_unwrapped():
    source = inspect.getsource(seedvr_model.NaSwinAttention.forward)
    assert "instrument_var_attention_argument" not in source
    assert "concat_win = instrument" not in source
    assert "q=concat_win(vid_q, txt_q)" in source
    assert "k=concat_win(vid_k, txt_k)" in source
    assert "v=concat_win(vid_v, txt_v)" in source


def test_stop_after_dit_exception_terminates_prompt_before_downstream(monkeypatch):
    class StopAfterDiTNode:
        RETURN_TYPES = ("IMAGE",)
        FUNCTION = "run"
        CATEGORY = "test"

        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {}}

        def run(self):
            raise comfy.model_management.StopAfterDiTProcessingException()

    class DownstreamOutputNode:
        RETURN_TYPES = ()
        FUNCTION = "run"
        OUTPUT_NODE = True
        CATEGORY = "test"

        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {"image": ("IMAGE",)}}

        def run(self, image):
            raise AssertionError("downstream node executed")

    class Server:
        client_id = None
        messages = []

        def send_sync(self, event, data, client_id):
            self.messages.append((event, data, client_id))

    async def run_prompt():
        monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "StopAfterDiTNode", StopAfterDiTNode)
        monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "DownstreamOutputNode", DownstreamOutputNode)
        prompt = {
            "1": {"class_type": "StopAfterDiTNode", "inputs": {}},
            "2": {"class_type": "DownstreamOutputNode", "inputs": {"image": ["1", 0]}},
        }
        executor = execution_module.PromptExecutor(Server(), cache_args={"ram": 0})
        await executor.execute_async(prompt, "prompt-stop-after-dit", {}, ["2"])
        return executor

    executor = asyncio.run(run_prompt())
    assert executor.success is True
    assert [message[0] for message in executor.status_messages] == [
        "execution_start",
        "execution_cached",
        "execution_success",
    ]
    assert executor.history_result == {"outputs": {}, "meta": {}}
