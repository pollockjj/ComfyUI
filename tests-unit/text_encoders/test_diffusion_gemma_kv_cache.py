import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from comfy.cli_args import args
if not torch.cuda.is_available():
    args.cpu = True

from comfy.text_encoders.diffusion_gemma import (  # noqa: E402
    DiffusionGemmaAttention,
    DiffusionGemmaConfig,
    DiffusionGemmaModel,
    DiffusionGenerate,
)


class TestDiffusionGemmaKVCache(unittest.TestCase):
    def test_unmasked_decoder_uses_native_gqa(self):
        config = DiffusionGemmaConfig(hidden_size=16, num_attention_heads=2)
        attention = DiffusionGemmaAttention(
            config, head_dim=8, num_kv_heads=1, has_v_proj=True,
            dtype=torch.float32, ops=torch.nn,
        ).eval()
        calls = []

        def capture(q, k, v, heads, **kwargs):
            calls.append((k.shape[1], kwargs.get("enable_gqa", False)))
            return torch.zeros(q.shape[0], q.shape[2], q.shape[1] * q.shape[3])

        hidden = torch.zeros(1, 2, 16)
        past = (torch.zeros(1, 1, 3, 8), torch.zeros(1, 1, 3, 8), 3)
        freqs = torch.ones(1, 1, 2, 8), torch.zeros(1, 1, 2, 8)
        with patch(
            "comfy.text_encoders.diffusion_gemma.optimized_attention_for_device",
            return_value=capture,
        ):
            attention(hidden, freqs_cis=freqs, past_key_value=past,
                      sliding_window=4, update_cache=False)
            attention(hidden, attention_mask=torch.zeros(1, 1, 2, 5),
                      freqs_cis=freqs, past_key_value=past,
                      sliding_window=4, update_cache=False)

        self.assertEqual(calls, [(1, True), (2, False)])

    def test_reserved_tail_matches_legacy_attention(self):
        config = DiffusionGemmaConfig(hidden_size=16, num_attention_heads=2)
        prefix = torch.arange(48, dtype=torch.float32).reshape(1, 3, 16) / 50
        canvas = torch.arange(32, dtype=torch.float32).reshape(1, 2, 16) / 40

        def freqs(length):
            return torch.ones(1, 1, length, 8), torch.zeros(1, 1, length, 8)

        for sliding_window in (None, 4):
            with self.subTest(sliding_window=sliding_window):
                attention = DiffusionGemmaAttention(
                    config, head_dim=8, num_kv_heads=1, has_v_proj=True,
                    dtype=torch.float32, ops=torch.nn,
                ).eval()
                with torch.no_grad():
                    for i, parameter in enumerate(attention.parameters()):
                        parameter.fill_((i + 1) / 100)
                _, legacy = attention(prefix, freqs_cis=freqs(3), past_key_value=(),
                                      sliding_window=sliding_window, update_cache=True)
                layer = SimpleNamespace(sliding_window=sliding_window)
                generate = DiffusionGenerate()
                generate.model = SimpleNamespace(decoder=SimpleNamespace(layers=[layer]))
                reserved = generate._reserve_kv_cache([legacy], 2)[0]
                committed = (reserved[0][:, :, :3].clone(), reserved[1][:, :, :3].clone())
                pointers = reserved[0].data_ptr(), reserved[1].data_ptr()

                legacy_out, _ = attention(canvas, freqs_cis=freqs(2), past_key_value=legacy,
                                          sliding_window=sliding_window, update_cache=False)
                reserved_out, _ = attention(canvas, freqs_cis=freqs(2), past_key_value=reserved,
                                            sliding_window=sliding_window, update_cache=False)
                self.assertTrue(torch.equal(reserved_out, legacy_out))
                self.assertEqual(reserved[2:], (3, 3))
                self.assertEqual(pointers, (reserved[0].data_ptr(), reserved[1].data_ptr()))
                self.assertTrue(torch.equal(reserved[0][:, :, :3], committed[0]))
                self.assertTrue(torch.equal(reserved[1][:, :, :3], committed[1]))

                legacy_out, legacy_next = attention(canvas, freqs_cis=freqs(2), past_key_value=legacy,
                                                     sliding_window=sliding_window, update_cache=True)
                reserved_out, reserved_next = attention(canvas, freqs_cis=freqs(2), past_key_value=reserved,
                                                         sliding_window=sliding_window, update_cache=True)
                self.assertTrue(torch.equal(reserved_out, legacy_out))
                compacted = generate._reserve_kv_cache([reserved_next], 2)[0]
                self.assertEqual(compacted[2:], (5, legacy_next[0].shape[2]))
                self.assertTrue(torch.equal(compacted[0][:, :, :compacted[3]], legacy_next[0]))
                self.assertTrue(torch.equal(compacted[1][:, :, :compacted[3]], legacy_next[1]))
                lengths = DiffusionGemmaModel._cached_kv_lens(
                    SimpleNamespace(decoder=SimpleNamespace(layers=[layer])), [compacted]
                )
                self.assertEqual(lengths, (0, 3) if sliding_window else (5, 0))


if __name__ == "__main__":
    unittest.main()
