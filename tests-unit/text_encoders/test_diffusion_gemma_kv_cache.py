import os
import sys
import unittest
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from comfy.cli_args import args
if not torch.cuda.is_available():
    args.cpu = True

from comfy.text_encoders.diffusion_gemma import DiffusionGenerate  # noqa: E402


class TestDiffusionGemmaKVCache(unittest.TestCase):
    def test_reserve_keeps_only_committed_prefix(self):
        generate = DiffusionGenerate()
        layers = [SimpleNamespace(sliding_window=4), SimpleNamespace(sliding_window=None)]
        generate.model = SimpleNamespace(decoder=SimpleNamespace(layers=layers))
        key = torch.arange(7).reshape(1, 1, 7, 1)
        value = key + 10
        past_key_values = [(key, value, 9, 5), (key[:, :, :5], value[:, :, :5], 5)]

        sliding, full = generate._reserve_kv_cache(past_key_values, reserve=2)

        self.assertEqual(sliding[0].shape, (1, 1, 5, 1))
        self.assertEqual(sliding[2:], (9, 3))
        self.assertTrue(torch.equal(sliding[0][:, :, :3], key[:, :, 2:5]))
        self.assertTrue(torch.equal(sliding[1][:, :, :3], value[:, :, 2:5]))
        self.assertEqual(full[0].shape, (1, 1, 7, 1))
        self.assertEqual(full[2:], (5, 5))
        self.assertTrue(torch.equal(full[0][:, :, :5], key[:, :, :5]))


if __name__ == "__main__":
    unittest.main()
