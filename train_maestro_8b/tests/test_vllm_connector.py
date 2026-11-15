"""Regression test for a RiskKV prefix ending inside a vLLM page."""
from importlib.util import find_spec
import unittest

import torch


@unittest.skipUnless(find_spec("vllm"), "vLLM is optional")
class VLLMRiskKVTests(unittest.TestCase):
    def test_packed_k_and_edited_v_write_only_prefix_slots(self):
        from train_maestro_8b.policy.vllm_connector import write_triton_kv

        page_size = 16
        paged = torch.full((3, 2, page_size, 6), -1.)
        values = torch.arange(19 * 2 * 2 * 3, dtype=torch.float32).view(19, 2, 2, 3)
        write_triton_kv(paged, values, [2, 0], page_size)

        for token in range(len(values)):
            page = 2 if token < page_size else 0
            slot = paged[page, :, token % page_size]
            torch.testing.assert_close(slot[:, :3], values[token, 0])
            torch.testing.assert_close(slot[:, 3:], values[token, 1])
        self.assertTrue(torch.all(paged[0, :, 3:] == -1))
        self.assertTrue(torch.all(paged[1] == -1))
