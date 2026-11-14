"""Stage1 feature-cache integration without downloading or training 8B."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import json
import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader

from train_maestro_8b.data.dataset import RiskDataset, RiskCollator
from train_maestro_8b.data.features import (
    CachedRiskDataset, collate_cached, write_feature_cache,
)
from train_maestro_8b.risk.model import WindowRiskHead
from train_maestro_8b.data.contracts import VERSION


class FakeFastTokenizer:
    is_fast = True
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [ord(c) % 80 + 3 for c in text]

    def __call__(self, text, add_special_tokens=False,
                 return_offsets_mapping=False, truncation=False):
        result = {"input_ids": self.encode(text)}
        if return_offsets_mapping:
            result["offset_mapping"] = [(i, i+1) for i in range(len(text))]
        return result


class FrozenStub(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(128, 8)

    def forward(self, input_ids, attention_mask, use_cache, return_dict):
        return SimpleNamespace(last_hidden_state=self.emb(input_ids))


class CacheTests(unittest.TestCase):
    def test_cache_round_trip_and_risk_head_only_backprop(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            data_file = root / "records.jsonl"
            record = {
                "schema": VERSION,
                "prompt": "abcdefghi",
                "reports": [
                    {"report_id":"a", "start":0, "end":4,
                     "risk":1, "mask":1},
                    {"report_id":"b", "start":4, "end":9,
                     "risk":0, "mask":1},
                ],
            }
            data_file.write_text(json.dumps(record) + "\n", encoding="utf-8")
            tokenizer = FakeFastTokenizer()
            ds = RiskDataset(data_file, tokenizer, max_length=32)
            loader = DataLoader(
                ds, batch_size=1, collate_fn=RiskCollator(
                    tokenizer, max_length=32, window_size=3, window_stride=2,
                ),
            )
            backbone = FrozenStub().eval().requires_grad_(False)
            cache_dir = root / "cache"
            info = write_feature_cache(
                backbone, loader, cache_dir,
                model_name="frozen-stub", window_size=3,
                window_stride=2, max_length=32, shard_size=1,
            )
            self.assertEqual(info["examples"], 1)
            cached = CachedRiskDataset(cache_dir)
            batch = collate_cached([cached[0]])
            self.assertEqual(batch["window_features"].shape[-1], 8)
            head = WindowRiskHead(8, head_size=12, top_k=2)
            result = head(**batch)
            result["loss"].backward()
            self.assertIsNone(backbone.emb.weight.grad)
            self.assertGreater(head.risk_head[-1].weight.grad.abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
