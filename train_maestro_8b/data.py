"""Datasets and collators for MAESTRO Main-Agent training."""
from __future__ import annotations

from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset

from orchestra.modes import MODE_TO_INDEX
from train_maestro_8b.schema import (
    ORCH_TOKEN,
    iter_jsonl,
    macro_vector,
    validate_critic_record,
    validate_stage1_record,
)


class Stage1Dataset(Dataset):
    def __init__(self, path: str) -> None:
        self.items = [validate_stage1_record(x) for x in iter_jsonl(path)]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.items[index]


class CriticDataset(Dataset):
    def __init__(self, path: str) -> None:
        self.items = [validate_critic_record(x) for x in iter_jsonl(path)]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.items[index]


def _ensure_orch_suffix(text: str) -> str:
    text = str(text).rstrip()
    if text.endswith(ORCH_TOKEN):
        return text
    return f"{text}\n\n{ORCH_TOKEN}"


class Stage1Collator:
    """Mask the O1 prompt and supervise only the O1-compatible Main JSON."""

    def __init__(self, tokenizer, *, max_length: int = 16384) -> None:
        self.tokenizer = tokenizer
        self.max_length = int(max_length)

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        rows = []
        macro_labels = []
        for item in batch:
            prefix = _ensure_orch_suffix(item["input"]) + "\n"
            target = item["target_json"].strip()
            full = prefix + target
            encoded = self.tokenizer(
                full,
                add_special_tokens=True,
                truncation=True,
                max_length=self.max_length,
            )
            prefix_ids = self.tokenizer(
                prefix,
                add_special_tokens=True,
                truncation=True,
                max_length=self.max_length,
            )["input_ids"]
            labels = list(encoded["input_ids"])
            masked = min(len(prefix_ids), len(labels))
            labels[:masked] = [-100] * masked
            encoded["labels"] = labels
            rows.append(encoded)
            macro_labels.append(MODE_TO_INDEX[item["macro_label"]])

        label_rows = [row.pop("labels") for row in rows]
        padded = self.tokenizer.pad(
            rows,
            padding=True,
            return_tensors="pt",
        )
        max_len = int(padded["input_ids"].size(1))
        padded["labels"] = torch.tensor(
            [
                labels + [-100] * (max_len - len(labels))
                for labels in label_rows
            ],
            dtype=torch.long,
        )
        padded["macro_labels"] = torch.tensor(
            macro_labels,
            dtype=torch.long,
        )
        return padded


class CriticCollator:
    def __init__(self, tokenizer, *, max_length: int = 16384) -> None:
        self.tokenizer = tokenizer
        self.max_length = int(max_length)

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        texts = [_ensure_orch_suffix(item["input"]) for item in batch]
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        vectors = [macro_vector(item["macro_values"]) for item in batch]
        encoded["critic_targets"] = torch.tensor(
            [item[0] for item in vectors],
            dtype=torch.float32,
        )
        encoded["critic_mask"] = torch.tensor(
            [item[1] for item in vectors],
            dtype=torch.float32,
        )
        return encoded


__all__ = [
    "Stage1Dataset",
    "CriticDataset",
    "Stage1Collator",
    "CriticCollator",
]
