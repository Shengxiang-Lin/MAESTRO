"""Memory-efficient risk dataset and exact token-offset collation."""
from __future__ import annotations
import json
from pathlib import Path
from torch.utils.data import Dataset
from .risk_schema import validate
from .risk_windows import windows, span_to_tokens

class RiskDataset(Dataset):
    """Indexes byte offsets, not all 16K-token prompts, in RAM.

    Filter overlong examples before training. Never silently drop early
    Sub-Agent reports, or the evidence-history labels would be corrupted.
    """
    def __init__(self, path, tokenizer, max_length=16384):
        self.path = Path(path)
        self.offsets = []
        self.overlong = 0
        with self.path.open("rb") as f:
            while True:
                pos = f.tell()
                line = f.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                row = validate(json.loads(line))
                length = len(tokenizer.encode(row["prompt"], add_special_tokens=False))
                if length > max_length:
                    self.overlong += 1
                else:
                    self.offsets.append(pos)
        if not self.offsets:
            raise ValueError("no usable risk samples")

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, idx):
        with self.path.open("rb") as f:
            f.seek(self.offsets[idx])
            return validate(json.loads(f.readline()))


class RiskCollator:
    def __init__(self, tokenizer, max_length=16384, window_size=128,
                 window_stride=96):
        if not tokenizer.is_fast:
            raise ValueError("fast tokenizer required for character offsets")
        if window_size < 1 or not 1 <= window_stride <= window_size:
            raise ValueError("invalid window settings")
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.window_size = window_size
        self.window_stride = window_stride

    def __call__(self, batch):
        import torch
        built = []
        for row in batch:
            enc = self.tokenizer(row["prompt"], add_special_tokens=False,
                                 return_offsets_mapping=True, truncation=False)
            ids = enc["input_ids"]
            if len(ids) > self.max_length:
                raise ValueError("overlong example: refusing silent truncation")
            all_windows = []
            for rid, report in enumerate(row["reports"]):
                start, end = span_to_tokens(
                    enc["offset_mapping"], report["start"], report["end"]
                )
                for left, right in windows(
                    start, end, self.window_size, self.window_stride
                ):
                    all_windows.append((left, right, rid))
            built.append((ids, all_windows, row["reports"]))
        bsz = len(built)
        length = max(len(ids) for ids, _, _ in built)
        nwin = max(len(w) for _, w, _ in built)
        nrep = max(len(r) for _, _, r in built)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            raise ValueError("tokenizer must have pad_token_id")
        ids = torch.full((bsz, length), pad_id, dtype=torch.long)
        attention = torch.zeros((bsz, length), dtype=torch.long)
        starts = torch.zeros((bsz, nwin), dtype=torch.long)
        ends = torch.zeros_like(starts)
        owners = torch.full((bsz, nwin), -1, dtype=torch.long)
        labels = torch.zeros((bsz, nrep), dtype=torch.float32)
        masks = torch.zeros_like(labels)
        for b, (tokens, parts, reports) in enumerate(built):
            ids[b, :len(tokens)] = torch.tensor(tokens, dtype=torch.long)
            attention[b, :len(tokens)] = 1
            for j, (left, right, owner) in enumerate(parts):
                starts[b, j], ends[b, j], owners[b, j] = left, right, owner
            for i, report in enumerate(reports):
                labels[b, i] = report["risk"]
                masks[b, i] = report["mask"]
        return dict(input_ids=ids, attention_mask=attention,
                    window_starts=starts, window_ends=ends,
                    window_owner=owners, report_labels=labels,
                    report_mask=masks)
