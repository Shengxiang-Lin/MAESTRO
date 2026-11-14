"""Extract frozen Qwen hidden states; optionally cache only compact windows."""
from __future__ import annotations

import json
from pathlib import Path
import torch
from torch.utils.data import Dataset

from .risk_model import extract_window_features


def create_frozen_backbone(model_name: str, *, dtype=torch.bfloat16,
                           load_in_4bit: bool = False, device=None):
    from transformers import AutoModel
    kwargs = {"dtype": dtype, "attn_implementation": "sdpa"}
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        if not torch.cuda.is_available():
            raise ValueError("4-bit feature extraction requires CUDA")
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True,
        )
        kwargs["device_map"] = {"": str(device or "cuda:0")}
    model = AutoModel.from_pretrained(model_name, **kwargs)
    if not load_in_4bit and device is not None:
        model.to(device)
    model.requires_grad_(False)
    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    return model


@torch.no_grad()
def features_from_batch(backbone, batch):
    output = backbone(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        use_cache=False, return_dict=True,
    )
    windows, global_ = extract_window_features(
        output.last_hidden_state, batch["attention_mask"],
        batch["window_starts"], batch["window_ends"], batch["window_owner"],
    )
    return {
        "window_features": windows.detach(),
        "global_feature": global_.detach(),
        "window_owner": batch["window_owner"],
        "report_labels": batch["report_labels"],
        "report_mask": batch["report_mask"],
    }


class CachedRiskDataset(Dataset):
    def __init__(self, folder):
        self.folder = Path(folder)
        manifest = json.loads((self.folder / "manifest.json").read_text())
        if manifest.get("schema") != "maestro.risk_features.v1":
            raise ValueError("wrong risk feature-cache version")
        self.entries = []
        for shard, count in manifest["shards"]:
            self.entries.extend((shard, j) for j in range(count))
        self._last_shard = None
        self._rows = None
        self.hidden_size = int(manifest["hidden_size"])
        self.model_name = str(manifest["model"])
        self.window_size = int(manifest["window_size"])
        self.window_stride = int(manifest["window_stride"])
        if not self.entries:
            raise ValueError("empty feature cache")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        shard, item = self.entries[idx]
        if self._last_shard != shard:
            # Safe tensor-only deserialize; no arbitrary pickle class loading.
            self._rows = torch.load(self.folder / shard, map_location="cpu", weights_only=True)
            self._last_shard = shard
        return self._rows[item]


def collate_cached(rows):
    bsz = len(rows)
    nwin = max(r["window_features"].shape[0] for r in rows)
    nrep = max(r["report_labels"].shape[0] for r in rows)
    dim = rows[0]["window_features"].shape[-1]
    features = torch.zeros((bsz, nwin, dim), dtype=torch.float32)
    global_ = torch.stack([r["global_feature"].float() for r in rows])
    owners = torch.full((bsz, nwin), -1, dtype=torch.long)
    labels = torch.zeros((bsz, nrep), dtype=torch.float32)
    masks = torch.zeros_like(labels)
    for i, row in enumerate(rows):
        w = row["window_features"].shape[0]
        n = row["report_labels"].shape[0]
        features[i, :w] = row["window_features"].float()
        owners[i, :w] = row["window_owner"]
        labels[i, :n] = row["report_labels"]
        masks[i, :n] = row["report_mask"]
    return {
        "window_features": features, "global_feature": global_,
        "window_owner": owners, "report_labels": labels, "report_mask": masks,
    }


def write_feature_cache(backbone, loader, out_dir, *, model_name,
                        window_size, window_stride, max_length,
                        shard_size=128):
    from tqdm import tqdm
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    shards = []
    chunk = []
    dim = None
    device = next(backbone.parameters()).device
    for batch in tqdm(loader, desc="Frozen 8B feature extraction"):
        features = features_from_batch(backbone, {
            k: v.to(device) for k, v in batch.items()
        })
        for i in range(features["window_owner"].shape[0]):
            valid = features["window_owner"][i] >= 0
            nrep = int((features["report_mask"][i].shape[0]))
            record = {
                "window_features": features["window_features"][i, valid].cpu().to(torch.bfloat16),
                "global_feature": features["global_feature"][i].cpu().to(torch.bfloat16),
                "window_owner": features["window_owner"][i, valid].cpu().long(),
                "report_labels": features["report_labels"][i, :nrep].cpu().float(),
                "report_mask": features["report_mask"][i, :nrep].cpu().float(),
            }
            if dim is None:
                dim = record["window_features"].shape[-1]
            chunk.append(record)
            if len(chunk) >= shard_size:
                filename = f"shard-{len(shards):05d}.pt"
                torch.save(chunk, out / filename)
                shards.append([filename, len(chunk)])
                chunk = []
    if chunk:
        filename = f"shard-{len(shards):05d}.pt"
        torch.save(chunk, out / filename)
        shards.append([filename, len(chunk)])
    manifest = {
        "schema": "maestro.risk_features.v1",
        "model": model_name, "hidden_size": dim,
        "window_size": window_size, "window_stride": window_stride,
        "max_length": max_length, "shards": shards,
        "examples": sum(x[1] for x in shards),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
