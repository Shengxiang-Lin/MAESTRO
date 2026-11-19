"""A-Trust-inspired white-box attention trust scoring (online O1 adapter).

Requires a local Hugging Face model with eager attention access and calibrated
classifiers trained with train.py. No text-only surrogate is used.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import numpy as np

DIMENSIONS = ("factuality", "logical_consistency", "relevance",
              "completeness", "clarity", "source_support")


@lru_cache(maxsize=1)
def load_whitebox(model_name: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto",
        attn_implementation="eager", trust_remote_code=False)
    model.eval()
    return tokenizer, model


def attention_features(report: dict, context: list[dict], model_name: str):
    """Per-layer mean and std attention to current incoming report tokens.

    Fixed extraction template is used for both fitting and inference.
    The query token attends to the report; its attention cannot see oracle
    fields or other hidden MAESTRO state.
    """
    import torch
    tokenizer, model = load_whitebox(model_name)
    background = "\n".join(
        str(x["id"]) + ": " + x["result"][:600] for x in context
        if x["id"] != report["id"])
    prefix = ("Previous delivered reports:\n" + background[-2800:] +
              "\nCurrent Sub-Agent task:\n" + report["task"][:500] +
              "\nCurrent Sub-Agent message:\n")
    content = report["result"][:1500] + "\n" + report["summary"][:400]
    suffix = "\nAssess reliability of the current message."
    # Exact token boundary from separately tokenized segments.
    a = tokenizer(prefix, add_special_tokens=False).input_ids
    b = tokenizer(content, add_special_tokens=False).input_ids
    c = tokenizer(suffix, add_special_tokens=False).input_ids
    if not b:
        raise ValueError("cannot extract attention from empty message")
    max_len = min(int(getattr(model.config, "max_position_embeddings", 4096)),
                  8192)
    while len(a) + len(b) + len(c) > max_len and a:
        a = a[1:]
    if len(a) + len(b) + len(c) > max_len:
        raise ValueError("message exceeds model context window")
    ids = a + b + c
    device = model.get_input_embeddings().weight.device
    with torch.no_grad():
        result = model(input_ids=torch.tensor([ids], device=device),
                       output_attentions=True, use_cache=False)
    if not result.attentions:
        raise RuntimeError("Model did not expose attentions; requires eager attention")
    values = []
    for att in result.attentions:
        if att is None:
            raise RuntimeError("A-Trust attention cannot be accessed for this backend")
        # [batch, heads, queries, keys]; last-token attention to message span.
        layer = att[0, :, -1, len(a):len(a)+len(b)].float()
        per_head = layer.sum(-1).cpu().numpy()
        values.extend([float(per_head.mean()), float(per_head.std())])
    return np.asarray(values, dtype=np.float32)


@lru_cache(maxsize=3)
def load_classifiers(directory: str):
    import joblib
    path = Path(directory)
    metadata = joblib.load(path / "metadata.joblib")
    if tuple(metadata["dimensions"]) != DIMENSIONS:
        raise ValueError("Classifier label dimensions mismatch")
    return metadata, [joblib.load(path / (d + ".joblib")) for d in DIMENSIONS]


async def review(*, reports: list[dict], llm, options: dict) -> dict:
    model_name = options.get("attention_model")
    classifier_dir = options.get("classifier_dir")
    if not model_name or not classifier_dir:
        raise ValueError("A-Trust needs attention_model and classifier_dir")
    metadata, models = load_classifiers(str(classifier_dir))
    if metadata["attention_model"] != model_name:
        raise ValueError("Attention backbone differs from classifier training")
    evidence, suggestions, scores = [], [], {}
    for report in reports:
        if report["status"] != "done":
            evidence.append({"report_id": report["id"], "risk": 0.95,
                             "reason": "Sub-Agent did not complete"})
            suggestions.append({"report_id": report["id"], "action": "retry",
                                "reason": "Incomplete result"})
            continue
        x = attention_features(report, reports, model_name).reshape(1, -1)
        probs = {}
        for name, model in zip(DIMENSIONS, models):
            if model.n_features_in_ != x.shape[1]:
                raise ValueError(f"Feature length mismatch for {name}")
            probs[name] = float(model.predict_proba(x)[0, list(model.classes_).index(1)])
        risk = 1 - min(probs.values())
        scores[report["id"]] = probs
        evidence.append({"report_id": report["id"], "risk": risk,
                         "reason": "Attention-based min trust dimension: " +
                         min(probs, key=probs.get)})
        if risk >= float(options.get("threshold", 0.5)):
            suggestions.append({"report_id": report["id"], "action": "verify",
                                "reason": "Low attention-derived trust score"})
    return {"evidence": evidence, "suggestions": suggestions,
            "issues": [], "details": {"dimension_probabilities": scores,
                                     "attention_model": model_name}}
