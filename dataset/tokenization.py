"""Token-ID normalization shared by MAESTRO's exporter and DA-GRPO policy.

Transformers' apply_chat_template(tokenize=True) may return a plain
list, tensor, numpy array, or BatchEncoding/dict (depending on version
and return_dict/return_tensors defaults). Always count actual token IDs.
"""
from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral
from typing import Any


def extract_token_ids(encoded: Any) -> list[int]:
    """Extract exactly one nonempty tokenized conversation, without truncation."""
    if isinstance(encoded, Mapping):
        if "input_ids" not in encoded:
            raise ValueError(
                "Tokenizer returned a mapping without 'input_ids': "
                f"{list(encoded.keys())}"
            )
        encoded = encoded["input_ids"]

    if hasattr(encoded, "tolist") and not isinstance(encoded, (list, tuple)):
        encoded = encoded.tolist()

    if isinstance(encoded, (list, tuple)) and len(encoded) == 1 and isinstance(
        encoded[0], (list, tuple)
    ):
        encoded = encoded[0]

    if not isinstance(encoded, (list, tuple)) or not encoded:
        raise ValueError(
            "Qwen chat template did not produce one nonempty token ID sequence "
            f"(received {type(encoded).__name__})"
        )

    if not all(isinstance(token, Integral) and not isinstance(token, bool)
               for token in encoded):
        raise ValueError(
            "Qwen chat template returned non-integer or batched token IDs; "
            "expected exactly one sequence"
        )
    return [int(token) for token in encoded]
