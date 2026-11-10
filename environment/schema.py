"""Shared deterministic identifiers and source/entity label normalization."""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from urllib.parse import unquote


def digest(*parts: object) -> str:
    raw = json.dumps(parts, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def display(value: str) -> str:
    """A display label, never proof that two different entities are identical."""
    label = unquote(str(value).rstrip(">").split("/")[-1].split(":")[-1])
    label = label.replace("_", " ")
    label = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", label)
    return " ".join(label.split())


def normalize_label(value: object) -> str:
    value = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return " ".join(re.findall(r"\w+", value, flags=re.UNICODE))
