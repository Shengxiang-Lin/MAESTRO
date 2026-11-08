"""Public task contracts and private, verifiable subtask records."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any


def digest(*parts: object) -> str:
    raw = json.dumps(parts, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def display(value: str) -> str:
    """Only a display label; never use it to prove identity equivalence."""
    import re
    from urllib.parse import unquote

    label = unquote(str(value).rstrip(">").split("/")[-1].split(":")[-1])
    label = label.replace("_", " ")
    label = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", label)
    return " ".join(label.split())


def normalize_label(value: object) -> str:
    import re
    import unicodedata

    value = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return " ".join(re.findall(r"\w+", value, flags=re.UNICODE))


@dataclass(frozen=True)
class SubtaskTemplate:
    template_id: str
    kind: str
    input_type: str
    output_type: str
    goal_pattern: str
    predicate: str | None = None
    dataset: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Subtask:
    subtask_id: str
    template_id: str
    kind: str
    goal: str
    inputs: dict[str, Any]
    output_type: str
    answer: dict[str, Any]
    evidence_ids: tuple[str, ...]
    tool_events: tuple[dict[str, Any], ...]
    summary: str
    summary_trace: tuple[str, ...]
    verification: str
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TaskGraph:
    task_id: str
    pattern: str
    nodes: tuple[dict[str, Any], ...]
    dependencies: tuple[dict[str, Any], ...]
    final_node: str
    question_brief: dict[str, Any]
    question: str
    gold_answer: dict[str, Any]
    optional_candidate_ids: tuple[str, ...] = ()
    validation: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
