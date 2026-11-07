"""Opt-in reuse of already passed Stage2A Fragment Goals across runs.

Only reuse results of the SAME review protocol and model configuration,
and re-bind to an identical frozen public outline hash. Never reuse an
uncertain/rejected Fragment or infer a pass from a previous task's status.
"""
from __future__ import annotations

import json
from pathlib import Path

from .goal_review import GOAL_REVIEW_METHOD


def load_reusable_fragments(audit_path, *, model_roles):
    if not audit_path:
        return {}
    audit_path = Path(audit_path)
    summary_path = audit_path.with_name("goal_generation_summary.json")
    if not audit_path.is_file() or not summary_path.is_file():
        raise ValueError("reuse_goal_audit_requires_matching_summary")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("goal_review_contract") != GOAL_REVIEW_METHOD:
        raise ValueError("stale_goal_review_contract_cannot_be_reused")
    if summary.get("model_roles") != model_roles:
        raise ValueError("goal_cache_model_roles_mismatch")
    cache = {}
    with audit_path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            event = json.loads(line)
            task_id = event.get("task_id")
            audit = event.get("audit", [])
            reference = next((
                row.get("outline_sha256") for row in audit
                if row.get("stage") == "reference"
            ), None)
            if not task_id or not reference:
                continue
            for row in audit:
                if row.get("stage") != "fragment_goal" or row.get("accepted") is not True:
                    continue
                passed = next((
                    attempt for attempt in reversed(row.get("attempts", []))
                    if isinstance(attempt.get("review"), dict)
                    and attempt["review"].get("verdict") == "pass"
                    and isinstance(attempt.get("goal"), str)
                ), None)
                if passed:
                    key = (str(task_id), reference, row["fragment_id"])
                    cache[key] = {
                        "goal": passed["goal"], "review": passed["review"],
                    }
    return cache


__all__ = ["load_reusable_fragments"]
