"""JSONL audit pipeline for P0 scenario validity."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from ..rendering.pipeline import program_from_dict, world_from_dict
from .validity import assess_scenario


def _task_id(record: Mapping[str, Any]) -> str:
    basis = str(record.get("source_program_signature") or "")
    if not basis:
        basis = program_from_dict(record["program"]).canonical_signature()
    digest = hashlib.sha1(basis.encode("utf-8")).hexdigest()[:12]
    return f"world-{digest}"


def audit_scenarios_jsonl(
    input_path: str | Path,
    *,
    output_path: str | Path | None = None,
    max_records: int | None = None,
) -> dict[str, Any]:
    source = Path(input_path)
    writer = None
    target = Path(output_path) if output_path is not None else None
    if target is not None:
        target.parent.mkdir(parents=True, exist_ok=True)
        writer = target.open("w", encoding="utf-8")
    total = accepted = 0
    reason_counts: Counter[str] = Counter()
    decision_source_histogram: Counter[int] = Counter()
    indispensable_histogram: Counter[int] = Counter()
    required_fact_histogram: Counter[int] = Counter()
    incomplete_node_histogram: Counter[int] = Counter()
    try:
        with source.open("r", encoding="utf-8") as reader:
            for line in reader:
                if max_records is not None and total >= max_records:
                    break
                if not line.strip():
                    continue
                record = json.loads(line)
                program = program_from_dict(record["program"])
                world = world_from_dict(record["world"])
                report = assess_scenario(program, world)
                total += 1
                accepted += int(report.accepted)
                reason_counts.update(report.reasons)
                recoverability = report.details.get("evidence_recoverability", {})
                indispensable = recoverability.get("indispensable_fact_ids", ())
                indispensable_histogram[len(indispensable)] += 1
                required = recoverability.get("required_fact_ids", indispensable)
                required_fact_histogram[len(required)] += 1
                cover = recoverability.get("minimum_decision_recovery_source_cover", {})
                min_size = cover.get("min_size")
                if isinstance(min_size, int):
                    decision_source_histogram[min_size] += 1
                information = report.details.get("goal_information_completeness", {})
                missing = information.get("missing_requirements", ())
                if missing:
                    affected_nodes = {
                        int(item.get("node_id"))
                        for item in missing
                        if isinstance(item, Mapping) and isinstance(item.get("node_id"), int)
                    }
                    incomplete_node_histogram[len(affected_nodes)] += 1
                if writer is not None:
                    writer.write(json.dumps({"task_id": _task_id(record), **report.to_dict()}, ensure_ascii=False) + "\n")
    finally:
        if writer is not None:
            writer.close()
    return {
        "input": str(source), "output": str(target) if target is not None else None,
        "audited": total, "accepted": accepted,
        "acceptance_rate": accepted / total if total else 0.0,
        "rejected": total - accepted,
        "rejection_reasons": dict(reason_counts.most_common()),
        "minimum_decision_recovery_source_count_histogram": {str(k): v for k, v in sorted(decision_source_histogram.items())},
        "indispensable_fact_count_histogram": {str(k): v for k, v in sorted(indispensable_histogram.items())},
        "required_fact_count_histogram": {str(k): v for k, v in sorted(required_fact_histogram.items())},
        "incomplete_decision_node_count_histogram": {str(k): v for k, v in sorted(incomplete_node_histogram.items())},
    }
