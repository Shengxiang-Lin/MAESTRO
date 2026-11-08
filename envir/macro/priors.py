"""Full original KQA-Pro KoPL ProgramIR as an internal reasoning prior.

ProgramIR is retained for later per-Sub-Agent inference/traces. Its operators
are not Macro Tasks and are not purported to execute during question generation.
"""
from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

from environment.task_construction.data_sources.kqa import record_to_ir


def load_kqa(path: str | Path, *, limit: int = 10000) -> tuple[list[dict], dict]:
    if limit < 1:
        raise ValueError("KQA record limit must be positive")
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("KQA-Pro input must be an array")
    patterns = {}
    stats = Counter()
    for record in data[:limit]:
        if not isinstance(record, dict) or not isinstance(record.get("program"), list):
            stats["missing_program"] += 1
            continue
        try:
            ir = record_to_ir(record)
        except (ValueError, TypeError, KeyError):
            stats["invalid_program"] += 1
            continue
        if len(ir.nodes) < 3:
            continue
        shape = ir.metrics()
        signature = ir.canonical_signature()
        stats["source_programs"] += 1
        if signature not in patterns:
            patterns[signature] = {
                "signature": signature,
                "program": ir.to_dict(),
                "source_question": str(record.get("question") or "")[:240],
                "logical_depth": shape["depth"],
                "branch_nodes": shape["branch_nodes"],
                "merge_nodes": shape["merge_nodes"],
                "operator_count": len(ir.nodes),
            }
    result = list(patterns.values())
    if not result:
        raise ValueError("no usable full KQA programs")
    return result, {**dict(stats), "unique_full_programs": len(result)}


def choose_kqa(programs: list[dict], motif: str, rng: random.Random,
               ordinal: int) -> tuple[dict, str]:
    def score(program):
        depth = program["logical_depth"]
        branches = program["branch_nodes"] + program["merge_nodes"]
        if motif in {"parallel_join", "fork_join", "mixed", "staged"}:
            return 2 * min(2, branches) + min(depth, 7)
        return min(depth, 9) + (1 if branches == 0 else 0)

    best = max(map(score, programs))
    eligible = [p for p in programs if score(p) >= max(3, best - 3)]
    return (eligible[(ordinal + rng.randrange(len(eligible))) % len(eligible)],
            "internal_kopl_program_shape_only")
