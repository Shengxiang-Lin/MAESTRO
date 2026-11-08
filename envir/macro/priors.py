"""Read structural priors from real KQA-Pro/MMQA and optional YAGO bank.

These are priors for DESIGN. No real-world correctness, media recognition,
or successful feedback execution is asserted during macro planning.
"""
from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

from envir.schema import display
from envir.scenarios.mmqa_motifs import load_motifs
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


def load_mmqa(path: str | Path, *, limit: int = 3000) -> tuple[list[dict], dict]:
    return load_motifs(path, limit=limit)


def relation_hints(store, *, context: str, limit: int = 3) -> list[dict]:
    """Only suggest observed predicates with topic overlap; never invent facts."""
    if store is None:
        return []
    terms = {token for token in context.lower().replace("-", " ").split()
             if len(token) >= 4}
    if not terms:
        return []
    rows = store.db.execute(
        "SELECT predicate, COUNT(*) n FROM facts "
        "WHERE object_kind='entity' GROUP BY predicate "
        "HAVING COUNT(*)>=2 ORDER BY n DESC LIMIT 1200"
    ).fetchall()
    scored = []
    for row in rows:
        label = display(row["predicate"])
        tokens = {s for s in label.lower().split() if len(s) >= 4}
        score = len(terms & tokens)
        if score:
            scored.append((-score, -int(row["n"]), str(row["predicate"]), label))
    scored.sort()
    return [{"predicate": predicate, "label": label,
             "usage": "semantic_prior_only_not_grounded_fact"}
            for _, _, predicate, label in scored[:limit]]


def choose_kqa(programs: list[dict], motif: str, rng: random.Random,
               ordinal: int) -> tuple[dict, str]:
    # A full KoPL program informs the internal complexity family. It never
    # fixes the macro boundary or creates a one-KoPL-node-one-agent mapping.
    def score(p):
        depth = p["logical_depth"]
        branches = p["branch_nodes"] + p["merge_nodes"]
        if motif in {"parallel_join", "fork_join", "mixed", "staged"}:
            return (2 * min(2, branches) + min(depth, 7))
        return min(depth, 9) + (1 if branches == 0 else 0)
    top_score = max(score(p) for p in programs)
    pool = [p for p in programs if score(p) >= max(3, top_score - 3)]
    selected = pool[(ordinal + rng.randrange(len(pool))) % len(pool)]
    return selected, "full_kopl_program_internal_reasoning_prior"


def choose_mmqa(motifs: list[dict], *, domain: str, motif: str,
                rng: random.Random, ordinal: int) -> tuple[dict, str]:
    family = {
        "chain": ("composition", "lookup"),
        "parallel_join": ("intersection", "comparison"),
        "fork_join": ("composition", "intersection"),
        "mixed": ("comparison", "intersection", "composition"),
        "staged": ("composition", "comparison"),
    }[motif]
    domain_pool = [m for m in motifs if m.get("domain_hint") == domain
                   and m.get("logical_pattern") in family]
    structural_pool = [m for m in motifs if m.get("logical_pattern") in family]
    candidates = domain_pool or structural_pool or motifs
    item = candidates[(ordinal + rng.randrange(len(candidates))) % len(candidates)]
    alignment = (
        "topic_and_composition_prior" if item.get("domain_hint") == domain
        else "composition_prior_only"
    )
    return item, alignment
