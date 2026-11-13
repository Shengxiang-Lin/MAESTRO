"""Executable, source-grounded Claim relations and counterfactual warrant audits.

The private ledger is derived from ACTUAL synthetic Source observations that
are consumed by the frozen report Macros. This is not a second hidden World,
and it must never appear in the Main-Agent's public prompt.
"""
from __future__ import annotations

from collections import Counter

from .evidence_operators import (
    VERSION as OPERATORS_VERSION, WARRANT_POLICY,
    extract_claim, relate_claims, check_warrant,
)

VERSION = "maestro.evidence_relation_ledger.v1"
_REPORT_PATTERNS = {"disputed_reports", "corroborated_dispute"}


def compile_evidence_ledger(scene: dict) -> dict:
    spec = scene["problem_spec"]
    graph = scene["macro_graph"]
    sources = scene["world"]["sources"]
    wave_of_source = {}
    tasks = {task["task_id"]: task for task in graph["tasks"]}
    for i, wave in enumerate(graph["reference_waves"], 1):
        for tid in wave:
            for sid in tasks[tid]["source_ids"]:
                wave_of_source.setdefault(sid, i)
    plan = spec.get("macro_plan", {})
    pattern = plan.get("pattern")
    output = {
        "version": VERSION, "operator_version": OPERATORS_VERSION,
        "visibility": "private_oracle_audit_not_main_input",
        "warrant_policy": WARRANT_POLICY,
        "pattern": pattern,
        "claims": [], "relations": [],
        "relation_counts": {},
        "warrant_transitions": [],
        "independent_comparable_pairs": 0,
        "verification_sensitive_pairs": 0,
        "source_first_wave": {}, "unpaired_claims": {},
    }
    if pattern not in _REPORT_PATTERNS:
        return output
    if not {"report_a", "report_b"} <= set(sources):
        raise ValueError("claim_pair_sources_missing")
    by_source = {}
    for sid, field in (("report_a", "reported_year_a"),
                       ("report_b", "reported_year_b")):
        source = sources[sid]
        if not source.get("evidence_origin_id"):
            raise ValueError("claim_pair_independence_not_declared")
        index = {}
        for obs in source["observations"]:
            claim = extract_claim(
                source, obs, field=field,
                attribute="reported_event_year",
                valid_scope="same_reported_event",
            )
            if claim["subject"] in index:
                raise ValueError("duplicate_claim_subject_per_source")
            index[claim["subject"]] = claim
        by_source[sid] = index
        output["source_first_wave"][sid] = wave_of_source[sid]
    # Source-specific observation budgets can differ. Only claims about a
    # shared case are comparable; unpaired claims remain unverified.
    common = set(by_source["report_a"]) & set(by_source["report_b"])
    output["unpaired_claims"] = {
        sid: len(set(index) - common)
        for sid, index in sorted(by_source.items())
    }
    counts = Counter()
    for subject in sorted(common):
        first = by_source["report_a"][subject]
        second = by_source["report_b"][subject]
        relation = relate_claims(first, second)
        kind = relation["relation"]
        if kind not in {"corroborates", "contradicts"}:
            raise ValueError("report_claim_relation_not_independent:" + kind)
        with_first = check_warrant([first])
        with_both = check_warrant([first, second])
        changed = (
            with_first["status"] != with_both["status"] or
            with_first["sufficient_to_accept"] !=
            with_both["sufficient_to_accept"]
        )
        output["claims"].extend([first, second])
        output["relations"].append({
            "subject": subject, "left_claim_id": first["claim_id"],
            "right_claim_id": second["claim_id"],
            "relation": kind,
            "same_claim_key": relation["alignment"]["claim_key"],
            "source_fact_refs": sorted(first["fact_refs"] + second["fact_refs"]),
        })
        output["warrant_transitions"].append({
            "subject": subject,
            "comparison_scope": "source_ablation_not_real_arrival_order",
            "first_source_only": with_first,
            "both_sources": with_both,
            "verification_state_changes": changed,
        })
        counts[kind] += 1
        output["independent_comparable_pairs"] += 1
        output["verification_sensitive_pairs"] += int(changed)
    output["relation_counts"] = dict(sorted(counts.items()))
    return output
