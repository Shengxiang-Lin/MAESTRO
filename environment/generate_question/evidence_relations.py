"""Executable, source-grounded Claim relations and counterfactual warrant audits.

The private ledger is derived from ACTUAL synthetic Source observations that
are consumed by the frozen report Macros. This is not a second hidden World,
and it must never appear in the Main-Agent's public prompt.
"""
from __future__ import annotations

from collections import Counter

from .evidence_operators import (
    VERSION as OPERATORS_VERSION, WARRANT_POLICY,
    extract_claim, align_claims, relate_claims, check_warrant, resolve_claim,
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
        "claims": [], "relations": [], "reference_evidence_program": [],
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
    # In keyed follow-up cases the second source can contain more rows than
    # the Macro is allowed to retrieve. Its unseen rows must NOT become
    # Claim evidence just because they exist privately in the World.
    accessible_refs = {}
    if spec.get("evidence_trust_version"):
        for task_feedback in scene["task_feedback"]:
            for sid in task_feedback["used_source_ids"]:
                if sid in {"report_a", "report_b"}:
                    accessible_refs.setdefault(sid, set()).update(
                        ref for row in task_feedback["workproduct"]
                        for ref in row["source_fact_refs"]
                        if ref.startswith(sid + ":")
                    )
    for sid, field in (("report_a", "reported_year_a"),
                       ("report_b", "reported_year_b")):
        source = sources[sid]
        if not source.get("evidence_origin_id"):
            raise ValueError("claim_pair_independence_not_declared")
        index = {}
        for obs in source["observations"]:
            if (spec.get("evidence_trust_version")
                    and not set(obs["fact_refs"]) & accessible_refs.get(sid, set())):
                continue
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
        # This is an executable PRIVATE evidence-program trace whose inputs
        # are existing Source observations, not fabricated Gold labels.
        # Real Main-Agent turns are assembled later from actual reports.
        output["reference_evidence_program"].extend([
            {
                "op": "ExtractClaim", "output": first["claim_id"],
                "source_id": first["source_id"],
                "fact_refs": first["fact_refs"],
            },
            {
                "op": "ExtractClaim", "output": second["claim_id"],
                "source_id": second["source_id"],
                "fact_refs": second["fact_refs"],
            },
            {
                "op": "AlignClaims",
                "inputs": [first["claim_id"], second["claim_id"]],
                "output": align_claims(first, second),
            },
            {
                "op": "RelateClaims",
                "inputs": [first["claim_id"], second["claim_id"]],
                "output": relation,
            },
            {
                "op": "CheckWarrant",
                "inputs": [first["claim_id"]],
                "output": with_first,
            },
            {
                "op": "CheckWarrant",
                "inputs": [first["claim_id"], second["claim_id"]],
                "output": with_both,
            },
            {
                "op": "ResolveClaim",
                "inputs": [first["claim_id"], second["claim_id"]],
                "output": resolve_claim([first, second]),
            },
        ])
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
    if spec.get("evidence_trust_version"):
        # Actual Macro availability differs from a hypothetical ablation:
        # a keyed follow-up cannot return until the case IDs are available.
        final_refs = {
            ref for row in scene["task_feedback"][-1]["workproduct"]
            for ref in row["source_fact_refs"]
        }
        reference_waves = sorted(set(output["source_first_wave"].values()))
        transitions = []
        answer_relevant = 0
        answer_relevant_late = 0
        cross_wave = 0
        for item in output["relations"]:
            subject = item["subject"]
            a = by_source["report_a"][subject]
            b = by_source["report_b"][subject]
            target = set(item["source_fact_refs"]) <= final_refs
            answer_relevant += int(target)
            a_wave = output["source_first_wave"]["report_a"]
            b_wave = output["source_first_wave"]["report_b"]
            cross_wave += int(a_wave != b_wave)
            answer_relevant_late += int(target and a_wave != b_wave)
            prior = check_warrant([])
            for wave in reference_waves:
                present = [
                    claim for claim in (a, b)
                    if output["source_first_wave"][claim["source_id"]] <= wave
                ]
                current = check_warrant(present)
                if (current["status"] != prior["status"] or
                        current["sufficient_to_accept"] !=
                        prior["sufficient_to_accept"]):
                    transitions.append({
                        "subject": subject,
                        "reference_wave": wave,
                        "new_source_ids": sorted(
                            sid for sid in ("report_a", "report_b")
                            if output["source_first_wave"][sid] == wave
                        ),
                        "before_status": prior["status"],
                        "after_status": current["status"],
                        "before_sufficient": prior["sufficient_to_accept"],
                        "after_sufficient": current["sufficient_to_accept"],
                        "evidence_count": current["evidence_count"],
                        "missing_evidence": current["missing_evidence"],
                        "answer_supporting_claim_pair": target,
                    })
                prior = current
        output["reference_wave_trust"] = {
            "visibility_scope": (
                "eligible_macro_workproducts_after_reference_wave;"
                "not_a_claim_that_freeform_finish_text_stated_every_row"
            ),
            "mode": plan.get("claim_verification_mode", "parallel"),
            "cross_wave_claim_pairs": cross_wave,
            "answer_relevant_claim_pairs": answer_relevant,
            "answer_relevant_cross_wave_pairs": answer_relevant_late,
            "transitions": sorted(
                transitions, key=lambda x: (
                    x["reference_wave"], x["subject"],
                )
            ),
        }
    return output
