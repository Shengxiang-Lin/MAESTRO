"""Difficulty diagnostics for the V2 source/observation and reasoning-IR pilot.

A missing-source error proves an access dependency only. Counterfactual
answer changes offer stronger evidence of semantic relevance, but do NOT
prove minimality of an arbitrary reasoning program or real Agent difficulty.
"""
from __future__ import annotations

import copy

from .program_ir import (
    MissingSourceError, answer_values, evaluate_program, program_complexity,
)


def _mutate_source(sid: str, sources: dict) -> dict:
    changed = copy.deepcopy(sources)
    rows = changed[sid]["observations"]
    if sid == "ledger":
        rows[0]["values"]["active"] = False
    elif sid == "codebook":
        rows[0]["values"]["position_name"] = "tackle"
    elif sid == "board":
        rows[0]["values"]["approved"] = False
    elif sid == "policy":
        rows[0]["values"]["cleared"] = False
    else:
        raise ValueError("no_pilot_counterfactual:" + sid)
    return changed


def audit_linked_case(case: dict) -> dict:
    program, sources = case["gold_program"], case["world"]["sources"]
    execution = evaluate_program(program, sources=sources)
    answer = answer_values(execution)
    if answer != case["gold_answer"]:
        raise ValueError("gold_answer_does_not_replay")
    # Deliberately weak no-join baseline. Not an exhaustive single-operator
    # oracle; do not interpret failure as a calibrated hardness estimate.
    active_only = [
        {"player_label": row["values"]["player_label"]}
        for row in sources["ledger"]["observations"]
        if row["values"]["active"]
    ]
    active_only.sort(key=lambda row: row["player_label"])
    source_tests = {}
    for sid in sorted(sources):
        remaining = {key: value for key, value in sources.items() if key != sid}
        try:
            evaluate_program(program, sources=remaining)
            access_blocked = False
        except MissingSourceError:
            access_blocked = True
        alternative = answer_values(evaluate_program(
            program, sources=_mutate_source(sid, sources)
        ))
        source_tests[sid] = {
            "missing_source_blocks_this_program": access_blocked,
            "counterfactual_changes_answer": alternative != answer,
            "alternative_answer": alternative,
        }
    return {
        "difficulty_target": case["difficulty_target"],
        "reasoning": program_complexity(program),
        "answer": answer,
        "answer_source_fact_refs": {
            row["values"]["player_label"]: list(row["source_fact_refs"])
            for row in execution["answer"]
        },
        "active_only_baseline_answer": active_only,
        "active_only_baseline_exact": active_only == answer,
        "source_tests": source_tests,
        "all_sources_counterfactually_relevant": all(
            value["counterfactual_changes_answer"]
            for value in source_tests.values()
        ),
        "limitations": [
            "Pilot example, not empirical evidence of full-benchmark difficulty.",
            "Missing source merely blocks the stated program, not all solutions.",
            "No exhaustive minimization of alternative programs performed.",
            "No original media pixels, video, audio or web pages were processed.",
        ],
    }
