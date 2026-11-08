"""Materialize clean, deterministic, multi-fact sources for Macro Task evidence.

Every source is a textual synthetic record surrogate. A modality hint (image,
video, audio) does not imply any actual media perception. Facts are generated
consistently for a single shared case reference and are never contradictory.
"""
from __future__ import annotations

import random

from envir.schema import digest


def _synthetic_facts(*, role: str, case_id: str, date: str,
                     issue: str, setting: str, user_role: str,
                     rng: random.Random) -> list[dict]:
    common = {
        "case_reference": (
            case_id, f"The source refers to review case {case_id}."
        ),
        "review_date": (
            date, f"The relevant review is dated {date}."
        ),
    }
    role_fact = {
        "material": (
            "observed_issue", issue,
            f"The primary observation concerns {issue} at the {setting}."
        ),
        "records": (
            "recorded_issue", issue,
            f"The indexed case record relates to {issue}."
        ),
        "rule_source": (
            "rule_scope", setting,
            f"The applicable guidance addresses activities at the {setting}."
        ),
        "independent_reports": (
            "independent_subject", issue,
            f"The independently recorded account concerns {issue}."
        ),
        "context": (
            "context_scope", setting,
            f"The background material concerns the {setting}."
        ),
        "capacity": (
            "resource_availability", rng.randint(3, 11),
            f"The case resource register records available capacity for {case_id}."
        ),
        "stakeholders": (
            "responsible_party", user_role,
            f"The coordination responsibility belongs to the {user_role}."
        ),
    }
    key, value, statement = role_fact.get(role, role_fact["context"])
    pairs = [
        ("case_reference", *common["case_reference"]),
        ("review_date", *common["review_date"]),
        (key, value, statement),
    ]
    return [
        {
            "fact_id": f"F{i+1}",
            "field": k,
            "value": val,
            "statement": stmt,
            "verification": "consistent_synthetic_record",
        }
        for i, (k, val, stmt) in enumerate(pairs)
    ]


def build_clean_world(plan: dict, *, seed: int) -> dict:
    rng = random.Random(int(digest(seed, plan["task_id"])[:14], 16))
    domain = plan["scenario"]["domain"]
    case_ref = f"{domain[:3].upper()}-{seed % 100:02d}-{plan['task_id'][-7:]}"
    day = 11 + rng.randrange(17)
    as_of = f"2026-10-{day:02d}"
    sources = {}
    grants = {}
    for task in plan["macro_graph"]["tasks"]:
        ids = []
        for requirement in task.get("source_requirements", []):
            role = requirement["role"]
            description = requirement["description"]
            source_id = "src-" + digest(plan["task_id"], role, description)
            if source_id not in sources:
                sources[source_id] = {
                    "source_id": source_id,
                    "role": role,
                    "title": description,
                    "modality_hint": requirement["candidate_modalities"][0],
                    "representation": "synthetic_text_record_not_raw_media",
                    "facts": _synthetic_facts(
                        role=role, case_id=case_ref, date=as_of,
                        issue=plan["scenario"]["issue"],
                        setting=plan["scenario"]["setting"],
                        user_role=plan["scenario"]["user_role"],
                        rng=rng,
                    ),
                }
            ids.append(source_id)
        grants[task["task_id"]] = sorted(ids)
    if not sources:
        raise ValueError("no root source requirements available")
    return {
        "world_id": "world-" + digest(plan["task_id"], "clean_v4"),
        "case_reference": case_ref,
        "as_of": as_of,
        "sources": sources,
        "source_grants": grants,
        "source_truth": "deterministic_simulated_only",
        "real_multimodal_files": False,
    }
