"""Materialize clean, deterministic, multi-fact sources for Macro Task evidence.

Every source is a textual synthetic record surrogate. A modality hint (image,
video, audio) does not imply any actual media perception. Facts are generated
consistently for a single shared case reference and are never contradictory.
"""
from __future__ import annotations

import random

from envir.schema import digest


# Domain-specific evidence descriptors are synthetic operational semantics,
# never copied truth from an original KQA/MMQA record.
DOMAIN_FIELDS = {
    "logistics": ("vehicle movement", "dispatch log", "gate access", "carrier"),
    "culture": ("museum event", "exhibition booking", "gallery schedule", "visitor"),
    "sports": ("competition event", "results register", "eligibility", "participant"),
    "education": ("conference session", "program entry", "venue assignment", "speaker"),
    "science": ("experimental run", "instrument log", "calibration", "researcher"),
    "commerce": ("product batch", "shipment record", "recall handling", "customer"),
    "environment": ("field observation", "survey register", "habitat protection", "ranger"),
    "civic": ("maintenance incident", "service order", "safety procedure", "technician"),
    "general": ("venue event", "booking record", "access requirement", "attendee"),
}


def _synthetic_facts(*, role: str, domain: str, case_id: str, date: str,
                     issue: str, setting: str, user_role: str,
                     rng: random.Random) -> list[dict]:
    event_label, record_label, rule_label, stakeholder_label = DOMAIN_FIELDS.get(
        domain, DOMAIN_FIELDS["general"])
    capacity = rng.randint(6, 13)
    requested = rng.randint(2, capacity)
    # Each source has its own independent facts and an explicit case and
    # date scope. Extra observations are meaningful source fields, not
    # hallucinated OCR or claims that a real media model ran.
    details = {
        "material": [
            ("observation_type", event_label,
             f"The primary observation concerns a {event_label}."),
            ("observed_issue", issue,
             f"The observed {event_label} concerns {issue}."),
            ("observation_location", setting,
             f"The primary account places the incident at the {setting}."),
        ],
        "records": [
            ("register_type", record_label,
             f"The relevant entries are recorded in the {record_label}."),
            ("indexed_subject", case_id,
             f"The {record_label} includes the case reference {case_id}."),
            ("record_scope", issue,
             f"The indexed entry relates to {issue}."),
        ],
        "rule_source": [
            ("requirement_type", rule_label,
             f"The applicable requirement area is {rule_label}."),
            ("rule_scope", setting,
             f"The guidance applies to activities at the {setting}."),
            ("review_condition", "review_case_before_action",
             f"Activities under {rule_label} require case-scoped review before action."),
        ],
        "independent_reports": [
            ("independent_type", event_label,
             f"A separately recorded account concerns the {event_label}."),
            ("independent_case", case_id,
             f"The separate report references the same case {case_id}."),
            ("independent_issue", issue,
             f"The separate report describes {issue}."),
        ],
        "context": [
            ("context_type", event_label,
             f"Background documents describe the {event_label} process."),
            ("context_location", setting,
             f"Those documents apply to the {setting}."),
            ("context_relevance", issue,
             f"The contextual material is relevant to {issue}."),
        ],
        "capacity": [
            ("available_units", capacity,
             f"The synthetic capacity register lists {capacity} available units."),
            ("requested_units", requested,
             f"The case requests {requested} units from the same register."),
            ("capacity_scope", case_id,
             f"Both capacity entries are scoped to {case_id}."),
        ],
        "stakeholders": [
            ("decision_owner", user_role,
             f"The coordinating role is the {user_role}."),
            ("stakeholder_group", stakeholder_label,
             f"The relevant stakeholder group is the {stakeholder_label}."),
            ("stakeholder_requirement", "documented_resolution",
             f"The stakeholder requires a documented resolution of {issue}."),
        ],
    }
    if role not in details:
        raise ValueError("unsupported source role: " + role)
    pairs = [
        ("case_reference", case_id,
         f"The source is indexed under case {case_id}."),
        ("review_date", date,
         f"The evidence in this source is scoped to {date}."),
        *details[role],
    ]
    return [
        {
            "fact_id": f"F{i+1}",
            "field": key,
            "value": value,
            "statement": claim,
            "verification": "consistent_synthetic_record",
        }
        for i, (key, value, claim) in enumerate(pairs)
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
                        role=role, domain=domain, case_id=case_ref, date=as_of,
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
