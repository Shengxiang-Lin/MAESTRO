"""Concrete information needs before macro delegation.

The original KQA/MMQA datasets influence question operations and evidence
composition. Names, dates, rules and measurements belong to a controlled
*synthetic* world. There are no actual media files or real-world guarantees.
"""
from __future__ import annotations

import random
from collections import Counter
from dataclasses import replace

from envir.schema import digest
from .patterns import CASES, CASE_VARIANTS
from .priors import choose_kqa, choose_mmqa


# Domain-local subject terminology and source surfaces. An event is a complete
# incident/case, not a single OCR or KoPL operation.
DOMAIN = {
    "logistics": ("freight movement", "vehicle", "TR", "gate monitoring report",
                  "dispatch and carrier register", "terminal access policy"),
    "culture": ("museum session", "session", "MU", "exhibition display report",
                "program and booking records", "venue scheduling guidelines"),
    "sports": ("tournament entry", "entry", "SP", "match video report",
               "participant and result register", "competition eligibility rules"),
    "education": ("symposium session", "session", "ED", "event poster report",
                  "session and speaker register", "conference scheduling rules"),
    "science": ("instrument run", "run", "LB", "instrument observation report",
                "experiment and maintenance records", "laboratory quality rules"),
    "commerce": ("distribution batch", "batch", "BT", "package label report",
                 "inventory and order register", "recall handling rules"),
    "environment": ("habitat observation", "observation", "WB", "field camera report",
                    "wildlife survey register", "habitat access rules"),
    "civic": ("maintenance job", "job", "CV", "inspection media report",
              "service and work-order register", "city maintenance requirements"),
    "general": ("venue booking", "booking", "EV", "site notice report",
                "booking and attendee register", "event access rules"),
}

# Final request families can share a macro topology; topology does not
# determine intent. Their operation requirements constrain task construction.
INTENT_FAMILIES = {
    "composition": ("compliance", "impact", "timeline", "feasibility", "mitigation"),
    "intersection": ("compliance", "impact", "feasibility", "prioritization", "mitigation"),
    "comparison": ("comparison", "timeline", "prioritization", "reconciliation", "mitigation"),
    "aggregation": ("impact", "reconciliation", "prioritization", "comparison"),
    "lookup": ("compliance", "timeline", "reconciliation", "mitigation",
               "feasibility", "comparison"),
}
REQUIRED_SOURCE_ROLES = {
    "compliance": ("event", "records", "rules"),
    "impact": ("event", "records", "rules"),
    "feasibility": ("event", "records", "rules", "capacity"),
    "mitigation": ("event", "records", "capacity"),
    "prioritization": ("event", "records", "capacity"),
    "comparison": ("event", "records"),
    "timeline": ("event", "independent"),
    "reconciliation": ("event", "independent"),
}
SOURCE_MODALITIES = {
    "event": "simulated_media_observation",
    "records": "table",
    "rules": "document",
    "capacity": "table",
    "independent": "document",
}


def _ops(kqa_pattern):
    return [n["op"] for n in kqa_pattern["program"]["nodes"]]


def choose_question_seed(*, ordinal, seed, kqa, mmqa):
    rng = random.Random(seed * 7919 + ordinal * 104729)
    base = CASES[ordinal % len(CASES)]
    iteration = ordinal // len(CASES)
    variants = CASE_VARIANTS[base.domain]
    case = replace(base, issue=variants[(iteration * 3 + seed) % len(variants)])
    # MMQA is picked BEFORE deciding the question and task graph. The
    # original compositional family influences the information requirement.
    pattern_hint = ("chain", "parallel_join", "mixed", "fork_join", "staged")[
        (iteration + ordinal * 3 + seed) % 5
    ]
    mp, alignment = choose_mmqa(
        mmqa, domain=case.domain, motif=pattern_hint,
        rng=rng, ordinal=ordinal,
    )
    kp, kqa_policy = choose_kqa(kqa, pattern_hint, rng, ordinal)
    ops = _ops(kp)
    mmqa_family = mp["logical_pattern"]
    candidates = INTENT_FAMILIES.get(mmqa_family, INTENT_FAMILIES["lookup"])
    # KQA operator semantics adjust the user's requested output, not the
    # number of Sub-Agents. Select can favor comparison; Count favors impact.
    preferred = (
        "comparison" if any(op.startswith("SELECT") for op in ops)
        else "impact" if "COUNT" in ops
        else "compliance" if any(op.startswith("VERIFY") for op in ops)
        else "reconciliation" if "INTERSECT" in ops
        else None
    )
    valid = list(candidates)
    if preferred in valid and (iteration % 3 == 0):
        intent = preferred
    else:
        intent = valid[(iteration + ordinal + seed) % len(valid)]
    # Every result requests a concrete case-scoped answer; two-case
    # comparisons are separate instances of the SAME work-product types.
    cases = 2 if intent == "comparison" else 4
    ids = [f"{DOMAIN[case.domain][2]}-{ordinal+101:04d}-{j+1}"
           for j in range(cases)]
    day = 11 + (ordinal % 17)
    as_of = f"2026-10-{day:02d}"
    # Vary whether record auditing is independent or uses the event's
    # discovered case IDs; no artificial edge is ever inserted.
    scoped_records = (rng.randrange(3) != 0)
    count_result = ("COUNT" in ops or mmqa_family == "aggregation")
    return {
        "seed_id": "need-" + digest(seed, ordinal, case.domain, intent, ids, mp["motif_id"]),
        "domain": case.domain,
        "setting": case.setting,
        "issue": case.issue,
        "user_role": case.decision_owner,
        "intent": intent,
        "case_term": DOMAIN[case.domain][1],
        "entity_plural": DOMAIN[case.domain][0] + "s",
        "case_ids": ids,
        "as_of": as_of,
        "record_scope": "event_scoped" if scoped_records else "independent",
        "capacity_scope": "case_scoped" if rng.randrange(2) else "independent",
        "independent_scope": "event_scoped" if rng.randrange(2) else "independent",
        "required_source_roles": list(REQUIRED_SOURCE_ROLES[intent]),
        "question_operation": (
            "pairwise_comparison" if intent == "comparison" else
            "thresholded_intersection" if intent in {"compliance", "impact", "feasibility"} else
            "numeric_prioritization" if intent in {"mitigation", "prioritization"} else
            "temporal_reconciliation"
        ),
        "aggregation_mode": "count" if count_result else "enumerate",
        "source_names": {
            "event": DOMAIN[case.domain][3],
            "records": DOMAIN[case.domain][4],
            "rules": DOMAIN[case.domain][5],
            "capacity": "operating capacity register",
            "independent": "independently logged observations",
        },
        "source_media_hint": list(case.entry_modalities),
        "source_priors": {
            "KQA": {
                "policy": kqa_policy,
                "signature": kp["signature"],
                "operators": ops,
                "source_question": kp.get("source_question", ""),
                "contributed_to": ["result_mode", "reasoning_operators"],
                "is_exact_KoPL_transfer": False,
            },
            "MultiModalQA": {
                "policy": alignment,
                "motif_id": mp["motif_id"],
                "source_question": mp["question_hint"],
                "question_type": mp["question_type"],
                "source_modalities": mp["source_modalities"],
                "logical_pattern": mmqa_family,
                "contributed_to": ["question_family", "source_roles"],
                "is_original_media_used": False,
            },
        },
    }


def make_world(need: dict, seed: int) -> dict:
    """Deterministically produce a coherent multi-source counterfactual world.

    Sources represent different information partitions. No source contains
    the final Gold answer, and no subtask can access undisclosed partitions.
    """
    rng = random.Random(int(digest(need["seed_id"], seed)[:14], 16))
    ids = need["case_ids"]
    n = len(ids)
    opening = 9 * 60
    cutoff = 13 * 60 + rng.choice((0, 30, 60))
    # Sorted but nontrivial event dates; one event intentionally outside
    # operational time bounds. Multiple subjects avoid trivial lookups.
    minutes = [opening + 45 + j * 80 + rng.randrange(0, 25)
               for j in range(n)]
    statuses = ["valid", "expired", "valid", "suspended"][:n]
    units = [max(1, 2 + j * 3 + rng.choice((-1, 0, 1)))
             for j in range(n)]
    urgency = [max(1, x + rng.choice((-1, 0, 1)))
               for x in [3, 2, 5, 4][:n]]
    capacity = [max(3, units[0] + 1), 2, 7, 1][:n]
    rng.shuffle(statuses)
    # Witness conditions: a valid early case, an invalid in-window case,
    # and a late nonvalid case. This makes rule/record counterfactuals real.
    statuses[0] = "valid"
    statuses[1] = "expired"
    if n >= 4:
        statuses[2] = "valid"
        statuses[3] = "suspended"
    deltas = [0, 17, -4, 24][:n]
    sources = {}
    source_titles = need["source_names"]
    source_ids = {
        role: f"{need['seed_id']}:{role}"
        for role in need["required_source_roles"]
    }
    event = [
        {"case_id": cid, "event_minute": minutes[j],
         "event_day": need["as_of"], "observation": f"observed {need['case_term']} {cid}"}
        for j, cid in enumerate(ids)
    ]
    records = [
        {"case_id": cid, "status": statuses[j], "units": units[j],
         "urgency": urgency[j], "registered_date": need["as_of"]}
        for j, cid in enumerate(ids)
    ]
    rules = {
        "valid_status": "valid",
        "closing_minute": cutoff,
        "min_affected_units": 5,
        "as_of": need["as_of"],
    }
    capacities = [{"case_id": cid, "available_units": capacity[j]}
                  for j, cid in enumerate(ids)]
    # Unrelated rows make an upstream case-ID handoff meaningful for scoped
    # investigators. Source documents are not pre-filtered to the answer.
    capacities.append({"case_id": f"UNRELATED-{need['seed_id'][-5:]}",
                       "available_units": 9})
    independent = [
        {"case_id": cid, "reported_minute": minutes[j] + deltas[j],
         "report_day": need["as_of"]}
        for j, cid in enumerate(ids)
    ]
    independent.append({
        "case_id": f"UNRELATED-{need['seed_id'][-5:]}",
        "reported_minute": opening + 29,
        "report_day": need["as_of"],
    })
    payloads = {
        "event": {"events": event, "day": need["as_of"]},
        "records": {"records": records, "day": need["as_of"]},
        "rules": {"rule": rules},
        "capacity": {"capacities": capacities},
        "independent": {"observations": independent},
    }
    for role in need["required_source_roles"]:
        sources[source_ids[role]] = {
            "source_id": source_ids[role],
            "role": role,
            "title": source_titles[role],
            "modality": (
                need["source_media_hint"][0] if role == "event"
                else SOURCE_MODALITIES[role]
            ),
            "representation": (
                "textual_observation_surrogate_not_an_actual_media_file"
                if role == "event" else "simulated_source_record"
            ),
            "observations": payloads[role],
        }
    return {
        "world_id": "world-" + digest(need["seed_id"], "controlled_sources"),
        "scope_ids": ids,
        "source_ids": source_ids,
        "sources": sources,
        "private_origin": "deterministic_synthetic_facts",
        "real_media_available": False,
        "dataset_facts_used": False,
    }


def public_question(need: dict) -> str:
    """Specify entities, date, evidence roles and answer schema, not work steps."""
    ids = need["case_ids"]
    joined_ids = ", ".join(ids[:-1]) + " and " + ids[-1]
    role_names = [need["source_names"][r] for r in need["required_source_roles"]]
    resources = ", ".join(role_names[:-1]) + " and " + role_names[-1]
    prefix = (
        f"At the {need['setting']}, consider {need['entity_plural']} "
        f"{joined_ids} on {need['as_of']} in connection with {need['issue']}. "
        f"Use the {resources}. "
    )
    intent = need["intent"]
    if intent == "compliance":
        ask = ("Which of these cases fail the applicable status or time-window "
               "requirements? Return the case IDs and the violated condition.")
    elif intent == "impact":
        ask = ("Which cases have a non-valid registered status and meet the "
               "minimum affected-unit threshold in the applicable rules? "
               "Report the case IDs and total affected units.")
        if need["aggregation_mode"] == "count":
            ask += " Include the number of affected cases."
    elif intent == "comparison":
        ask = ("Which of the two cases has the larger recorded workload? "
               "Return the case ID and the difference in units, matching "
               "each case observation to its registered entry.")
    elif intent == "timeline":
        ask = ("Which case has the largest absolute time difference between "
               "the primary observation and the independent report? "
               "Return the case ID and the gap in minutes.")
    elif intent == "reconciliation":
        ask = ("Which case timestamps disagree by more than ten minutes "
               "between the primary and independent observations? "
               "List those case IDs and their time differences.")
    elif intent == "feasibility":
        ask = ("Which cases have valid registration, occur within the allowed "
               "time window and have enough available capacity for their "
               "recorded workload? Return the case IDs.")
    elif intent == "prioritization":
        ask = ("Which case should be reviewed first using workload, urgency "
               "and unused operating capacity? Apply the documented priority "
               "formula (workload × urgency − available units) and give "
               "the highest-ranked case ID and score.")
    elif intent == "mitigation":
        ask = ("Which case has the largest workload shortfall relative to "
               "available capacity? Return the case ID, shortfall in units "
               "and the resulting immediate mitigation priority.")
    else:
        raise ValueError("unsupported V3 intent")
    return prefix + ask


def seed_statistics(seeds: list[dict]) -> dict:
    return {
        "domain_counts": dict(Counter(s["domain"] for s in seeds)),
        "intent_counts": dict(Counter(s["intent"] for s in seeds)),
        "mmqa_families": dict(Counter(
            s["source_priors"]["MultiModalQA"]["logical_pattern"] for s in seeds)),
        "question_operations": dict(Counter(s["question_operation"] for s in seeds)),
        "record_scope": dict(Counter(s["record_scope"] for s in seeds)),
        "distinct_kqa_signatures": len({s["source_priors"]["KQA"]["signature"] for s in seeds}),
        "distinct_mmqa_motifs": len({s["source_priors"]["MultiModalQA"]["motif_id"] for s in seeds}),
    }
