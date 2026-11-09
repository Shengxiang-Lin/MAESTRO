"""Optional schema-first V2 scene author and independent critic.

Only scenario prose and source titles are editable; Goal family, difficulty,
source ID/modality/roles, and information requirements remain frozen.
No old MMQA question, answer, raw source content or sampled World is shown.
"""
from __future__ import annotations

import copy
import json
import re

from .spec import spec_issues, VERSION as SPEC_VERSION
from .intent import intent_contract

AUTHOR_SYSTEM = """Design one coherent, concrete FICTIONAL information-work scenario
based on a topic label and a FROZEN reasoning contract. This is a simulated
benchmark, not a claim about real teams, people, publications, images, audio,
videos or events.

Return concise narrative fields for the existing scenario, plus honest
hypothetical titles for each source. NEVER change any source ID/modality,
difficulty, solution operation, constraints, field mapping or answer.
The sources have FROZEN roles. They may describe (depending on the pattern):
approved category labels, time-window intervals for tracked events,
a binary-status disagreement, or per-group numerical thresholds.
The first source holds entry facts, the second translates source-specific
codes, the third gives independent observations or standards, and
a fourth (when present) supplies entry-specific clearance. NEVER silently
convert time-window matching or disagreement into category approval.
There are no real media to perceive; images, video and audio are simulated
structured annotations. Do not assert what original media actually shows.
Do not add any specific observed fact, person, historical year/date, filter,
threshold, score, or condition; those values do not exist yet.

Prefer topic-specific plausible user roles, purposeful evidence use, and
sources with distinguishable roles rather than generic boilerplate.
Return JSON ONLY:
{"scenario":{"setting":"fictional setting mentioning exact topic",
 "user_role":"...", "purpose":"...", "motivation":"...",
 "entity_kind":"short plural subject noun"},
 "source_titles":[{"source_id":"exact id","title":"..."}]}"""

CRITIC_SYSTEM = """Independent reviewer of a proposed FICTIONAL schema.
Evaluate topic coherence, relation semantics, source fidelity, and whether
narrative/sources add unsupported real facts, media observations, or changed
conditions. Treat the frozen contract as authoritative and do not infer what
original MultiModalQA question, answer or media said.

Return JSON only:
{"verdict":"pass|revise|reject",
 "topic_coherent":true,"same_relation_semantics":true,
 "source_faithful":true,"no_unlicensed_facts":true,
 "reason":"short reason"}
Pass only if all four booleans are true."""


def _narrative_issues(spec: dict, proposal: dict) -> list[str]:
    errors = []
    if not isinstance(proposal, dict):
        return ["proposal_not_object"]
    narrative = proposal.get("scenario")
    titles = proposal.get("source_titles")
    if not isinstance(narrative, dict) or not isinstance(titles, list):
        return ["invalid_proposal_structure"]
    keys = ("setting", "user_role", "purpose", "motivation", "entity_kind")
    if set(narrative) != set(keys):
        errors.append("unexpected_narrative_fields")
    for key in keys:
        value = narrative.get(key)
        if not isinstance(value, str) or not 3 <= len(value.strip()) <= 250:
            errors.append("invalid_" + key)
        elif re.search(r"\b(TaskGraph|WorkProduct|synthetic_attribute|original media)\b",
                       value, re.I):
            errors.append("implementation_or_original_media_claim")
    topic = spec["scenario"]["topic"]
    if isinstance(narrative.get("setting"), str):
        if topic.casefold() not in narrative["setting"].casefold():
            errors.append("topic_missing_from_setting")
    source_ids = {s["source_id"] for s in spec["sources"]}
    if (len(titles) != len(source_ids) or
            any(not isinstance(x, dict) or set(x) != {"source_id", "title"}
                for x in titles)):
        errors.append("invalid_source_title_list")
    else:
        ids = [row["source_id"] for row in titles]
        if set(ids) != source_ids or len(set(ids)) != len(ids):
            errors.append("source_ids_modified")
        if any(not isinstance(row["title"], str) or not 5 <= len(row["title"]) <= 170
               for row in titles):
            errors.append("invalid_source_titles")
    return sorted(set(errors))


def enrich_one(spec: dict, *, author, critic,
               max_revisions: int = 2) -> tuple[dict, dict]:
    if max_revisions < 0 or max_revisions > 8:
        raise ValueError("max revisions must be 0..8")
    frozen = {
        "topic": spec["scenario"]["topic"],
        "difficulty": spec["difficulty_target"],
        "goal_family": spec["family"],
        "semantic_pattern": spec.get("reasoning_pattern", "approval"),
        "source_scoped_conditions": (
            intent_contract(spec)["nonnegotiable_conditions"]
            if spec.get("version") == SPEC_VERSION else []
        ),
        "source_contract": [{
            "source_id": role["source_id"],
            "modality": role["modality"],
            "purpose": role["purpose"],
        } for role in spec["sources"]],
        "requirements": spec["requirements"],
    }
    failures = []
    for attempt in range(max_revisions + 1):
        try:
            proposal = author.complete_json(AUTHOR_SYSTEM, json.dumps({
                "frozen_contract": frozen,
                "previous_errors": failures[-2:],
            }, ensure_ascii=False))
            issues = _narrative_issues(spec, proposal)
            if issues:
                failures.append({"stage": "schema_guard", "errors": issues})
                continue
            review = critic.complete_json(CRITIC_SYSTEM, json.dumps({
                "frozen_contract": frozen,
                "candidate": proposal,
            }, ensure_ascii=False))
            axes = (
                "topic_coherent", "same_relation_semantics",
                "source_faithful", "no_unlicensed_facts",
            )
            if not (review.get("verdict") == "pass" and
                    all(review.get(key) is True for key in axes)):
                failures.append({"stage": "critic", "review": review})
                continue
            result = copy.deepcopy(spec)
            result["scenario"].update(proposal["scenario"])
            proposed_titles = {
                row["source_id"]: row["title"]
                for row in proposal["source_titles"]
            }
            for role in result["sources"]:
                role["title"] = proposed_titles[role["source_id"]]
            if spec_issues(result):
                failures.append({"stage": "post_validation", "errors": spec_issues(result)})
                continue
            result["scenario_authoring"] = {
                "status": "accepted", "attempts": attempt + 1,
                "review": review, "formal_equivalence_proven": False,
            }
            return result, result["scenario_authoring"]
        except Exception as exc:
            failures.append({
                "stage": "backend", "type": type(exc).__name__,
                "error": str(exc)[:240],
            })
    fallback = copy.deepcopy(spec)
    fallback["scenario_authoring"] = {
        "status": "fallback", "attempts": max_revisions + 1,
        "failures": failures, "formal_equivalence_proven": False,
    }
    return fallback, fallback["scenario_authoring"]
