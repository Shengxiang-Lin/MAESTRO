"""Optional schema-first V2 scene author and independent critic.

Only scenario prose and source titles are editable; Goal family, difficulty,
source ID/modality/roles, and information requirements remain frozen.
No old MMQA question, answer, raw source content or sampled World is shown.
"""
from __future__ import annotations

import copy
import json
import re

from environment.generate_question.planning.spec import spec_issues, VERSION as SPEC_VERSION
from environment.generate_question.authoring.intent import intent_contract
from environment.generate_question.authoring.surface_quality import unlicensed_calendar_years, topic_mentioned

AUTHOR_SYSTEM = """Design one coherent, concrete FICTIONAL information-work scenario
based on a topic label and a FROZEN reasoning contract. Keep the original
Topic's exact proper name in the narrative setting. You may omit a
non-essential parenthetical type qualifier (e.g. "(film)") but must NEVER
substitute a different person, organization, product or city. This is a simulated
benchmark, not a claim about real teams, people, publications, images, audio,
videos or events.

Return concise narrative fields for the existing scenario, plus honest
hypothetical titles for each source. NEVER change any source ID/modality,
difficulty, solution operation, constraints, field mapping or answer.
The sources have FROZEN roles. They may describe (depending on the pattern):
approved category labels, time-window intervals for tracked events,
a binary-status disagreement, or per-group numerical thresholds.
For macro_first patterns, each pre-defined Macro Task has its own
independently executable local evidence search. Preserve the frozen
Macro-level dependencies and sources. Do not reveal the internal Gold
Solver operations in the user-facing question. The final objective
is the authoritative year named in the frozen Goal; do not invent an
extra eligibility condition or assume original media was inspected.
For evidence_graph patterns, the source topology is frozen in a typed
information-acquisition graph. Preserve its distinct observable inputs and
keyed-only stores. A conflict graph starts from two disagreeing independent
inventories; a parallel graph combines visual and caption clues; a
candidate-disambiguation graph requires independent notes. Do not imply
that any keyed source can be fully browsed or that the public question must
reveal the entire source traversal.
For clue_chain patterns, the image is a browsable illustration that yields
a marker only after feature identification, the second is a KEYED index
that reveals an accession identifier for that marker only, and the third
is a KEYED archive revealing an accession year for that identifier only.
Never describe the latter two as freely browseable full tables.
For other patterns, the first source holds entry facts, the second
translates source-specific codes, the third gives independent
observations or standards, and a fourth (when present) supplies
entry-specific clearance. NEVER silently
convert time-window matching or disagreement into category approval.
There are no real media to perceive; images, video and audio are simulated
structured annotations. Do not assert what original media actually shows.
Do not add any specific observed fact, person, historical year/date, filter,
threshold, score, or condition; those values do not exist yet. In
particular, do not invent a calendar year (e.g. 2024), school term,
fiscal year or named historical event unless it appears verbatim in the
FROZEN original topic.

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
Years and named dates absent from the ORIGINAL topic are unlicensed,
even if framed as an invented project cycle; do not assume the simulator
contains them. An appealing narrative must not imply an extra
eligibility rule or unsupported source content.
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
        if not isinstance(value, str):
            errors.append(f"scenario_{key}_invalid_type")
        elif len(value.strip()) < 3:
            errors.append(f"scenario_{key}_too_short")
        elif len(value.strip()) > 250:
            errors.append(f"scenario_{key}_too_long")
        elif re.search(r"\b(TaskGraph|WorkProduct|synthetic_attribute|original media)\b",
                       value, re.I):
            errors.append(f"scenario_{key}_internal_jargon")
    topic = spec["scenario"]["topic"]
    if isinstance(narrative.get("setting"), str):
        if not topic_mentioned(narrative["setting"], topic):
            errors.append("topic_missing_from_setting")
    # Do not authorize a new historical anchor simply because both
    # Scenario Author and Critic happen to find it plausible.
    if spec.get("version") == SPEC_VERSION:
        proposed_text = " ".join(
            str(narrative.get(key, "")) for key in keys
        ) + " " + " ".join(
            str(x.get("title", "")) for x in titles
            if isinstance(x, dict)
        )
        if unlicensed_calendar_years(proposed_text, topic):
            errors.append("unlicensed_calendar_year_not_in_topic")
    if spec.get("reasoning_pattern") == "macro_first_v2":
        phrase = " ".join(str(narrative.get(k, "")) for k in keys).lower()
        if re.search(r"\b(fictional|simulated|synthetic|hypothetical|imaginary)\b",
                     phrase):
            errors.append("v2_meta_fiction_in_public_scenario")
        goal = spec["macro_plan"]["goal"]["public_label"]
        if goal not in {"founding year", "release year"} and re.search(
            r"\b(founding year|release year)\b", phrase
        ):
            errors.append("v2_unsupported_goal_substitution")
    if (spec.get("reasoning_pattern") == "macro_first_v2"
            and spec.get("semantic_evidence_contract_version")):
        from environment.generate_question.authoring.semantic_evidence import (
            compile_semantic_contract, semantic_scenario_issues,
        )
        # Validate the *proposed* narrative and source titles, before
        # either LLM review or persistence; frozen Solver remains unchanged.
        if (isinstance(titles, list) and
                all(isinstance(x, dict)
                    and isinstance(x.get("source_id"), str)
                    and isinstance(x.get("title"), str) for x in titles)
                and len({x["source_id"] for x in titles}) == len(titles)):
            proposed = copy.deepcopy(spec)
            if isinstance(narrative, dict):
                proposed["scenario"].update(narrative)
            title_by_id = {x["source_id"]: x["title"] for x in titles}
            if all(role["source_id"] in title_by_id for role in proposed["sources"]):
                for role in proposed["sources"]:
                    role["title"] = title_by_id[role["source_id"]]
                errors.extend(semantic_scenario_issues(
                    proposed, compile_semantic_contract(spec)
                ))
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
    semantic_contract = None
    if spec.get("semantic_evidence_contract_version"):
        from environment.generate_question.authoring.semantic_evidence import compile_semantic_contract
        semantic_contract = compile_semantic_contract(spec)
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
        **({
            "semantic_evidence_contract": semantic_contract,
            "frozen_goal": spec["macro_plan"]["goal"],
            "macro_task_roles": [
                {"role": t["role"], "source_scope": t["source_scope"]}
                for t in spec["macro_plan"]["macro_tasks"]
            ],
            "media_clues": (
                semantic_contract["necessary_question_clues"]
                if semantic_contract else spec["macro_plan"]["clues"]
            ),
        } if spec.get("reasoning_pattern") == "macro_first_v2" else {}),
    }
    from environment.generate_question.authoring.scenario_contracts import (
        has_scene_realism, canonicalize_scenario_proposal,
        scenario_domain_context,
    )
    if has_scene_realism(spec):
        frozen["domain_context"] = scenario_domain_context(spec)
    failures = []
    repair_history = []
    for attempt in range(max_revisions + 1):
        try:
            from environment.generate_question.authoring.prompt_templates import (
                SCENARIO_AUTHOR, SCENARIO_CRITIC, SCENARIO_AUTHOR_SCHEMA_LOCKED,
            )
            v2 = spec.get("reasoning_pattern") == "macro_first_v2"
            scenario_author_system = (
                SCENARIO_AUTHOR_SCHEMA_LOCKED if has_scene_realism(spec)
                else SCENARIO_AUTHOR if v2 else AUTHOR_SYSTEM
            )
            proposal = author.complete_json(scenario_author_system, json.dumps({
                "frozen_contract": frozen,
                "previous_errors": failures[-2:],
            }, ensure_ascii=False))
            # In new V2 scenes, source titles are always regenerated from
            # the frozen modality/schema; LLM-authored titles are ignored.
            proposal = canonicalize_scenario_proposal(spec, proposal)
            if has_scene_realism(spec):
                from environment.generate_question.authoring.scenario_contracts import repair_scenario_fields
                proposal, repairs = repair_scenario_fields(spec, proposal)
                if repairs:
                    repair_history.append({"attempt": attempt + 1,
                                           "field_repairs": repairs})
            issues = _narrative_issues(spec, proposal)
            if issues:
                failures.append({"stage": "schema_guard", "errors": issues})
                continue
            scenario_critic_system = SCENARIO_CRITIC if v2 else CRITIC_SYSTEM
            review = critic.complete_json(scenario_critic_system, json.dumps({
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
                "field_repair_history": repair_history,
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
        "failures": failures, "field_repair_history": repair_history,
        "formal_equivalence_proven": False,
    }
    return fallback, fallback["scenario_authoring"]
