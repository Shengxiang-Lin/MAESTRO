"""Post-TaskGraph Semantic Scenario Contract (v4).

The Constructor interprets an immutable typed TaskGraph and synthetic World.
It NEVER sees the original MMQA question or original MMQA excerpts. The
output is a coherent, explicitly hypothetical *semantic contract* shared by
Question/Task/Trace/Answer authoring, not new observations or a new answer.
Only Python may bind source facts and freeze the final question operation.
"""
from __future__ import annotations

import copy
import json
import re

from .scene_grounding import (
    fingerprint, frozen_fingerprint, source_records,
    evidence_summary, source_ablation_signal,
)
from .semantic_contract import semantic_contract


VERSION = "semantic_scenario_contract_v1"
MODE = "synthetic_semantic_contract_not_media_annotation"

AUTHOR_SYSTEM = """You are a SEMANTIC SCENARIO REALIZER, not a question writer.

A frozen Macro TaskGraph has already been executed on a synthetic World.
Translate its abstract source-bound candidates, flags, scores, groups and
relative order into ONE coherent, realistic HYPOTHETICAL domain situation.

Your input includes:
- the Topic and domain, an immutable Goal/answer contract, and a task-derived
  reference Question (a semantic boundary; do not copy its wording);
- the EXISTING source records with exact source IDs, candidate IDs, fact refs,
  boolean flags, scores, relative indices and group IDs;
- a compact view of actual Macro Task operations, prerequisites and results.

No original MultiModalQA question is supplied. Do not assume the original
image, video or audio content was observed: the source's "image" modality
is only a source handle, NOT a depiction of an object. The World attributes
are explicitly synthetic and DO NOT establish real historical facts.

Construct one unified explanation that binds:
  scenario setting + user role + practical motivation;
  candidate kind + semantic meaning of each existing data field;
  one domain-appropriate display label and honest role per source_id;
  a user information need that matches the PROVIDED goal exactly.

Use general domain knowledge for *fictional scenario plausibility only*,
not for inventing actual event names, years, factual identities, measured
outcomes, additional facts, scores, sources or source media contents.
Do not invent any new operation: e.g. no correction, cross-year research,
new eligibility threshold, inferred image feature, identity resolution or
causal inference unless the immutable Goal explicitly requires it.
"Same group key" means same simulated group, NEVER automatically same person.
"Relative time index" means simulation order, NEVER calendar dates.
A positive flag is a hypothetical recorded mark, NOT a confirmed property
of an unobserved photo, actual real-world award or historical event.
Avoid vague "a reviewer reviews records" scenarios: give a domain-specific
user motivation and clear entry/attribute roles while staying within the
frozen evidence.

If no sensible interpretation exists that uses exactly the existing
candidate fields and goal, say status="not_realizable" and give a specific
reason. Do NOT stretch the evidence to force a positive answer.

Return strictly the JSON schema:
{
 "status":"realizable|not_realizable",
 "reason":"none for realizable; specific reason for not_realizable",
 "scenario":{"setting":"short hypothetical context including topic",
             "user_role":"domain-appropriate user role",
             "motivation":"why the user needs the frozen result"},
 "semantic_bindings":{
     "candidate_kind":"specific fictional domain entry type",
     "score":"meaning of EXISTING synthetic numeric score",
     "positive_flag":"meaning of EXISTING synthetic boolean flag",
     "relative_time_index":"meaning of relative recorded order",
     "group_key":"meaning of existing simulated group"},
 "source_bindings":[
   {"source_id":"exact source ID","candidate_label":"fictional display label",
    "source_role":"what this already authorized source record represents"}
 ],
 "user_need":"one substantive information need supported by existing Goal"
}
For status=not_realizable use empty strings/array for fields outside reason.
Do not add a natural-language final Question or Gold answer.
"""

CRITIC_SYSTEM = """You are an INDEPENDENT SEMANTIC SCENARIO CONTRACT REVIEWER.

Do not rubber-stamp plausible prose. The input World, Goal and TaskGraph
are authoritative and SYNTHETIC. The candidate merely provides a fictional
interpretation, never an assertion that original media was actually read.

Review SIX axes, each yes/no/uncertain:
domain_plausibility: user role, motivation and setting make sense for this
exact Topic and domain; not a generic data review with a topic pasted in.
semantic_coherence: candidate kind, score, flag, time order, group meanings,
source roles and user_need describe ONE coherent hypothetical domain World.
source_faithfulness: all roles are genuine interpretations of the supplied
source inventory, no new or implicitly observed actual image/video/audio
content, historical event, source subtype, entity fact or extra source.
goal_compatibility: SAME existing Goal operation, condition, candidate
universe, target work product, dependencies and answer scope; no new
filter, computation, identity linking, lookup or date interpretation.
evidence_boundary: proposed details are explicitly fictional meaning
assignments, NOT assertions about real named people/events or what an
original photo, video or audio actually shows, detects or confirms.
user_need_alignment: the user's information need genuinely requests
exactly the frozen Goal result; no added domain-specific query conditions.

The task-derived reference Question is an intent constraint, not a
writing template. No original MultiModalQA question is available or
needed. Source-ablation is only a diagnostic, not a Task necessity proof.
Give specific criticisms on REVISE, not generic boilerplate.
PASS iff ALL axes=yes and issues=[]; otherwise revise or uncertain.
No formal proof of semantics, reality or multi-source necessity is claimed.

Return JSON:
{
 "domain_plausibility":"yes|no|uncertain",
 "semantic_coherence":"yes|no|uncertain",
 "source_faithfulness":"yes|no|uncertain",
 "goal_compatibility":"yes|no|uncertain",
 "evidence_boundary":"yes|no|uncertain",
 "user_need_alignment":"yes|no|uncertain",
 "verdict":"pass|revise|uncertain",
 "issues":["concrete finding, empty for pass"],
 "summary":"concise concrete assessment"
}
"""

_BINDING_KEYS = (
    "candidate_kind", "score", "positive_flag",
    "relative_time_index", "group_key",
)
_SCENARIO_KEYS = ("setting", "user_role", "motivation")
_PROPOSAL_KEYS = (
    "status", "reason", "scenario", "semantic_bindings",
    "source_bindings", "user_need",
)
_ROLE_KEYS = ("source_id", "candidate_label", "source_role")
REVIEW_AXES = (
    "domain_plausibility", "semantic_coherence", "source_faithfulness",
    "goal_compatibility", "evidence_boundary", "user_need_alignment",
)

PROPOSAL_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {"enum": ["realizable", "not_realizable"]},
        "reason": {"type": "string"},
        "scenario": {
            "type": "object",
            "properties": {k: {"type": "string"} for k in _SCENARIO_KEYS},
            "required": list(_SCENARIO_KEYS),
            "additionalProperties": False,
        },
        "semantic_bindings": {
            "type": "object",
            "properties": {k: {"type": "string"} for k in _BINDING_KEYS},
            "required": list(_BINDING_KEYS),
            "additionalProperties": False,
        },
        "source_bindings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {k: {"type": "string"} for k in _ROLE_KEYS},
                "required": list(_ROLE_KEYS),
                "additionalProperties": False,
            },
        },
        "user_need": {"type": "string"},
    },
    "required": list(_PROPOSAL_KEYS),
    "additionalProperties": False,
}
REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        **{axis: {"enum": ["yes", "no", "uncertain"]}
           for axis in REVIEW_AXES},
        "verdict": {"enum": ["pass", "revise", "uncertain"]},
        "issues": {"type": "array", "items": {"type": "string"}},
        "summary": {"type": "string"},
    },
    "required": [*REVIEW_AXES, "verdict", "issues", "summary"],
    "additionalProperties": False,
}

def _base_scene(scene: dict) -> dict:
    """Exclude the overlay when deriving the frozen independent Goal."""
    return {key: value for key, value in scene.items()
            if key != "scene_grounding"}


def frozen_question_intent(scene: dict) -> dict:
    """Not LLM generated. The authoritative, task-derived answer contract."""
    spec = semantic_contract(_base_scene(scene))
    return {
        "goal_id": spec["goal_id"],
        "operation": spec["operation"],
        "target_universe": spec["target_universe"],
        "required_fields": spec["allowed_fields"],
        "hard_constraints": spec["hard_constraints"],
        "required_answer": spec["required_answer"],
        "final_workproduct_type": spec["final_workproduct_type"],
        "source_ids": [row["source_id"] for row in spec["source_inventory"]],
    }


def source_fact_view(scene: dict) -> list[dict]:
    """No original MMQA text snippets; only currently executable facts."""
    result = []
    for row in source_records(scene):
        result.append({
            "source_id": row["source_id"],
            "modality": row["modality"],
            "representation": "synthetic_source_role_not_perceived_media",
            "fields": copy.deepcopy(row["fields"]),
        })
    return result


def task_operation_view(scene: dict) -> list[dict]:
    """One compact operation/actual outcome per Macro Task."""
    outcome = []
    for task in evidence_summary(scene):
        result = task["computed_result"]
        detail = {
            key: copy.deepcopy(value)
            for key, value in result.items() if key in {
                "candidate_count", "source_count", "selected_count",
                "total_count", "top_candidate", "ranked_candidates",
                "ordered_candidates", "earliest_time_index",
                "matched_groups", "synthetic_groups", "eligible_candidates",
                "priority_order", "candidate_selection", "answer_status",
            }
        }
        outcome.append({
            "task_number": task["task_number"],
            "operation": task["operation"],
            "work_type": task["work_type"],
            "deliverable_type": task["deliverable_type"],
            "prerequisites": task["prerequisites"],
            "source_ids": task["source_ids"],
            "result": detail,
        })
    return outcome


def constructor_input(scene: dict) -> dict:
    return {
        "topic": scene["scenario"]["topic"],
        "domain": scene["scenario"]["domain"],
        "goal_contract": frozen_question_intent(scene),
        "reference_question": scene["question"],
        "source_records": source_fact_view(scene),
        "task_operations": task_operation_view(scene),
        "source_ablation_diagnostic": source_ablation_signal(scene),
        "evidence_boundary": (
            "Synthetic source facts only: never infer original MMQA "
            "question details, actual images, historical facts or "
            "unprovided real media observations."
        ),
    }


def _str(value, maximum=360) -> bool:
    return (isinstance(value, str) and 2 <= len(value.strip()) <= maximum
            and "\n" not in value and "\r" not in value)


def _prose(proposal: dict) -> str:
    return " ".join(
        str(x) for x in (
            *proposal["scenario"].values(),
            *proposal["semantic_bindings"].values(),
            proposal["user_need"],
            *(item["source_role"] for item in proposal["source_bindings"]),
            *(item["candidate_label"] for item in proposal["source_bindings"]),
        )
    )


def validate_contract_proposal(scene: dict, proposal: dict) -> list[str]:
    """Mechanically check what can be checked; leave plausibility to Critic."""
    if not isinstance(proposal, dict) or set(proposal) != set(_PROPOSAL_KEYS):
        return ["invalid_contract_schema"]
    if proposal.get("status") not in {"realizable", "not_realizable"}:
        return ["invalid_realizability_status"]
    if proposal["status"] == "not_realizable":
        reason = proposal.get("reason")
        return [] if _str(reason, 650) else ["missing_not_realizable_reason"]
    errors = []
    if not isinstance(proposal.get("reason"), str) or (
            proposal["reason"].strip().lower() not in {"none", "n/a"}):
        errors.append("unexpected_rejection_reason")
    scenario = proposal.get("scenario")
    bindings = proposal.get("semantic_bindings")
    roles = proposal.get("source_bindings")
    if not isinstance(scenario, dict) or set(scenario) != set(_SCENARIO_KEYS):
        return ["invalid_scenario_schema"]
    if not isinstance(bindings, dict) or set(bindings) != set(_BINDING_KEYS):
        return ["invalid_semantic_bindings"]
    if not isinstance(roles, list):
        return ["invalid_source_bindings"]
    for key in _SCENARIO_KEYS:
        if not _str(scenario[key], maximum=550):
            errors.append("invalid_scenario_" + key)
    for key in _BINDING_KEYS:
        if not _str(bindings[key], maximum=360):
            errors.append("invalid_binding_" + key)
    if not _str(proposal.get("user_need"), maximum=450):
        errors.append("invalid_user_need")
    original = {row["source_id"] for row in source_fact_view(scene)}
    supplied = []
    labels = set()
    for role in roles:
        if not isinstance(role, dict) or set(role) != set(_ROLE_KEYS):
            errors.append("invalid_source_binding")
            continue
        sid = role["source_id"]
        if not isinstance(sid, str):
            errors.append("invalid_source_id")
            continue
        supplied.append(sid)
        for field in ("candidate_label", "source_role"):
            if not _str(role[field], maximum=300):
                errors.append("invalid_" + field)
        label = str(role.get("candidate_label")).strip().casefold()
        if label in labels:
            errors.append("duplicate_candidate_labels")
        labels.add(label)
    if len(supplied) != len(original) or set(supplied) != original:
        errors.append("source_inventory_changed")
    if len(supplied) != len(set(supplied)):
        errors.append("duplicate_source_id")
    # Require that the concrete situation actually names its topic, rather
    # than attaching an irrelevant original-MMQA story to an arbitrary Goal.
    topic = str(scene["scenario"]["topic"]).strip()
    context = " ".join([
        str(scenario.get("setting", "")),
        str(scenario.get("motivation", "")),
        str(proposal.get("user_need", "")),
    ])
    if topic and topic.casefold() not in context.casefold():
        errors.append("topic_not_anchored")
    if all("record" in str(scenario[key]).casefold()
           or "review" in str(scenario[key]).casefold()
           for key in _SCENARIO_KEYS):
        errors.append("generic_record_review_story")
    # Minimal operation-family guard. This catches obvious goal drift before
    # the model Critic; it is deliberately permissive about natural wording.
    intent_terms = {
        "lookup": r"\b(?:which|identify|find|select|list|marked|positive)\b",
        "aggregation": r"\b(?:how many|count|number|total)\b",
        "comparison": r"\b(?:compare|rank|order|highest|top|largest|greatest)\b",
        "intersection": r"\b(?:overlap|shared|common|both|recur|match|intersection)\b",
        "composition": r"\b(?:connect|link|relationship|correspond|reference|map|relat)\w*\b",
        "temporal": r"\b(?:earliest|order|sequence|first|before|after|timeline)\b",
        "prioritization": r"\b(?:priorit|rank|order|highest|descending|top)\w*\b",
        "feasibility": r"\b(?:qualif|eligible|eligib|satisf|meet|threshold|at least)\w*\b",
    }
    goal_id = scene["goal"]["intent"]
    if (goal_id in intent_terms and isinstance(proposal.get("user_need"), str)
            and not re.search(intent_terms[goal_id],
                              proposal["user_need"], re.I)):
        errors.append("user_need_operation_missing")
    if not errors:
        prose = _prose(proposal)
        if re.search(
            r"\b(?:visually confirmed|visual confirmation|"
            r"(?:detected|observed|verified)\s+(?:in|from|by)\s+"
            r"(?:the\s+)?(?:image|photo|poster|screenshot|video)|"
            r"(?:image|photo|video)\s+(?:shows|depicts|confirms))\b",
            prose, re.I,
        ):
            errors.append("unsupported_media_perception")
        if re.search(
            r"\b(?:historically|actually|in real life|verified real)\b",
            prose, re.I,
        ):
            errors.append("real_world_assertion")
        if re.search(
            r"\b(?:18|19|20)\d{2}\b",
            prose.replace(topic, ""), re.I,
        ):
            errors.append("unsupported_calendar_year")
    return sorted(set(errors))


def validate_contract_review(review: dict) -> tuple[dict, list[str]]:
    if not isinstance(review, dict) or set(review) != set(REVIEW_SCHEMA["required"]):
        return {}, ["invalid_contract_critic_schema"]
    errors = []
    for axis in REVIEW_AXES:
        if review.get(axis) not in {"yes", "no", "uncertain"}:
            errors.append("invalid_contract_critic_axis:" + axis)
    if (review.get("verdict") not in {"pass", "revise", "uncertain"}
            or not isinstance(review.get("issues"), list)
            or any(not isinstance(item, str) or not item.strip()
                   for item in review.get("issues", []))
            or not _str(review.get("summary"), maximum=1500)):
        errors.append("invalid_contract_critic_fields")
    if errors:
        return review, errors
    verdict = (
        "uncertain" if any(review[axis] == "uncertain" for axis in REVIEW_AXES)
        else "revise" if (any(review[axis] == "no" for axis in REVIEW_AXES)
                          or review["issues"])
        else "pass"
    )
    return {**review, "verdict": verdict}, []


def _canonical(proposal: dict) -> dict:
    canonical = copy.deepcopy(proposal)
    canonical["source_bindings"].sort(key=lambda row: row["source_id"])
    return canonical


def materialize_contract(scene: dict, proposal: dict, review: dict) -> dict:
    errors = validate_contract_proposal(scene, proposal)
    if errors or proposal["status"] != "realizable":
        raise ValueError("cannot_materialize_contract:" + ",".join(errors))
    original = source_fact_view(scene)
    semantic = copy.deepcopy(proposal["semantic_bindings"])
    roles = {row["source_id"]: row for row in proposal["source_bindings"]}
    source_cards = []
    for row in original:
        sid = row["source_id"]
        source_cards.append({
            "source_id": sid,
            "modality": row["modality"],
            "candidate_key": row["fields"]["entry"]["value"],
            "candidate_label": roles[sid]["candidate_label"],
            "source_role": roles[sid]["source_role"],
            "fact_bindings": copy.deepcopy(row["fields"]),
            "fact_origin": "deterministic_synthetic_world_not_original_mmqa",
        })
    return {
        "version": VERSION,
        "mode": MODE,
        "status": "accepted",
        "topic": scene["scenario"]["topic"],
        "scenario": copy.deepcopy(proposal["scenario"]),
        "semantic_bindings": semantic,
        "source_cards": source_cards,
        "user_need": proposal["user_need"],
        "question_intent": frozen_question_intent(scene),
        "review": copy.deepcopy(review),
        "original_world_fingerprint": fingerprint(scene["world"]),
        "frozen_solution_fingerprint": frozen_fingerprint(scene),
        "proposal_fingerprint": fingerprint(_canonical(proposal)),
        "validation": {
            "source_coverage": "exact",
            "fact_values": "bound_from_frozen_world",
            "operation_replay": "original_unchanged",
            "source_grounding": "hypothetical_semantics_not_real_media",
            "llm_review_not_formal_proof": True,
            "goal_level_source_ablation": source_ablation_signal(scene),
        },
    }


def _proposal_from_contract(contract: dict) -> dict:
    return {
        "status": "realizable",
        "reason": "none",
        "scenario": copy.deepcopy(contract.get("scenario")),
        "semantic_bindings": copy.deepcopy(contract.get("semantic_bindings")),
        "source_bindings": [
            {
                "source_id": item.get("source_id"),
                "candidate_label": item.get("candidate_label"),
                "source_role": item.get("source_role"),
            }
            for item in contract.get("source_cards", [])
            if isinstance(item, dict)
        ],
        "user_need": contract.get("user_need"),
    }


def validate_scenario_contract(scene: dict) -> list[str]:
    """Re-materialize accepted v4 overlay from frozen World/Goal, compare all."""
    contract = scene.get("scene_grounding")
    if not isinstance(contract, dict):
        return ["invalid_scenario_contract"]
    if (contract.get("version") != VERSION
            or contract.get("mode") != MODE
            or contract.get("status") != "accepted"
            or contract.get("topic") != scene["scenario"]["topic"]):
        return ["scenario_contract_identity_mismatch"]
    errors = []
    proposal = _proposal_from_contract(contract)
    errors.extend(validate_contract_proposal(scene, proposal))
    if errors:
        return sorted(set(errors))
    review, reviewer_errors = validate_contract_review(contract.get("review"))
    errors.extend(reviewer_errors)
    if review.get("verdict") != "pass" or any(
            review.get(axis) != "yes" for axis in REVIEW_AXES):
        errors.append("scenario_contract_review_not_passed")
    if contract.get("proposal_fingerprint") != fingerprint(_canonical(proposal)):
        errors.append("scenario_contract_proposal_drift")
    if contract.get("original_world_fingerprint") != fingerprint(scene["world"]):
        errors.append("scenario_contract_world_drift")
    if contract.get("frozen_solution_fingerprint") != frozen_fingerprint(scene):
        errors.append("scenario_contract_solution_drift")
    if contract.get("question_intent") != frozen_question_intent(scene):
        errors.append("scenario_contract_goal_drift")
    if not errors:
        expected = materialize_contract(scene, proposal, contract["review"])
        if expected != contract:
            errors.append("scenario_contract_materialization_drift")
    return sorted(set(errors))


def ground_one_contract(scene: dict, *, constructor, critic,
                        max_revisions: int = 2) -> tuple[dict | None, dict]:
    """Independent LLM semantic authoring + deterministic checks + critic."""
    if "scene_grounding" in scene:
        raise ValueError("scene_already_grounded")
    request = constructor_input(scene)
    attempts = []
    previous = None
    feedback = None
    for revision in range(max_revisions + 1):
        user = dict(request)
        if previous is not None:
            user["previous_proposal"] = previous
            user["reviewer_feedback"] = feedback
        elif feedback is not None:
            user["reviewer_feedback"] = feedback
        entry = {"round": revision + 1}
        try:
            proposal = constructor.complete_json(
                AUTHOR_SYSTEM, json.dumps(user, ensure_ascii=False),
                response_schema=PROPOSAL_SCHEMA)
            entry["proposal"] = proposal
            local_issues = validate_contract_proposal(scene, proposal)
            if local_issues:
                entry["validation_errors"] = local_issues
                entry["failure"] = "invalid_semantic_contract"
                previous, feedback = proposal, {"issues": local_issues}
                attempts.append(entry)
                continue
            if proposal["status"] == "not_realizable":
                entry["failure"] = "not_realizable"
                attempts.append(entry)
                return None, {
                    "status": "not_realizable",
                    "reason": proposal["reason"],
                    "attempts": attempts,
                    "revision_rounds": revision,
                    "author_calls": len(attempts),
                    "critic_calls": sum("critic_review" in a for a in attempts),
                }
            review_payload = {
                "proposed_scenario_contract": proposal,
                "goal_contract": request["goal_contract"],
                "reference_question": request["reference_question"],
                "source_records": request["source_records"],
                "task_operations": request["task_operations"],
                "source_ablation_diagnostic": request["source_ablation_diagnostic"],
                "evidence_boundary": request["evidence_boundary"],
            }
            review_raw = critic.complete_json(
                CRITIC_SYSTEM,
                json.dumps(review_payload, ensure_ascii=False),
                response_schema=REVIEW_SCHEMA)
            review, review_issues = validate_contract_review(review_raw)
            entry["critic_review"] = review
            if review_issues:
                entry["validation_errors"] = review_issues
                entry["failure"] = "invalid_semantic_critic_response"
                previous, feedback = proposal, {"issues": review_issues}
                attempts.append(entry)
                continue
            if review["verdict"] == "pass":
                overlay = materialize_contract(scene, proposal, review)
                verify = validate_scenario_contract({
                    **scene, "scene_grounding": overlay})
                if verify:
                    entry["validation_errors"] = verify
                    entry["failure"] = "scenario_contract_replay_failure"
                    previous, feedback = proposal, {"issues": verify}
                    attempts.append(entry)
                    continue
                entry["accepted"] = True
                attempts.append(entry)
                return overlay, {
                    "status": "accepted",
                    "reason": "typed_contract_and_semantic_critic_pass",
                    "attempts": attempts,
                    "revision_rounds": revision,
                    "author_calls": len(attempts),
                    "critic_calls": sum("critic_review" in a for a in attempts),
                }
            entry["failure"] = "critic_" + review["verdict"]
            previous, feedback = proposal, review
            attempts.append(entry)
            if review["verdict"] == "uncertain":
                break
        except Exception as exc:
            entry["failure"] = "model_or_protocol_error"
            entry["detail"] = str(exc)[:260]
            attempts.append(entry)
            if "did not return a JSON object" in str(exc):
                previous = None
                feedback = {"issues": [
                    "Return a strict JSON object matching the output schema."
                ]}
                continue
            break
    return None, {
        "status": "rejected",
        "reason": attempts[-1].get("failure", "invalid_contract")
                  if attempts else "no_attempt",
        "last_validation_errors": (
            attempts[-1].get("validation_errors", []) if attempts else []),
        "attempts": attempts,
        "revision_rounds": max(0, len(attempts)-1),
        "author_calls": len(attempts),
        "critic_calls": sum("critic_review" in a for a in attempts),
    }
