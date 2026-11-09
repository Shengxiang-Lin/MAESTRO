"""Semantic-first synthetic World construction, after the frozen Macro Plan.

No original MultiModalQA Question or raw media observations enter the LLM.
The LLM only designs a hypothetical *schema and source roles*; Python
samples numeric World observations AFTER that specification is reviewed.
World facts, EvidenceGraphs and Feedback must be rebuilt and replayed.
"""
from __future__ import annotations

import copy
import json
import re

from envir.schema import digest
from .goal_grammar import GOALS
from .semantic_contract import GOAL_SEMANTICS
from .scene_grounding import fingerprint

VERSION = "semantic_world_spec_v1"
BINDING_FIELDS = (
    "candidate_kind", "score", "positive_flag",
    "relative_time_index", "group_key",
)
SCENARIO_FIELDS = ("subject_type", "setting", "user_role", "motivation")
SOURCE_FIELDS = ("source_id", "candidate_label", "source_role")
REVIEW_AXES = (
    "topic_coherence", "field_coherence", "source_faithfulness",
    "goal_compatibility", "no_unsupported_media_or_history",
)

AUTHOR_SYSTEM = """You are designing a SYNTHETIC semantic World schema BEFORE
any source scores, flags, timestamps, groups or candidate IDs are sampled.

The Macro TaskGraph, existing source handles, and Goal are already frozen.
Your role is to make a concise, domain-relevant fictional problem setting,
including the meanings of existing candidate/score/flag/relative-order/group
fields and the role of EACH already authorized source handle.

IMPORTANT: This is NOT grounding real photographs, events or MMQA answers.
You will NOT see any original MMQA Question, any original MMQA source text,
or any simulated score/flag values. You must NOT infer these contents.

The topic label is the strongest domain clue. A domain_hint may be WRONG,
especially if the original MMQA question mixed several domains. Do not
pretend a real film is a sports league or a named song is a corporation.
Use general knowledge ONLY to decide a natural hypothetical subject type,
not to assert real identities, dates, award winners, media contents or events.

The frozen Goal supports ONLY abstract operations, such as count marked
records, rank numeric scores, filter two known conditions, order relative
record indices, or match recorded group keys. Select domain-meaningful
fictional attributes that preserve these exact operations and values:
- numeric score must represent a synthetic 0..100-scale internal rating,
  not real goals, box office, release years, prices or historical ranks;
- positive_flag is a simulated recorded binary status, not "visually
  confirmed" original media contents;
- relative_time_index is a synthetic ordering index, not a real date;
- group_key is a grouping category, NOT an identity resolution oracle;
- no new source, operation, filter, threshold, entity match, correction,
  factual observation, historical claim or causal relation may be invented;
- original image/video/audio modalities are source HANDLE categories only,
  never proof that any picture was viewed or a scene was observed.
The new user_need must retain exactly the supplied Goal; an EMPTY
intersection/feasibility result is perfectly valid. Never reject just
because no item might satisfy a condition: values have NOT been sampled yet.

All source_id values must be copied EXACTLY. Give each source a distinctive
fictional record label and plausible role consistent with its modality,
without pretending original media was perceived. Prefer concrete fictional
workflows over the same boilerplate "review records" story. The resulting
scenario, field semantics, source roles and need MUST form ONE consistent
World. Do not output sampled values or a final answer.

Only when no semantically coherent synthetic interpretation exists despite
these allowed fictional roles, set status="not_realizable" with a reason.
For not_realizable set empty strings/array for all other fields.

Return exactly JSON:
{
 "status":"realizable|not_realizable",
 "reason":"none for realizable, specific explanation otherwise",
 "scenario":{"subject_type":"film/award/book/event/etc.",
    "setting":"one-sentence hypothetical scene about topic",
    "user_role":"specific user role",
    "motivation":"domain-specific reason for this existing Goal"},
 "semantic_bindings":{"candidate_kind":"fictional candidate type",
    "score":"name/meaning of 0..100 internal score",
    "positive_flag":"name/meaning of recorded binary marker",
    "relative_time_index":"name/meaning of ordinal record index",
    "group_key":"name/meaning of grouping/category"},
 "source_bindings":[{"source_id":"exact provided source ID",
    "candidate_label":"unique fictional record label",
    "source_role":"honest simulated source role"}],
 "user_need":"concrete user question intent; same operation as frozen Goal"
}
"""

CRITIC_SYSTEM = """You are an independent SCHEMA-FIRST SYNTHETIC WORLD
reviewer. You are given a frozen Macro Goal, topic, source inventory, task
operations and a proposed fictional semantic schema. NO numeric World exists
yet; therefore you CANNOT judge a proposed schema by resulting counts or
whether any future candidate happens to be eligible. Empty answers are valid.

Assess five axes:
topic_coherence: does this hypothetical subject type actually fit the
topic label? The domain_hint is WEAK and may be factually misleading.
field_coherence: do candidate kind, internal 0..100 score, binary marker,
relative ordinal index and group category form ONE coherent fictional world?
source_faithfulness: exactly the existing source modalities/IDs, nothing
that implies original media were examined or a new source was acquired.
goal_compatibility: the proposed user_need only requires the frozen
operation/threshold/candidate scope/output, no extra data or computations.
no_unsupported_media_or_history: no claims of real events, observed image
details, named historical outcomes, actual dates or genuine measured stats.

Synthetic examples and fictional category names are permitted. Do not
reject merely because the scenario is hypothetical; that is intentional.
Do not demand that all source records have positive marks or overlapping
group IDs. Explicitly reject story-world mismatches (film as sports team),
generic meaningless scenes, and ungrounded extra operations.

Use 'yes'|'no'|'uncertain' on each axis. Pass iff all are yes and issues=[].
Return strictly JSON:
{"topic_coherence":"yes|no|uncertain",
 "field_coherence":"yes|no|uncertain",
 "source_faithfulness":"yes|no|uncertain",
 "goal_compatibility":"yes|no|uncertain",
 "no_unsupported_media_or_history":"yes|no|uncertain",
 "verdict":"pass|revise|uncertain",
 "issues":["concrete reason if rejected"],
 "summary":"concise factual assessment"}
"""

PROPOSAL_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {"enum": ["realizable", "not_realizable"]},
        "reason": {"type": "string"},
        "scenario": {
            "type": "object",
            "properties": {k: {"type": "string"} for k in SCENARIO_FIELDS},
            "required": list(SCENARIO_FIELDS), "additionalProperties": False,
        },
        "semantic_bindings": {
            "type": "object",
            "properties": {k: {"type": "string"} for k in BINDING_FIELDS},
            "required": list(BINDING_FIELDS), "additionalProperties": False,
        },
        "source_bindings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {k: {"type": "string"} for k in SOURCE_FIELDS},
                "required": list(SOURCE_FIELDS),
                "additionalProperties": False,
            },
        },
        "user_need": {"type": "string"},
    },
    "required": ["status", "reason", "scenario", "semantic_bindings",
                 "source_bindings", "user_need"],
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


def source_manifest_from_plan(plan: dict) -> list[dict]:
    """Extract existing source identity/role WITHOUT consulting the old World."""
    seen = {}
    for task in plan["macro_graph"]["tasks"]:
        for source in task.get("source_requirements", []):
            sid = "src-" + digest(
                plan["task_id"], source["role"], source["description"])
            row = {
                "source_id": sid,
                "modality": source["candidate_modalities"][0],
                "original_doc_id": source.get("original_doc_id"),
                "source_origin": source.get("source_origin"),
                "task_source_role": source["role"],
            }
            if sid in seen and seen[sid] != row:
                raise ValueError("conflicting_plan_source_role:" + sid)
            seen[sid] = row
    if not seen:
        raise ValueError("plan_has_no_source_handles")
    return [seen[sid] for sid in sorted(seen)]


def goal_contract_from_plan(plan: dict) -> dict:
    intent = plan["goal"]["intent"]
    op, required, constraints, result = GOAL_SEMANTICS[intent]
    return {
        "goal_id": intent,
        "operation": op,
        "required_fields": list(required),
        "hard_constraints": list(constraints),
        "expected_result": result,
        "artifact_type": GOALS[intent]["artifact"],
        "source_universe": [
            item["source_id"] for item in source_manifest_from_plan(plan)
        ],
    }


def plan_input(plan: dict) -> dict:
    """Input deliberately excludes old World, original MMQA QA and snippets."""
    tasks = plan["macro_graph"]["tasks"]
    by_id = {t["task_id"]: i + 1 for i, t in enumerate(tasks)}
    return {
        "topic": plan["scenario"]["topic"],
        "domain_hint_weak": plan["scenario"]["domain"],
        "goal_contract": goal_contract_from_plan(plan),
        "source_manifest": source_manifest_from_plan(plan),
        "task_operations": [{
            "task_number": by_id[t["task_id"]],
            "operation": t["operation_spec"]["op"],
            "work_type": t["work_type"],
            "depends_on": [by_id[tid] for tid in t["depends_on"]],
            "output_types": sorted(t["outputs"].values()),
        } for t in tasks],
        "simulated_media_notice": (
            "Only source handles; no original image/audio/video decoded. "
            "All values will be generated after semantic specification."
        ),
    }


def _short(value, limit=440) -> bool:
    return isinstance(value, str) and 2 <= len(value.strip()) <= limit and (
        "\n" not in value and "\r" not in value)


def _all_prose(proposal: dict) -> str:
    return " ".join([
        *proposal["scenario"].values(),
        *proposal["semantic_bindings"].values(),
        proposal["user_need"],
    ] + [
        entry[field] for entry in proposal["source_bindings"]
        for field in ("candidate_label", "source_role")
    ])


def validate_proposal(plan: dict, proposal: dict) -> list[str]:
    """Strict source/field schema and provenance checks, not keyword scoring."""
    if not isinstance(proposal, dict) or set(proposal) != set(PROPOSAL_SCHEMA["required"]):
        return ["invalid_semantic_world_schema"]
    if proposal.get("status") not in {"realizable", "not_realizable"}:
        return ["invalid_realizability"]
    if proposal["status"] == "not_realizable":
        return [] if _short(proposal.get("reason"), 700) else ["missing_not_realizable_reason"]
    errors = []
    if (not isinstance(proposal.get("reason"), str)
            or proposal["reason"].strip().lower() not in {"none", "n/a"}):
        errors.append("unexpected_reason")
    scene = proposal.get("scenario")
    meanings = proposal.get("semantic_bindings")
    roles = proposal.get("source_bindings")
    if not isinstance(scene, dict) or set(scene) != set(SCENARIO_FIELDS):
        return ["invalid_semantic_scene_schema"]
    if not isinstance(meanings, dict) or set(meanings) != set(BINDING_FIELDS):
        return ["invalid_semantic_field_schema"]
    if not isinstance(roles, list):
        return ["invalid_semantic_source_roles"]
    for name in SCENARIO_FIELDS:
        if not _short(scene[name], 650):
            errors.append("invalid_scene_" + name)
    for name in BINDING_FIELDS:
        if not _short(meanings[name], 360):
            errors.append("invalid_meaning_" + name)
    if not _short(proposal.get("user_need"), 500):
        errors.append("invalid_user_need")
    source_ids = {r["source_id"] for r in source_manifest_from_plan(plan)}
    listed, labels = [], set()
    for row in roles:
        if not isinstance(row, dict) or set(row) != set(SOURCE_FIELDS):
            errors.append("invalid_source_binding")
            continue
        sid = row["source_id"]
        if not isinstance(sid, str):
            errors.append("invalid_source_id")
            continue
        listed.append(sid)
        for name in ("candidate_label", "source_role"):
            if not _short(row[name], 320):
                errors.append("invalid_" + name)
        label = str(row.get("candidate_label")).casefold().strip()
        if label in labels:
            errors.append("duplicate_candidate_label")
        labels.add(label)
    if set(listed) != source_ids or len(listed) != len(source_ids):
        errors.append("source_inventory_changed")
    if len(set(listed)) != len(listed):
        errors.append("duplicate_source_id")
    if errors:
        return sorted(set(errors))
    prose = _all_prose(proposal)
    # Reject explicit claims of perception, not legitimate hypothetical roles.
    if re.search(
        r"\b(?:visual(?:ly)?\s+confirm(?:ed|ation)|"
        r"(?:seen|observed|verified|detected)\s+(?:in|on|from)\s+"
        r"(?:the\s+)?(?:photo|image|video|poster|screenshot)|"
        r"(?:photo|image|video)\s+(?:shows|depicts|proves))\b",
        prose, re.I,
    ):
        errors.append("unsupported_actual_media_perception")
    if re.search(r"\b(?:historically\s+(?:true|accurate|confirmed)|"
                 r"real[- ]world\s+(?:winner|result|score)|"
                 r"verified\s+(?:historical|actual))\b",
                 prose, re.I):
        errors.append("unsupported_historical_claim")
    return sorted(set(errors))


def validate_review(review: dict) -> tuple[dict, list[str]]:
    if not isinstance(review, dict) or set(review) != set(REVIEW_SCHEMA["required"]):
        return {}, ["invalid_semantic_world_critic_schema"]
    errors = []
    for axis in REVIEW_AXES:
        if review.get(axis) not in {"yes", "no", "uncertain"}:
            errors.append("invalid_critic_axis:" + axis)
    if (review.get("verdict") not in {"pass", "revise", "uncertain"}
            or not isinstance(review.get("issues"), list)
            or any(not _short(x, 600) for x in review.get("issues", []))
            or not _short(review.get("summary"), 1500)):
        errors.append("invalid_semantic_world_critic_fields")
    if errors:
        return review, errors
    expected = (
        "uncertain" if any(review[key] == "uncertain" for key in REVIEW_AXES)
        else "revise" if (any(review[key] == "no" for key in REVIEW_AXES)
                          or review["issues"])
        else "pass"
    )
    return {**review, "verdict": expected}, []


def canonical_proposal(proposal: dict) -> dict:
    normalized = copy.deepcopy(proposal)
    normalized["source_bindings"] = sorted(
        normalized["source_bindings"], key=lambda row: row["source_id"])
    return normalized


def materialize_spec(plan: dict, proposal: dict, review: dict) -> dict:
    errors = validate_proposal(plan, proposal)
    if errors or proposal["status"] != "realizable":
        raise ValueError("invalid_semantic_spec:" + ",".join(errors))
    return {
        "version": VERSION,
        "mode": "hypothetical_schema_before_world_values",
        "status": "accepted",
        "topic": plan["scenario"]["topic"],
        "goal_contract": goal_contract_from_plan(plan),
        "scenario": copy.deepcopy(proposal["scenario"]),
        "semantic_bindings": copy.deepcopy(proposal["semantic_bindings"]),
        "source_bindings": copy.deepcopy(canonical_proposal(proposal)["source_bindings"]),
        "user_need": proposal["user_need"],
        "review": copy.deepcopy(review),
        "proposal_fingerprint": fingerprint(canonical_proposal(proposal)),
        "graph_fingerprint": fingerprint(plan["macro_graph"]),
        "source_manifest_fingerprint": fingerprint(source_manifest_from_plan(plan)),
        "generation_policy": {
            "source_facts_generated_after_spec": True,
            "original_mmqa_question_seen_by_model": False,
            "original_mmqa_media_perceived": False,
            "source_values_model_generated": False,
        },
    }


def proposal_from_spec(spec: dict) -> dict:
    return {
        "status": "realizable", "reason": "none",
        "scenario": copy.deepcopy(spec.get("scenario")),
        "semantic_bindings": copy.deepcopy(spec.get("semantic_bindings")),
        "source_bindings": copy.deepcopy(spec.get("source_bindings")),
        "user_need": spec.get("user_need"),
    }


def validate_spec(plan: dict, spec: dict) -> list[str]:
    """Check a stored spec against immutable Plan; replay checks World too."""
    if not isinstance(spec, dict) or spec.get("version") != VERSION or (
            spec.get("mode") != "hypothetical_schema_before_world_values"):
        return ["invalid_semantic_world_spec_version"]
    if spec.get("status") != "accepted" or (
            spec.get("topic") != plan["scenario"]["topic"]):
        return ["semantic_world_spec_identity_changed"]
    proposal = proposal_from_spec(spec)
    errors = validate_proposal(plan, proposal)
    if errors:
        return errors
    review, issues = validate_review(spec.get("review"))
    errors.extend(issues)
    if review.get("verdict") != "pass" or any(
            review.get(axis) != "yes" for axis in REVIEW_AXES):
        errors.append("semantic_world_review_not_passed")
    if spec.get("graph_fingerprint") != fingerprint(plan["macro_graph"]):
        errors.append("semantic_world_graph_drift")
    if spec.get("source_manifest_fingerprint") != fingerprint(
            source_manifest_from_plan(plan)):
        errors.append("semantic_world_manifest_drift")
    if spec.get("goal_contract") != goal_contract_from_plan(plan):
        errors.append("semantic_world_goal_drift")
    if spec.get("proposal_fingerprint") != fingerprint(
            canonical_proposal(proposal)):
        errors.append("semantic_world_proposal_drift")
    if errors:
        return sorted(set(errors))
    try:
        expected = materialize_spec(plan, proposal, spec["review"])
        if expected != spec:
            errors.append("semantic_world_spec_materialization_changed")
    except (ValueError, KeyError, TypeError):
        errors.append("semantic_world_spec_materialization_failed")
    return sorted(set(errors))


def realize_spec_one(plan: dict, *, author, critic, max_revisions=2):
    """Schema first, before any numeric World values or EvidenceGraphs exist."""
    if not 0 <= max_revisions <= 8:
        raise ValueError("max_revisions must be 0..8")
    request = plan_input(plan)
    attempts, previous, feedback = [], None, None
    for revision in range(max_revisions + 1):
        payload = dict(request)
        if previous is not None:
            payload["previous_proposal"] = previous
        if feedback is not None:
            payload["reviewer_feedback"] = feedback
        attempt = {"round": revision + 1}
        try:
            proposal = author.complete_json(
                AUTHOR_SYSTEM, json.dumps(payload, ensure_ascii=False),
                response_schema=PROPOSAL_SCHEMA)
            attempt["proposal"] = proposal
            issues = validate_proposal(plan, proposal)
            if issues:
                attempt.update(failure="schema_or_provenance", issues=issues)
                previous, feedback = proposal, {"issues": issues}
                attempts.append(attempt)
                continue
            if proposal["status"] == "not_realizable":
                attempt["failure"] = "not_realizable"
                attempts.append(attempt)
                return None, {
                    "status": "not_realizable", "reason": proposal["reason"],
                    "attempts": attempts, "author_calls": len(attempts),
                    "critic_calls": sum("review" in a for a in attempts),
                    "revision_rounds": revision,
                }
            payload = {
                "proposal": proposal, **request,
                "focus": "Semantic coherency BEFORE synthetic values are sampled.",
            }
            review_raw = critic.complete_json(
                CRITIC_SYSTEM, json.dumps(payload, ensure_ascii=False),
                response_schema=REVIEW_SCHEMA)
            review, issues = validate_review(review_raw)
            attempt["review"] = review
            if issues:
                attempt.update(failure="critic_protocol", issues=issues)
                previous, feedback = proposal, {"issues": issues}
                attempts.append(attempt)
                continue
            if review["verdict"] == "pass":
                spec = materialize_spec(plan, proposal, review)
                check = validate_spec(plan, spec)
                if check:
                    attempt.update(failure="spec_validation", issues=check)
                    previous, feedback = proposal, {"issues": check}
                    attempts.append(attempt)
                    continue
                attempt["accepted"] = True
                attempts.append(attempt)
                return spec, {
                    "status": "accepted",
                    "reason": "schema_and_independent_critic_pass",
                    "attempts": attempts, "author_calls": len(attempts),
                    "critic_calls": sum("review" in a for a in attempts),
                    "revision_rounds": revision,
                }
            attempt["failure"] = "critic_" + review["verdict"]
            previous, feedback = proposal, review
            attempts.append(attempt)
            if review["verdict"] == "uncertain":
                break
        except Exception as exc:
            attempt.update(failure="model_or_protocol_error",
                           detail=str(exc)[:260])
            attempts.append(attempt)
            if "did not return a JSON object" in str(exc):
                previous = None
                feedback = {"issues": ["Return JSON matching required schema."]}
                continue
            break
    return None, {
        "status": "rejected",
        "reason": (attempts[-1].get("failure", "no_attempt")
                   if attempts else "no_attempt"),
        "attempts": attempts,
        "author_calls": len(attempts),
        "critic_calls": sum("review" in a for a in attempts),
        "revision_rounds": max(0, len(attempts) - 1),
    }


def compile_semantic_world_scene(original: dict, spec: dict) -> dict:
    """Rebuild World + EvidenceGraph AFTER semantic spec, preserve MacroPlan.

    Accepts a precompiled TaskGraph bank to avoid rebuilding all KQA/MMQA
    plans. The legacy *values*, source facts and feedback in 'original' are
    NEVER read as inputs to this new World, only the frozen MacroPlan and
    its deterministic seed are used. Original snippets are excluded.
    """
    if "scene_grounding" in original or "question_authoring" in original or (
            "semantic_world_spec" in original):
        raise ValueError("use_original_unrealized_taskgraph_bank")
    problems = validate_spec(original, spec)
    if problems:
        raise ValueError("invalid_semantic_spec:" + ";".join(problems))
    from .world import build_clean_world
    from .evidence import run_clean_tasks
    rebuilt = build_clean_world(
        original,
        seed=original["task_plan_provenance"]["seed"],
        semantic_spec=spec,
        contexts=None,
    )
    executed = run_clean_tasks(original, rebuilt)
    public = copy.deepcopy(original["public"])
    public["available_source_descriptions"] = [
        {
            "source_id": sid,
            "title": source["title"],
            "modality_hint": source["modality_hint"],
            "representation": source["representation"],
        } for sid, source in sorted(rebuilt["sources"].items())
    ]
    realized = {
        **original,
        "world": rebuilt,
        **executed,
        "public": public,
        "semantic_world_spec": spec,
    }
    from .validation import validate_clean_scene
    errors = validate_clean_scene(realized)
    if errors:
        raise ValueError("semantic_world_replay_failed:" + ";".join(errors))
    return realized


def validate_semantic_world_scene(scene: dict) -> list[str]:
    """Pure replay from Plan + approved schema; detects value manipulation."""
    spec = scene.get("semantic_world_spec")
    if spec is None:
        return []
    errors = validate_spec(scene, spec)
    if errors:
        return errors
    from .world import build_clean_world
    try:
        expected = build_clean_world(
            scene,
            seed=scene["task_plan_provenance"]["seed"],
            semantic_spec=spec,
            contexts=None,
        )
        if scene["world"] != expected:
            errors.append("semantic_world_values_or_source_roles_drift")
    except (ValueError, TypeError, KeyError, IndexError) as exc:
        errors.append(
            "semantic_world_reinstantiation_error:" +
            type(exc).__name__ + ":" + str(exc)[:120])
    return sorted(set(errors))
