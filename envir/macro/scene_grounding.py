"""Post-TaskGraph LLM scene/world concretization without changing clean evidence.

The LLM can inspect reference Question, compact EvidenceGraphs and factual
World rows. It creates *semantic labels* for the already simulated records,
not extra observations. Code deterministically binds every World value to
its original source:fact_id. A separate Critic screens contextual plausibility.
No actual image/video perception or original MMQA gold is inferred.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from collections import Counter

FIELDS = (
    "candidate_key", "synthetic_attribute_score",
    "synthetic_attribute_flag", "synthetic_time_index",
    "synthetic_group_key",
)
DISPLAY = {
    "candidate_key": "entry",
    "synthetic_attribute_score": "score",
    "synthetic_attribute_flag": "positive_mark",
    "synthetic_time_index": "relative_order",
    "synthetic_group_key": "group",
}

CONSTRUCTOR_SYSTEM = """You are a domain-aware SYNTHETIC SCENE CONSTRUCTOR.

The TaskGraph, computed operations, Source World and EvidenceGraphs are
ALREADY BUILT and IMMUTABLE. Your job is to make this existing abstract
World understandable as one plausible, concrete, hypothetical scenario.

You are shown:
- reference_question: the existing task-derived Question; use as a semantic
  anchor, not a sentence to copy;
- original_mmqa_question: an optional, WEAK topic/style cue, NEVER evidence
  that a new event or fact is true and NEVER the target operation;
- a compact Macro TaskGraph and its actual EvidenceGraph operations;
- source_records: ALL candidate IDs, flags, scores, relative indices and
  groups already present in the frozen synthetic World, with source fact IDs;
- original source excerpt proxies when available, marked as context only.

You MAY use general domain knowledge to propose a realistic *hypothetical*
use case: e.g. a curator reconciling exhibit entries, an editor reviewing
fictional sports submissions, a technician comparing audit scores. Choose
specific, meaningful roles for the existing abstract numerical/boolean
fields. You MAY invent fictional labels for synthetic entries, clearly
as an interpretation of the simulation, not names extracted from media.
Write short, concrete fields rather than long explanations: ideally each
field meaning <= 130 characters, source role <= 160 characters, and user
need <= 220 characters. These are WRITING GUIDELINES, not brittle rejection
thresholds; exact source/fact bindings and meaning are what matter.

You MUST NOT:
- change, recompute or invent a single source fact, score, flag, index,
  group membership, source ID or candidate ID;
- make historical or current factual claims about the real topic;
- pretend that image/video/audio pixels were analyzed;
- add a source, operation, event, constraint, deadline, correction,
  causal relation, entity match or expected final answer;
- infer that two distinct candidates are identical just because they
  share a synthetic group key;
- invent a source subtype (poster, personal account, news article) as
  something actually supplied by the original dataset.

IMPORTANT: The outputs are a SYNTHETIC INTERPRETATION LAYER, not verified
annotations of original MMQA sources. A hypothetical record card label is
fine; claiming an actual screenshot depicted a person is not.
An assigned boolean flag MAY be described as an imaginary review marker;
it MUST NOT become "visually confirmed", "detected in the image",
"verified by the photograph" or a factual claim about what pixels show.
Distinguish "a hypothetical crest-check flag" from "the actual crest has
a crown". Likewise, a synthetic time index is a relative record order,
not evidence of a historical release date or an actual event chronology.
Where domain-specific imagery is unavailable, describe its *source role*
as an image-source metadata record, not an observed photograph/poster.
The original MMQA Question may contain details about unseen original media.
Use it only as inspiration for WHAT DOMAIN to simulate, never claim its
specific event/visual details were observed in these simulated sources.
The chosen field meanings must not contradict boolean flags, numeric scores,
group IDs or relative time indices. Use 'relative sequence' not calendar dates.

For EVERY supplied source_id provide exactly one role+candidate label,
matching its already supplied candidate_key. Keep distinct candidates
distinguishable. Explain a genuine information need the existing Goal and
TaskGraph can solve; don't add an extra step to make the story interesting.

Return ONLY strict JSON:
{"scene_context":"one or two concrete hypothetical sentences",
 "candidate_kind":"domain-appropriate class of entries",
 "field_meanings":{"score":"...", "positive_flag":"...",
                   "relative_time_index":"...", "group_key":"..."},
 "source_roles":[{"source_id":"exact input source id",
                  "candidate_label":"hypothetical display name",
                  "source_role":"hypothetical role of that source record"}],
 "user_need":"the already supported final information need"}
"""

CRITIC_SYSTEM = """You are a skeptical SYNTHETIC WORLD CONCRETIZATION CRITIC.

The original synthetic World values and compiled TaskGraph/EvidenceGraph
are authoritative. The candidate is only a human-readable interpretation
layer. Do NOT assume original images were viewed or real historical claims
were verified. The optional original MMQA Question is inspiration, NOT truth.

Check independently:
- domain_plausibility: a coherent realistic HYPOTHETICAL use case with
  meaningful, domain-appropriate entry and field labels, not boilerplate.
- frozen_facts: no new or changed values, dates, entities alleged to be
  factual, inferred media contents, or unsupported relations. Carefully
  distinguish fictional marker semantics from claims that a visual feature
  was actually observed, confirmed, detected or verified in source media.
  A generated label MUST NOT be treated as actual image/video recognition.
- Source-ablation is merely a goal-result influence diagnostic. Do not
  fabricate stronger multi-source links to compensate for weak influence;
  label sources honestly and leave TaskGraph improvements to a later stage.
- source_roles: labels refer only to supplied sources; they do not imply
  possession of unprovided documents or observed source media.
- solution_compatibility: the same operation, filters, source dependencies,
  and final substantive goal work without an additional operation.
- question_alignment: scene_context, user_need and candidate labels support
  the original task-derived reference Question and selected Goal.
  The original MMQA Question must NOT replace the selected Goal.

PASS iff all five axes are YES and issues=[].
For definite local repairable issues return REVISE with concrete reasons.
If an issue cannot be established from records, return UNCERTAIN.
Do not reflexively reject the creation of fictional labels or ordinary
hypothetical context; those are the purpose of the stage. NEVER certify
historical truth, real media interpretation, or formal task necessity.
Return JSON:
{"domain_plausibility":"yes|no|uncertain",
 "frozen_facts":"yes|no|uncertain",
 "source_roles":"yes|no|uncertain",
 "solution_compatibility":"yes|no|uncertain",
 "question_alignment":"yes|no|uncertain",
 "verdict":"pass|revise|uncertain",
 "issues":["specific concern if any"],
 "summary":"concise concrete assessment"}
"""

PROP_SCHEMA = {
    "type": "object",
    "properties": {
        "scene_context": {"type": "string"},
        "candidate_kind": {"type": "string"},
        "field_meanings": {
            "type": "object",
            "properties": {key: {"type": "string"} for key in (
                "score", "positive_flag", "relative_time_index", "group_key",
            )},
            "required": [
                "score", "positive_flag", "relative_time_index", "group_key",
            ],
            "additionalProperties": False,
        },
        "source_roles": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {key: {"type": "string"} for key in (
                    "source_id", "candidate_label", "source_role",
                )},
                "required": ["source_id", "candidate_label", "source_role"],
                "additionalProperties": False,
            },
        },
        "user_need": {"type": "string"},
    },
    "required": [
        "scene_context", "candidate_kind", "field_meanings",
        "source_roles", "user_need",
    ],
    "additionalProperties": False,
}
REVIEW_AXES = (
    "domain_plausibility", "frozen_facts", "source_roles",
    "solution_compatibility", "question_alignment",
)
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


def fingerprint(value) -> str:
    packed = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(packed.encode("utf-8")).hexdigest()


def frozen_fingerprint(scene: dict) -> str:
    """Includes every bytewise immutable part of the compiled solution."""
    return fingerprint({
        key: scene[key] for key in (
            "macro_graph", "goal", "world", "task_evidence_graphs",
            "feedback_bank", "goal_evidence_coverage",
        )
    })


def source_records(scene: dict) -> list[dict]:
    """Make source-fact bindings exact; never ask LLM to fabricate a value."""
    records = []
    for sid, source in sorted(scene["world"]["sources"].items()):
        facts = {fact["field"]: fact for fact in source["facts"]}
        if not set(FIELDS) <= facts.keys():
            raise ValueError("source_missing_groundable_fields:" + sid)
        values = {
            DISPLAY[field]: {
                "value": copy.deepcopy(facts[field]["value"]),
                "fact_ref": sid + ":" + facts[field]["fact_id"],
                "verification": facts[field]["verification"],
            }
            for field in FIELDS
        }
        excerpt = facts.get("original_context_excerpt")
        records.append({
            "source_id": sid,
            "modality": source["modality_hint"],
            "source_provenance": source.get("source_origin"),
            "original_doc_id": source.get("original_doc_id"),
            "representation": source["representation"],
            "fields": values,
            "original_excerpt_context_only": (
                str(excerpt["value"])[:360] if excerpt else None
            ),
        })
    return records


def evidence_summary(scene: dict) -> list[dict]:
    """LLM sees actual typed operation results and source evidence, not just
    aspirational goal text, but avoid passing duplicate verbose trace text."""
    summaries = []
    for task in scene["macro_graph"]["tasks"]:
        nodes = scene["task_evidence_graphs"][task["task_id"]]["nodes"]
        operation_nodes = [
            node for node in nodes
            if node["kind"] == "typed_operation_evaluation"
        ]
        if len(operation_nodes) != 1:
            raise ValueError("operation_evidence_missing:" + task["task_id"])
        summaries.append({
            "task_number": len(summaries) + 1,
            "operation": task["operation_spec"]["op"],
            "work_type": task["work_type"],
            "deliverable_type": list(task["outputs"].values()),
            "prerequisites": [
                next(i + 1 for i, earlier in enumerate(
                    scene["macro_graph"]["tasks"])
                    if earlier["task_id"] == dep)
                for dep in task["depends_on"]
            ],
            "source_ids": operation_nodes[0]["result"]["source_ids"],
            "computed_result": operation_nodes[0]["result"],
        })
    return summaries


def constructor_input(scene: dict) -> dict:
    return {
        "topic": scene["scenario"]["topic"],
        "domain": scene["scenario"]["domain"],
        "goal_id": scene["goal"]["intent"],
        "reference_question": scene["question"],
        "original_mmqa_question_weak_style_cue": scene.get(
            "source_priors", {}).get("MultiModalQA", {}).get("original_question"),
        "source_records": source_records(scene),
        "evidence_graph_operations": evidence_summary(scene),
        "goal_source_ablation_proxy": source_ablation_signal(scene),
        "simulation_notice": (
            "All source scores, flags, order and group attributes are "
            "already computed synthetic facts; actual image contents have "
            "not been read. Propose interpretation labels only."
        ),
    }


def _single_line(value, *, limit=220) -> bool:
    return (isinstance(value, str) and 2 <= len(value.strip()) <= limit
            and "\n" not in value and "\r" not in value
            and "synthetic_" not in value.lower())


def validate_proposal(scene: dict, proposal: dict) -> list[str]:
    """High-confidence checks. World value binding is done in Python."""
    errors = []
    if not isinstance(proposal, dict) or set(proposal) != set(PROP_SCHEMA["required"]):
        return ["invalid_proposal_schema"]
    for key, limit in (
        ("scene_context", 650), ("candidate_kind", 160),
        ("user_need", 360),
    ):
        if not _single_line(proposal.get(key), limit=limit):
            errors.append("invalid_" + key)
    roles = proposal.get("source_roles")
    expected = {row["source_id"]: row for row in source_records(scene)}
    if not isinstance(roles, list) or len(roles) != len(expected):
        errors.append("source_role_count_mismatch")
        roles = []
    seen, labels = set(), set()
    for row in roles:
        if not isinstance(row, dict) or set(row) != {
            "source_id", "candidate_label", "source_role",
        }:
            errors.append("invalid_source_role")
            continue
        sid = row["source_id"]
        if sid not in expected or sid in seen:
            errors.append("unknown_or_duplicate_source_id")
        seen.add(sid)
        for key in ("candidate_label", "source_role"):
            if not _single_line(row[key], limit=(140 if key == "candidate_label" else 320)):
                errors.append("invalid_" + key)
        label = row["candidate_label"].casefold().strip()
        if label in labels:
            errors.append("duplicate_candidate_label")
        labels.add(label)
    if seen != set(expected):
        errors.append("source_role_coverage_mismatch")
    meanings = proposal.get("field_meanings")
    if not isinstance(meanings, dict) or set(meanings) != {
        "score", "positive_flag", "relative_time_index", "group_key",
    }:
        errors.append("field_meaning_schema_invalid")
    elif any(not _single_line(value, limit=320) for value in meanings.values()):
        errors.append("invalid_field_meanings")
    # No media content assertions or claims about actual original documents.
    prose = " ".join([
        str(proposal.get("scene_context", "")),
        str(proposal.get("user_need", "")),
    ] + [
        str(row.get("source_role", "")) for row in roles if isinstance(row, dict)
    ] + [
        str(row.get("candidate_label", "")) for row in roles
        if isinstance(row, dict)
    ] + [
        str(value) for value in (meanings or {}).values()
    ] if isinstance(meanings, dict) else [])
    if re.search(r"\b(?:image|photo|poster|screenshot|video|footage)\b"
                 r".{0,60}\b(?:shows?|depicts?|contains?|reveals?)\b",
                 prose, re.I):
        errors.append("unobserved_media_content_claim")
    # Frozen World contains synthetic flags, NOT source-media perception.
    # A free-form hypothetical context is allowed; explicit claims of
    # *confirmed actual visual observation* are not.
    if re.search(
        r"\b(?:visual(?:ly)?\s+(?:confirmation|confirmed|verification|"
        r"verified|detected)|image[- ]derived\s+(?:observation|confirmation)|"
        r"logo\s+(?:confirmed|verified|observed|detected)|"
        r"(?:confirmed|verified|detected)\s+(?:in|from|by)\s+(?:the\s+)?"
        r"(?:image|photo|poster|screenshot))\b",
        prose, re.I
    ):
        errors.append("unsupported_media_perception_certainty")
    if re.search(r"\b(?:verified|authentic|historically|actually)\b"
                 r".{0,30}\b(?:fact|award|event|winner|record)\b",
                 prose, re.I):
        errors.append("unverified_real_world_claim")
    return sorted(set(errors))


def validate_review(review: dict) -> tuple[dict, list[str]]:
    issues = []
    if not isinstance(review, dict):
        return {}, ["invalid_critic_json"]
    if set(review) != set(REVIEW_SCHEMA["required"]):
        return review, ["invalid_critic_schema"]
    for axis in REVIEW_AXES:
        if review.get(axis) not in {"yes", "no", "uncertain"}:
            issues.append("invalid_critic_axis:" + axis)
    if (review.get("verdict") not in {"pass", "revise", "uncertain"}
            or not isinstance(review.get("issues"), list)
            or any(not isinstance(x, str) or not x.strip()
                   for x in review.get("issues", []))
            or not _single_line(review.get("summary"), limit=1400)):
        issues.append("invalid_critic_fields")
    # Keep decision conservative in the face of contradictory outputs.
    if not issues:
        expected = (
            "uncertain" if any(review[key] == "uncertain" for key in REVIEW_AXES)
            else "revise" if (any(review[key] == "no" for key in REVIEW_AXES)
                              or bool(review["issues"]))
            else "pass"
        )
        if review["verdict"] != expected:
            review = {**review, "verdict": expected}
    return review, issues


def _goal_signature(goal_id: str, records: list[dict]):
    """Goal-level *proxy* outcome, NOT a re-execution of WorkProduct logic."""
    rows = [{
        "candidate": row["fields"]["entry"]["value"],
        "score": row["fields"]["score"]["value"],
        "positive": row["fields"]["positive_mark"]["value"],
        "time": row["fields"]["relative_order"]["value"],
        "group": row["fields"]["group"]["value"],
    } for row in records]
    ordered = sorted(rows, key=lambda x: (-x["score"], x["candidate"]))
    groups = {}
    for row in rows:
        groups.setdefault(row["group"], []).append(row["candidate"])
    links = tuple(sorted(
        (key, tuple(sorted(members)))
        for key, members in groups.items()
        if len(members) >= 2
    ))
    if goal_id == "lookup":
        return tuple(sorted(r["candidate"] for r in rows if r["positive"]))
    if goal_id == "aggregation":
        return sum(r["positive"] for r in rows)
    if goal_id == "feasibility":
        return tuple(sorted(
            r["candidate"] for r in rows
            if r["positive"] and r["score"] >= 50))
    if goal_id in {"comparison", "prioritization"}:
        return tuple(r["candidate"] for r in ordered)
    if goal_id == "temporal":
        return tuple(r["candidate"] for r in sorted(
            rows, key=lambda r: (r["time"], r["candidate"])))
    if goal_id in {"composition", "intersection"}:
        return links
    raise ValueError("unsupported_goal_for_source_ablation:" + goal_id)


def source_ablation_signal(scene: dict) -> dict:
    """Check whether omitting each existing source changes *goal-level data*.

    This is a diagnostic, not a validity criterion. A SourceEvidence root
    may still be necessary for full traceability even when removing it does
    not change the short final answer (e.g. a negative flag in a count).
    """
    records = source_records(scene)
    goal_id = scene["goal"]["intent"]
    signature = _goal_signature(goal_id, records)
    changed, unchanged = [], []
    for record in records:
        subset = [
            value for value in records
            if value["source_id"] != record["source_id"]
        ]
        (changed if _goal_signature(goal_id, subset) != signature
         else unchanged).append(record["source_id"])
    return {
        "method": "leave_one_source_out_goal_proxy_not_taskgraph_proof",
        "goal_id": goal_id,
        "source_count": len(records),
        "influential_sources": changed,
        "non_influential_sources": unchanged,
        "note": (
            "Non-influential means no change to this goal-level proxy "
            "result, NOT that the source's evidence is irrelevant or "
            "that any Macro Task can be removed."
        ),
    }


def materialize_grounding(scene: dict, proposal: dict, review: dict) -> dict:
    """LLM writes labels; Python binds immutable values + fact references."""
    errors = validate_proposal(scene, proposal)
    if errors:
        raise ValueError("cannot_materialize:" + ",".join(errors))
    roles = {role["source_id"]: role for role in proposal["source_roles"]}
    cards = []
    for record in source_records(scene):
        sid = record["source_id"]
        role = roles[sid]
        cards.append({
            "source_id": sid,
            "modality": record["modality"],
            "candidate_key": record["fields"]["entry"]["value"],
            "candidate_label": role["candidate_label"],
            "source_role": role["source_role"],
            "fact_bindings": copy.deepcopy(record["fields"]),
            "fact_origin": "deterministic_synthetic_world_not_original_mmqa",
        })
    return {
        "version": "synthetic_world_concretization_v1",
        "mode": "synthetic_interpretation_not_original_media_annotation",
        "status": "accepted",
        "topic": scene["scenario"]["topic"],
        "goal_id": scene["goal"]["intent"],
        "scene_context": proposal["scene_context"],
        "candidate_kind": proposal["candidate_kind"],
        "field_meanings": copy.deepcopy(proposal["field_meanings"]),
        "user_need": proposal["user_need"],
        "source_cards": cards,
        "review": copy.deepcopy(review),
        "original_world_fingerprint": fingerprint(scene["world"]),
        "frozen_solution_fingerprint": frozen_fingerprint(scene),
        "proposal_fingerprint": fingerprint(proposal),
        "validation": {
            "source_coverage": "exact",
            "fact_values": "bound_from_frozen_world",
            "operation_replay": "original_unchanged",
            "source_grounding": "synthetic_hypothesis_not_real_media",
            "llm_review_not_formal_proof": True,
            "goal_level_source_ablation": source_ablation_signal(scene),
        },
    }


def validate_grounding(scene: dict) -> list[str]:
    """Replay-time guard for accepted overlays: no silent source/fact drift."""
    g = scene.get("scene_grounding")
    if g is None:
        return []
    errors = []
    if not isinstance(g, dict) or g.get("status") != "accepted":
        return ["invalid_scene_grounding"]
    if g.get("version") != "synthetic_world_concretization_v1":
        errors.append("unknown_grounding_version")
    if g.get("mode") != "synthetic_interpretation_not_original_media_annotation":
        errors.append("grounding_claims_real_media")
    if g.get("topic") != scene["scenario"]["topic"] or (
            g.get("goal_id") != scene["goal"]["intent"]):
        errors.append("grounding_topic_or_goal_drift")
    if g.get("original_world_fingerprint") != fingerprint(scene["world"]):
        errors.append("grounding_world_fingerprint_changed")
    if g.get("frozen_solution_fingerprint") != frozen_fingerprint(scene):
        errors.append("grounding_solution_fingerprint_changed")
    review, review_issues = validate_review(g.get("review", {}))
    errors.extend(review_issues)
    if review.get("verdict") != "pass" or any(
            review.get(key) != "yes" for key in REVIEW_AXES):
        errors.append("grounding_review_not_passed")
    proposal = {
        "scene_context": g.get("scene_context"),
        "candidate_kind": g.get("candidate_kind"),
        "field_meanings": g.get("field_meanings"),
        "source_roles": [
            {"source_id": card.get("source_id"),
             "candidate_label": card.get("candidate_label"),
             "source_role": card.get("source_role")}
            for card in g.get("source_cards", [])
            if isinstance(card, dict)
        ],
        "user_need": g.get("user_need"),
    }
    errors.extend(validate_proposal(scene, proposal))
    if g.get("proposal_fingerprint") != fingerprint(proposal):
        errors.append("grounding_proposal_changed")
    try:
        expected = materialize_grounding(scene, proposal, g["review"])
        if g != expected:
            errors.append("grounding_fact_bindings_changed")
    except (ValueError, KeyError, TypeError, IndexError):
        errors.append("grounding_materialization_failed")
    return sorted(set(errors))


def ground_one(scene: dict, *, constructor, critic,
               max_revisions: int = 2) -> tuple[dict | None, dict]:
    """Author, deterministic facts, independent Critic, bounded repair."""
    if "scene_grounding" in scene:
        raise ValueError("scene_already_grounded")
    request = constructor_input(scene)
    attempts = []
    feedback = None
    last = None
    for revision in range(max_revisions + 1):
        user = dict(request)
        if last is not None:
            user["previous_proposal"] = last
            user["critic_feedback"] = feedback
        entry = {"round": revision + 1}
        try:
            proposal = constructor.complete_json(
                CONSTRUCTOR_SYSTEM, json.dumps(user, ensure_ascii=False),
                response_schema=PROP_SCHEMA,
            )
            entry["proposal"] = proposal
            local = validate_proposal(scene, proposal)
            if local:
                entry["validation_errors"] = local
                entry["failure"] = "local_schema_or_media_validation"
                feedback = {"issues": local}
                last = proposal
                attempts.append(entry)
                continue
            critic_payload = {
                "proposal": proposal,
                "reference_question": request["reference_question"],
                "goal_id": request["goal_id"],
                "source_records": request["source_records"],
                "evidence_graph_operations": request["evidence_graph_operations"],
                "goal_source_ablation_proxy": request["goal_source_ablation_proxy"],
                "original_mmqa_question_weak_style_cue": request[
                    "original_mmqa_question_weak_style_cue"],
                "review_focus": (
                    "Is this a plausible, consistent synthetic interpretation "
                    "of exactly the existing source facts and TaskGraph?"),
            }
            review_raw = critic.complete_json(
                CRITIC_SYSTEM, json.dumps(critic_payload, ensure_ascii=False),
                response_schema=REVIEW_SCHEMA,
            )
            review, review_errors = validate_review(review_raw)
            entry["critic_review"] = review
            if review_errors:
                entry["validation_errors"] = review_errors
                entry["failure"] = "critic_response_protocol"
                feedback = {"issues": review_errors}
                last = proposal
                attempts.append(entry)
                continue
            if review["verdict"] == "pass":
                overlay = materialize_grounding(scene, proposal, review)
                probe = {**scene, "scene_grounding": overlay}
                replay_errors = validate_grounding(probe)
                if replay_errors:
                    entry["validation_errors"] = replay_errors
                    entry["failure"] = "grounding_replay_validation"
                    feedback = {"issues": replay_errors}
                    last = proposal
                    attempts.append(entry)
                    continue
                entry["accepted"] = True
                attempts.append(entry)
                return overlay, {
                    "status": "accepted",
                    "reason": "critic_pass_and_exact_fact_binding",
                    "attempts": attempts,
                    "revision_rounds": revision,
                    "author_calls": revision + 1,
                    "critic_calls": sum(
                        "critic_review" in row for row in attempts),
                }
            entry["failure"] = "critic_" + review["verdict"]
            last = proposal
            feedback = review
            attempts.append(entry)
            if review["verdict"] == "uncertain":
                break
        except Exception as exc:
            entry["failure"] = "model_or_protocol_error"
            entry["detail"] = str(exc)[:240]
            attempts.append(entry)
            break
    return None, {
        "status": "rejected",
        "reason": attempts[-1].get(
            "failure", "validation_failed") if attempts else "no_attempt",
        "last_validation_errors": (
            attempts[-1].get("validation_errors", []) if attempts else []
        ),
        "attempts": attempts,
        "revision_rounds": max(0, len(attempts) - 1),
        "author_calls": len(attempts),
        "critic_calls": sum(
            "critic_review" in row for row in attempts),
    }
