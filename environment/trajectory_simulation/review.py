"""Check teacher language against fixed steps, never generate another solution."""
from __future__ import annotations

import json
from typing import Any, Mapping

from .contract import operation_steps, source_evidence


def review_language(role: Any, prompt: str, texts: list[str]) -> tuple[bool, list[str]]:
    if getattr(role, "offline", False):
        return True, []
    verdict = role.review(prompt)
    if not isinstance(verdict, Mapping):
        return False, ["invalid_language_review"]
    issues = verdict.get("issues")
    if verdict.get("accept") is True and issues == []:
        return True, []
    errors = []
    for issue in issues[:3] if isinstance(issues, list) else []:
        if not isinstance(issue, Mapping):
            continue
        quote, reason = issue.get("quote"), issue.get("reason")
        if (isinstance(quote, str) and quote.strip()
                and any(quote in text for text in texts)
                and isinstance(reason, str) and reason.strip()):
            errors.append("language_inconsistency:" + json.dumps(
                {"quote": quote, "reason": reason}, ensure_ascii=False))
    return False, errors or ["invalid_language_review"]


def _prompt(packet: Mapping[str, Any], rules: str) -> str:
    return (
        "Audit ONLY the generated language against the supplied authoritative contract. "
        "All supplied answer values and step outputs are authoritative and correct. "
        "FALSE boolean outcomes and EMPTY sets are valid expected results, not invalid "
        "tasks. Checking a condition that returns false is not a contradiction in the "
        "instruction. Observed records are not automatically input candidates: an empty "
        "input stays empty through filters even if other records satisfy the predicate. "
        "Do NOT solve the original question, change these results, or demand external proof. "
        "Accept concise wording and omissions that do not change task meaning. "
        "Reject specific contradictions, invented execution/evidence, or wrong task scope. "
        + rules +
        ' Return only {"accept":true|false,"issues":[{"quote":"exact generated text",'
        '"reason":"brief contradiction with the supplied contract"}]}. '
        "At most three issues; each quote <=160 characters and reason <=220 characters. "
        "Quote only generated language, not contract text. If no concrete error, accept "
        "with issues=[]. No extended analysis.\n" +
        json.dumps(packet, ensure_ascii=False, separators=(",", ":"))
    )


def action_review_prompt(action, guidance, public_state, completed_results=(), fragments=()):
    return _prompt({
        "generated_action": action,
        "ordered_subgoal_contracts": [{
            "task_index": index, "steps": operation_steps(item["operations"]),
            "required_tools": item["suggested_tools"],
        } for index, item in enumerate(guidance["ready"])],
        "completed_public_history": public_state["subtask_history"],
        "may_complete": guidance["may_complete"],
        "completed_result_types": [{
            "answer": item["answer"], "answer_kind": item["answer_kind"],
        } for item in completed_results],
        "verified_subgoal_contracts": [{
            "task_index": index, "answer_kind": fragment["answer_kind"],
            "final_answer": fragment["expected_answer"],
            "step_results": [item["answer"] for item in fragment["operation_results"]],
            "inspected_source_facts": source_evidence([
                *fragment["observations"], *fragment["known_observations"]]),
        } for index, fragment in enumerate(fragments)],
    }, (
        "For delegation, each task must match its SAME ordered subgoal: no unrelated task, "
        "changed predicate/comparator/direction, extra filter, omitted necessary operation, "
        "or swapped tasks. Context may mention the overall objective without imposing its "
        "future filters in this task. Respect dependency edges, not merely list order; "
        "FIND_ALL starts a new branch, and UNION/INTERSECT combine specified inputs. "
        "Treat entity_set results as entity references, even when encoded as lists of "
        "names or when a relation is labelled nationality. Do NOT reject a listed filter "
        "because you think its input is a literal, lacks attributes or requires later "
        "evidence. The provided operator/type contracts authorize the operation; only "
        "check whether the instruction matches it. A noun description like 'person "
        "records' is not an extra type-filter unless removals by type are instructed. "
        "Querying/retrieving an attribute is the same step, not an extra retrieval. "
        "An already satisfied type condition may be implicit in a later reference to "
        "'remaining persons'. Reject only a clear changed operation or dependency. "
        "The inspected_source_facts and successful step_results demonstrate that these filters "
        "are legal in this synthetic world. Do not substitute general world knowledge "
        "about which entity types normally have an attribute. "
        "For complete, the answer has already passed exact validation; only check whether "
        "its reasoning contradicts completed public results. "
    ))


def feedback_review_prompt(language, payload):
    fragment = payload["fragment"]
    return _prompt({
        "generated_language": language,
        "assigned_task": payload["assigned_task"],
        "authoritative_answer": fragment["expected_answer"],
        "verified_step_results": [{
            "step": index + 1, "answer": item["answer"],
        } for index, item in enumerate(fragment.get("operation_results", []))],
        "execution_steps": operation_steps(fragment["operations"]),
        "predecessor_results": payload["dependency_results"],
        "new_observations": source_evidence(fragment["observations"]),
        "previously_read_observations": source_evidence(fragment.get("known_observations", [])),
    }, (
        "The summary must interpret THIS final subtask answer, not repeat an earlier "
        "subtask's conclusion. For example an empty input remains empty through a filter; "
        "it does not rediscover an entity from another source. Preserve relation direction. "
        "Use exact title-to-source aliases from observations. Do not claim a reused source "
        "was newly retrieved or fabricate an entity/value/link. Describe the supplied "
        "per-operation outputs faithfully, including independent branches and merges. "
        "Relations are MULTI-VALUED: observing Mira|nationality|Westridge in one source "
        "does NOT invalidate Mira|nationality|Ravencrest in another. Consider all supplied "
        "facts. Do not demand a single source contain every fact in a multi-source query. "
        "A list of used sources does not claim each source individually proves the result. "
    ))
