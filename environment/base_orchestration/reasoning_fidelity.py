"""Stage-3 Main reasoning fidelity against the *currently visible* handoff.

This is not an alternative planner: the DAG already fixes which work may
be dispatched. These checks concern only unsupported assertions about
what the current round can RETURN, number of delegated tasks, and
premature task completion. Conservative lexical checks deliberately defer
ambiguous language to the independent semantic Critic.
"""
from __future__ import annotations

import re


VERSION = "stage3.evidence_contract.v1"

_NUMBERS = {
    "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8,
}

# Only strong statements of current-round work delegation. Counting
# videos, documents, evidence sources or tool calls as Sub-Agent tasks
# would be wrong, so those concepts are deliberately excluded.
_DISPATCH_COUNT = re.compile(
    r"\b(?:delegate|dispatch|assign|launch)\s+(?:to\s+)?"
    r"(?:(?:exactly|a\s+total\s+of|the\s+next)\s+)?"
    r"(?P<count>\d+|one|two|three|four|five|six|seven|eight)\s+"
    r"(?:(?:independent|parallel|separate|specialized)\s+)?"
    r"(?:sub[ -]?agents?|agents?|tasks?|investigations?|delegations?)\b",
    re.I,
)

# Match only a direct assertion that THIS step produces a value.
# The scope ends at discourse transitions introducing later work.
_ROUND_CONTAINS = re.compile(
    r"\b(?:this\s+)?(?:round|batch|delegation)\s+"
    r"(?:consists?\s+of|contains?|has|will\s+dispatch)\s+"
    r"(?:exactly\s+)?(?P<count>\d+|one|two|three|four|five|six|seven|eight)\s+"
    r"(?:(?:parallel|independent|separate)\s+)?"
    r"(?:tasks?|sub[ -]?agents?|delegations?)\b",
    re.I,
)
_DIRECT_RETURN = re.compile(
    r"\b(?:this|the|current|next|pending)\s+"
    r"(?:(?:specific|assigned|index|source|keyed|current)\s+){0,3}"
    r"(?:task|lookup|query|investigation|request|round|step|delegation)"
    r"(?:\s+(?:will|should|can|is\s+expected\s+to|is\s+meant\s+to|"
    r"is\s+going\s+to))?\s+"
    r"(?:return|retrieve|yield|provide|produce|output|obtain|"
    r"extract|report|give|identify)\s+"
    r"(?P<claim>[^.!?;\n]{1,150})",
    re.I,
)
# A named current source may be described as returning a field even
# when the reasoning avoids "this lookup". Never analyze a different,
# future source this way.
_SOURCE_RETURN = re.compile(
    r"\b(?P<source>index_1|index_2|stat_a|stat_b|archive)\s+"
    r"(?:(?:lookup|query|search)\s+)?"
    r"(?:(?:will|should|can|is\s+expected\s+to)\s+)?"
    r"(?:return|retrieve|yield|provide|produce|obtain|"
    r"extract|report|give)\s+(?P<claim>[^.!?;\n]{1,150})",
    re.I,
)
_FUTURE_CLAUSE = re.compile(
    r"\s*(?:,|\band\s+then\b|\bthen\b|\bbefore\b|"
    r"\bso\s+that\b|\bin\s+order\s+to\b|\bto\s+enable\b|"
    r"\bto\s+(?:(?:later|subsequently)\s+)?"
    r"(?:retrieve|look\s+up|obtain|calculate|compute|verify)\b|"
    r"\bwhich\s+(?:will|can|should)\b|\bfor\s+(?:later|subsequent)\b)"
    r"\s*",
    re.I,
)

_FINAL_STEP = re.compile(
    r"\b(?:this|the|current|next)\s+"
    r"(?:(?:assigned|index|keyed)\s+){0,2}"
    r"(?:task|lookup|step|query|investigation)\s+"
    r"(?:is|will\s+be|marks|represents)\s+"
    r"(?:the\s+)?(?:last|final)\s+"
    r"(?:necessary\s+)?(?:task|lookup|step|stage)\b"
    r"|\bexecuting\s+(?:this|the)\s+(?:task|lookup|step|query)\s+"
    r"is\s+(?:the\s+)?(?:last|final)\s+(?:necessary\s+)?step\b",
    re.I,
)
_FINAL_ANSWER = re.compile(
    r"\b(?:this|the|current|next)\s+"
    r"(?:(?:assigned|index|keyed)\s+){0,2}"
    r"(?:task|lookup|step|query|investigation)\s+"
    r"(?:will|should|can|is\s+expected\s+to)\s+"
    r"(?:finish|complete|fully\s+resolve|answer)\s+"
    r"(?:the\s+)?(?:entire|overall|original|user'?s?)?\s*"
    r"(?:investigation|question|request|task)\b",
    re.I,
)

# Distinguish field names, NOT the incidental contents of source texts.
# For example, looking at a report containing years is not equivalent
# to RETURNING exhibition_year as the current Macro's output.
_FIELD_VOCAB = {
    "case_id", "candidate_id", "marker_code",
    "record_key_1", "record_key_2", "review_year",
    "exhibition_year", "reported_year_a", "reported_year_b",
    "completed_count", "total_count", "completion_rate_percent",
    "metric_a", "metric_b", "event_step", "category_name",
}


def _field_alias(field: str) -> re.Pattern:
    parts = field.split("_")
    expr = r"[\s_-]+".join(re.escape(part) for part in parts)
    # Allow ordinary singular/plural variants for count / year.
    suffix = r"s?" if parts[-1] in {"count", "year"} else ""
    return re.compile(r"(?<!\w)" + expr + suffix + r"(?!\w)", re.I)



# An *entire* public Result must be a finite explicit identifier list
# before we compute a set relation. Substring extraction from narrative
# prose would silently drop conditions or include unrelated examples.
_LITERAL_CASE_LIST = re.compile(
    r"^The current task returned (?P<count>\d+) case identifiers?: "
    r"(?P<ids>Case-\d+(?:,\s*Case-\d+)*)\.$",
    re.I,
)


def _literal_case_identifiers(result: str) -> list[str] | None:
    if not isinstance(result, str):
        return None
    match = _LITERAL_CASE_LIST.fullmatch(result.strip())
    if not match:
        return None
    cases = [part.strip() for part in match.group("ids").split(",")]
    if len(cases) != int(match.group("count")) or len(set(cases)) != len(cases):
        return None
    return cases


def visible_identifier_set_witness(
    upstream_results: list[str], *, output_fields: list[str],
) -> dict | None:
    """Derive ONLY a literal set overlap from two returned Sub-Agent Results.

    Never label either Result correct, independently verified, or trustworthy;
    do not use structured Oracle rows, hidden future task values, or the
    question's intended Gold answer. Invalid/incomplete text is UNKNOWN.
    """
    if len(upstream_results) != 2 or output_fields != ["case_id"]:
        return None
    groups = [_literal_case_identifiers(value) for value in upstream_results]
    if any(group is None for group in groups):
        return None
    left, right = groups
    shared = sorted(set(left) & set(right))
    return {
        "relation": "literal_identifier_set_intersection",
        "input_identifier_sets": [left, right],
        "shared_identifiers": shared,
        "scope": (
            "The overlap is computed from two already returned public "
            "Result strings, not an assertion that their source claims "
            "are true, independently verified or sufficient for the "
            "final user answer."
        ),
    }


def build_round_evidence_contract(
    assignments: list[dict], delivered: dict, *,
    has_later_delegation: bool = False,
    upstream_result_groups: list[list[str] | None] | None = None,
) -> dict:
    """Public facts plus a PRIVATE boolean for checking premature finish.

    No future source content, answer, task ID, Macro ID or gold program
    enters the public/Author/Critic projection. Previous feedback is
    copied only from delivered Main-visible results.
    """
    if not assignments or not all(isinstance(x, dict) for x in assignments):
        raise ValueError("stage3_evidence_contract_requires_assignments")
    current = []
    for item in assignments:
        current.append({
            "instruction": item["target_instruction"],
            "context": item["context"],
            "source_ids": list(item["source_ids"]),
            "output_fields": list(item["output_fields"]),
            "needs_upstream": bool(item["needs_upstream"]),
        })
    if upstream_result_groups is not None and (
        len(upstream_result_groups) != len(assignments)
    ):
        raise ValueError("stage3_visible_result_group_cardinality_mismatch")
    witnessed_relations = []
    for i, item in enumerate(assignments):
        group = (upstream_result_groups[i]
                 if upstream_result_groups is not None else None)
        if group is None:
            continue
        witness = visible_identifier_set_witness(
            group, output_fields=item["output_fields"],
        )
        if witness is not None:
            witnessed_relations.append({
                "current_round_task_slot": i,
                **witness,
            })
    prior = []
    for report in delivered.values():
        finish = report["finish_result"]
        prior.append({
            "status": finish["status"],
            "result": finish["result"],
            "summary": finish["summary"],
        })
    return {
        "version": VERSION,
        "public": {
            "current_round_task_count": len(current),
            "current_round_tasks": current,
            "already_returned_reports": prior,
            "observed_identifier_set_relations": witnessed_relations,
            "interpretation": (
                "The named output fields are the current handoff, not "
                "every observation that a source might contain. "
                "Prior reports are observations, not automatically "
                "independent proof of every claim."
            ),
        },
        "private_limits": {
            "has_later_delegation": bool(has_later_delegation),
        },
    }


def check_reasoning_fidelity(reasoning: str, contract: dict) -> list[str]:
    """Precise error codes; abstain on ambiguous causal/linguistic claims."""
    if not isinstance(reasoning, str):
        return ["invalid_main_reasoning"]
    public = contract["public"]
    count = public["current_round_task_count"]
    issues = []
    for mention in list(_DISPATCH_COUNT.finditer(reasoning)) + list(
        _ROUND_CONTAINS.finditer(reasoning)
    ):
        token = mention.group("count").casefold()
        observed = int(token) if token.isdigit() else _NUMBERS[token]
        if observed != count:
            issues.append("reasoning_dispatch_task_count_mismatch")
            break

    if contract["private_limits"]["has_later_delegation"]:
        if _FINAL_STEP.search(reasoning) or _FINAL_ANSWER.search(reasoning):
            issues.append("reasoning_premature_final_step")

    allowed = {
        field for task in public["current_round_tasks"]
        for field in task["output_fields"]
    }
    known = _FIELD_VOCAB | allowed
    current_sources = {
        source.casefold()
        for task in public["current_round_tasks"]
        for source in task["source_ids"]
    }
    direct = [
        match.group("claim")
        for match in _DIRECT_RETURN.finditer(reasoning)
    ]
    direct.extend(
        match.group("claim")
        for match in _SOURCE_RETURN.finditer(reasoning)
        if match.group("source").casefold() in current_sources
    )
    for claim in direct:
        # "The next lookup returns key1, so I can later retrieve year"
        # must be accepted: only inspect the directly returned object.
        clause = _FUTURE_CLAUSE.split(claim, maxsplit=1)[0]
        missing_specific_field = any(
            _field_alias(field).search(clause)
            for field in sorted(known - allowed)
        )
        missing_count_pair = (
            (re.search(r"\bcompleted\s+(?:and|/)\s+total\s+counts?\b",
                       clause, re.I)
             and not {"completed_count", "total_count"} <= allowed)
            or
            (re.search(r"\bfirst\s+(?:and|/)\s+second\s+counts?\b",
                       clause, re.I)
             and not {"metric_a", "metric_b"} <= allowed)
        )
        if missing_specific_field or missing_count_pair:
            issues.append("reasoning_current_output_field_mismatch")
            break
    return sorted(set(issues))
