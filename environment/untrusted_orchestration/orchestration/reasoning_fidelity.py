"""Main-visible semantic checks for Stage-4C reasoning.

These guards use only the *public* evidence requests generated from already
returned Sub-Agent feedback. No private fault type, Gold or future result can
justify an asserted discrepancy. They deliberately target high-precision
contradictions, leaving ordinary narrative evaluation to the LLM Critic.
"""
from __future__ import annotations

import re


_UNCERTAIN = re.compile(
    r"\b(?:conflict(?:ing)?|contradic(?:tion|tory)|disagree(?:ment)?|"
    r"inconsisten(?:t|cy)|discrepan(?:t|cy)|mismatch(?:ed)?|"
    r"unresolved|uncertain|ambiguous|verify|verification|recheck|"
    r"cross-check|reconcile|validate|resolve|re-examin\w*|"
    r"re-validat\w*|compar\w*|disjoint|unaligned|not\s+(?:yet\s+)?confirmed|"
    r"needs?\s+(?:another\s+)?check)\b", re.I,
)
_FALSE_FINALITY = (
    re.compile(
        r"\b(?:previous|prior|earlier|last)\s+(?:round|step|attempt|"
        r"investigation)\s+(?:has\s+)?successfully\s+"
        r"(?:resolved|reconciled|verified)\b", re.I,
    ),
    re.compile(
        r"\b(?:the|this)\s+(?:conflict|discrepancy|mismatch|"
        r"inconsistency|contradiction)\s+"
        r"(?:has\s+been|was|is)\s+(?:fully\s+)?"
        r"(?:resolved|reconciled|verified)\b", re.I,
    ),
    re.compile(
        r"\b(?:no|without\s+any|no\s+remaining)\s+"
        r"(?:conflict|discrepancy|mismatch|inconsistency)\b", re.I,
    ),
    re.compile(
        r"\b(?:this|the)\s+(?:joined\s+)?(?:result|join|"
        r"candidate\s+match|binding)\s+(?:is|was|has\s+been)\s+"
        r"(?:fully\s+|already\s+|complete\s+and\s+)?"
        r"(?:verified|confirmed|reconciled)\b", re.I,
    ),
)


def reasoning_evidence_issues(
    reasoning: str, evidence_requests: list[dict]
) -> list[str]:
    """Reject only clearly false certainty about an unresolved VERIFY.

    A statement such as "the report *claims* verification, but its
    identifiers disagree" is not a claim that Main completed verification.
    Avoid deciding whether a particular reported value is true from Gold.
    """
    if not isinstance(reasoning, str):
        return ["reasoning_not_text"]
    verify = [
        item for item in evidence_requests
        if item.get("operation") == "verify_conflicting_evidence"
    ]
    if not verify:
        return []
    # A pending verification cannot justify dropping that verification.
    # This is distinct from a harmless future conditional such as
    # "once independently verified, we can finish".
    if re.search(
        r"\b(?:no\s+(?:need|reason)\s+to|need\s+not)\s+"
        r"(?:verify|recheck|cross-check|reconcile|validate)\b"
        r"|\b(?:we|I)\s+(?:can|should|may)\s+"
        r"(?:now\s+)?(?:complete|finalize|finish)\s+"
        r"(?:the\s+)?(?:task|answer|question)\b",
        reasoning, re.I,
    ):
        return ["premature_completion_without_required_verification"]
    # The current action must motivate the very check being delegated.
    if not _UNCERTAIN.search(reasoning):
        return ["missing_current_conflict_or_verification_motivation"]
    for phrase in re.split(r"(?<=[.!?])\s+", reasoning):
        if not any(pattern.search(phrase) for pattern in _FALSE_FINALITY):
            continue
        # Future conditional ("once this conflict is resolved") is not
        # a past-tense claim that verification already happened.
        if re.search(
            r"\b(?:once|until|if|when|unless|after)\b.{0,90}"
            r"\b(?:conflict|discrepancy|mismatch|inconsistency)\b"
            r".{0,90}\b(?:resolved|reconciled|verified)\b",
            phrase, re.I,
        ):
            continue
        # Quoting a previous Sub-Agent's *claim* is allowed if Main explicitly
        # questions it in the same sentence.
        if (re.search(r"\b(?:claims?|asserts?|reported?|allegedly)\b", phrase, re.I)
                and _UNCERTAIN.search(phrase)):
            continue
        if re.search(
            r"\b(?:not|never|cannot|hasn't|hadn't|wasn't|isn't)\s+"
            r"(?:already\s+|yet\s+)?(?:resolved|verified|confirmed|reconciled)\b",
            phrase, re.I,
        ):
            continue
        return ["prematurely_claims_conflict_verified_or_resolved"]
    return []


def derivation_instructions(task: dict) -> list[str]:
    """Describe planned operations and named inputs, without executing them."""
    clauses = []
    fields = task.get("outputs", {}).get("typed_relation", [])
    intermediates = [f for f in fields if re.fullmatch(r"entity_\d+", f)]
    if intermediates:
        clauses.append(
            "Return intermediate relationship identifier(s): "
            + ", ".join(intermediates)
            + ". These are lookup keys, not final place names or final answer fields."
        )
    if ("observation_id" in fields and "observation_year" in fields
            and not any(f.startswith(("date_", "latitude", "longitude")) for f in fields)):
        clauses.append(
            "Return observation identifiers and years for the year-based count. "
            "This assignment does not extract full dates or coordinates."
        )
    for node in task.get("operator_chain", []):
        op = node["op"]
        if op == "goal_compute":
            spec = node.get("goal_operation", {})
            inputs = ", ".join(node.get("input_fields", []))
            name = spec.get("function") or spec.get("op", "")
            if name == "ratio":
                fields = node["input_fields"]
                rule = fields[-2] + " divided by " + fields[-1]
            elif name == "temporal_difference":
                rule = "elapsed days between " + inputs
            elif name == "temporal_midpoint":
                rule = "midpoint between " + inputs
            else:
                rule = name.replace("_", " ") + " using " + inputs
            clauses.append("Calculate " + node["output_field"] + " as " + rule + ".")
        elif op == "derive_arithmetic":
            symbol = " divided by " if node["operation"] == "ratio" else " absolute difference from "
            clauses.append("Calculate " + node["output_field"] + " from "
                           + node["left_field"] + symbol + node["right_field"] + ".")
        elif op == "derive_date_gap":
            clauses.append("Calculate elapsed days between " + node["left_field"]
                           + " and " + node["right_field"] + ".")
        elif op == "count":
            clauses.append("Count the returned records after the preceding filters and deduplication.")
        elif op == "filter_gte":
            clauses.append("Keep " + node["field"] + " greater than or equal to "
                           + str(node["value"]) + ".")
    return clauses


def reasoning_action_issues(reasoning: str, assignments: list[dict], *,
                            question: str = "", visible_text: str | None = None) -> list[str]:
    """High-precision action/meaning checks, separate from LLM prose review."""
    issues = []
    if not isinstance(reasoning, str):
        return ["reasoning_not_text"]
    if visible_text is not None:
        identifiers = re.compile(
            r"\b(?:Dossier-[A-Za-z0-9]+|PRJ-[A-Za-z0-9]+|ENT-[A-Za-z0-9]+|"
            r"EVT-[A-Za-z0-9]+|OBS-[A-Za-z0-9]+|Case-\d+|Entry-\d+|"
            r"KEY[12]-[A-Za-z0-9]+|MK-[A-Za-z0-9]+|RK-[A-Za-z0-9]+)\b",
            re.I,
        )
        known = {value.casefold() for value in identifiers.findall(visible_text)}
        unknown = sorted({value for value in identifiers.findall(reasoning)
                          if value.casefold() not in known})
        if unknown:
            from difflib import get_close_matches
            received = {value.casefold(): value for value in identifiers.findall(visible_text)}
            details = []
            for value in unknown:
                matches = get_close_matches(value.casefold(), sorted(known), n=2, cutoff=0.85)
                hints = [received[key] for key in matches]
                details.append(value + (
                    " (already received spellings: " + ", ".join(hints) + ")" if hints else ""
                ))
            issues.append("unobserved_identifier_in_reasoning:" + "; ".join(details))
    if assignments and all(not a.get("source_ids") and not a.get("tools")
                           for a in assignments):
        # Reading an already returned record is valid. Promise of a new
        # external source/tool acquisition is not.
        if re.search(
            r"\b(?:I|we)\s+(?:must|will|need to|should)\s+"
            r"(?:now\s+)?(?:re[- ]?)?(?:query|browse|search|fetch|open)\b"
            r".{0,90}\b(?:source|web|website|image|audio|video|briefing|table)\b",
            reasoning, re.I,
        ):
            issues.append("source_free_action_claims_external_acquisition")
    instructions = " ".join(a.get("target_instruction", "") for a in assignments)
    if ("elapsed days between date_a, date_b" in instructions
            or "elapsed days between date_a and date_b" in instructions
            or "midpoint between date_a, date_b" in instructions):
        if re.search(
            r"\b(?:earliest\s+(?:and|to)\s+(?:the\s+)?latest|"
            r"minimum\s+and\s+maximum|first\s+and\s+last)\s+"
            r"(?:(?:recorded|observation|logged)\s+)?dates?\b",
            reasoning, re.I,
        ):
            issues.append("date_pair_replaced_with_observation_date_extrema")
    fields = {f for a in assignments for f in a.get("output_fields", [])}
    if ("observation_id" in fields and "observation_year" in fields
            and not any(f.startswith(("date_", "latitude", "longitude")) for f in fields)):
        if re.search(
            r"\b(?:retrieve|retrieving|extract|extracting|query|querying)\b"
            r"[^.!?]{0,110}\b(?:dates?\s+and\s+coordinates?|coordinates?\s+and\s+dates?)\b",
            reasoning, re.I,
        ):
            issues.append("observation_year_lookup_claims_date_coordinate_extraction")
    if question:
        for match in re.finditer(
            r"\bfinal answer requires[^.!?]{0,80}\b(?:label|name|field)\b"
            r"[^.!?]{0,25}\b(entity_\d+)\b", reasoning, re.I,
        ):
            if match[1].casefold() not in question.casefold():
                issues.append("intermediate_identifier_claimed_as_final_place_name")
        for match in re.finditer(
            r"\bquestion\s+(?:explicitly\s+)?(?:requires|demands|asks for)"
            r"[^.!?]{0,85}?\b(entity_\d+)\b", reasoning, re.I,
        ):
            if match[1].casefold() not in question.casefold():
                issues.append("intermediate_field_claimed_as_question_output")
    return list(dict.fromkeys(issues))
