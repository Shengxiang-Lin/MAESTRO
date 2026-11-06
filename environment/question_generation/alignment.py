"""Evidence-linked language review; this is not a formal equivalence proof."""
from __future__ import annotations

import json
import re
from copy import deepcopy
from dataclasses import replace

from environment.trajectory_simulation.eligibility import digest
from .public_semantics import public_question_contract

ALIGNMENT_VERSION = 1
SCHEMA = {
    "type": "object",
    "properties": {
        "alignments": {
            "type": "array", "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "status": {"type": "string", "enum": ["match", "mismatch", "uncertain"]},
                    "spans": {"type": "array", "items": {"type": "integer"}},
                    "reason": {"type": "string", "maxLength": 160},
                },
                "required": ["id", "status", "spans", "reason"],
                "additionalProperties": False,
            },
        },
        "extra_constraints": {
            "type": "array", "items": {
                "type": "object", "properties": {
                    "span": {"type": "integer"},
                    "reason": {"type": "string", "maxLength": 160},
                }, "required": ["span", "reason"], "additionalProperties": False,
            },
        },
    },
    "required": ["alignments", "extra_constraints"], "additionalProperties": False,
}


def question_spans(question):
    # Clauses keep offsets and exact text. The reviewer returns integer IDs,
    # avoiding quote transcription errors, including commas inside numbers.
    boundaries = [0]
    boundaries.extend(m.end() for m in re.finditer(
        r"[;?.]\s+|,(?!\d)\s+|\n+", question,
    ))
    boundaries.append(len(question))
    return [{"id": i, "start": a, "end": b, "text": question[a:b]}
            for i, (a, b) in enumerate(zip(boundaries, boundaries[1:])) if a < b]


def check_alignment(gold, world, canonical, question, backend, *, executor_view=False):
    contract = public_question_contract(world, canonical, gold)
    spans = question_spans(question)
    rules = (
        "Audit each public obligation against the candidate text independently. "
        "Do not solve the task or use world knowledge. For every obligation return "
        "its exact id ONCE, status and candidate span IDs supporting that judgment. "
        "Interpret the actual candidate clause and its referents, then compare "
        "its owner, endpoints, value, operator and output with the contract. "
        "A retained number alone is NOT a match if its owner changed. "
        "An anchor's date cannot transfer to another person reached through a place. "
        "Directed spouse/about/winner relations cannot be assumed symmetric. "
        "Find the names / identify entities by name / which entities are equivalent "
        "for QUERY_NAME. How many / count distinct entities are equivalent COUNT. "
        "VERIFY asks whether ANY recorded value passes, not whether all pass. "
        "SELECT ranks over the full input pool including ties. INTERSECT requires "
        "the same entities; UNION allows either complete branch. "
        "FIND_ALL means discovery within task evidence, not an initially supplied list. "
        "Natural synonyms, pronouns with clear referents, grouped filters, and multiple "
        "sentences are allowed. Do not require procedural steps in the question. "
        "For a missing condition cite the relevant incomplete clause; for ambiguity "
        "return uncertain. Use short factual reasons, no deliberation. "
        "Report every invented restriction or output/resource assumption in "
        "extra_constraints (empty if none). There is no global verdict to guess. "
        "The canonical question resolves null public labels. Never approve an "
        "ambiguous direction solely because other conditions match."
    )
    from environment.trajectory_simulation.eligibility import semantic_review_nodes, semantic_review_steps
    steps = semantic_review_steps(semantic_review_nodes(gold, world))
    meanings = {f"S{s['step']}": s["meaning"] for s in steps}
    items, extras, batches = [], [], []
    # Bound the required output, not the inspected semantics. Each batch sees
    # the entire candidate and graph but answers only four obligations. This
    # prevents a truncated list from masquerading as complete coverage.
    obligations = contract["obligations"]
    client = replace(backend, temperature=0.0, max_tokens=max(2048, backend.max_tokens))
    for start in range(0, len(obligations), 4):
        required = obligations[start:start+4]
        if executor_view:
            required = [{"id": o["id"], "meaning": meanings[o["id"]]} for o in required]
            view = {"executor_steps": [
                {"id": f"S{s['step']}", "inputs": s["input_steps"], "meaning": s["meaning"]}
                for s in steps
            ]}
        else:
            view = {"scopes": contract["scopes"]}
        request = json.dumps({
            **view, "canonical_question": canonical,
            "candidate_spans": spans, "required_obligations": required,
        }, ensure_ascii=False, separators=(",", ":"))
        schema = deepcopy(SCHEMA)
        entries = schema["properties"]["alignments"]
        entries["minItems"] = entries["maxItems"] = len(required)
        properties = entries["items"]["properties"]
        properties["id"]["enum"] = [o["id"] for o in required]
        properties["reason"]["maxLength"] = 96
        properties["spans"].update(minItems=1, maxItems=4)
        properties["spans"]["items"]["enum"] = [s["id"] for s in spans]
        schema["properties"]["extra_constraints"]["maxItems"] = 2
        schema["properties"]["extra_constraints"]["items"]["properties"]["span"]["enum"] = [
            s["id"] for s in spans
        ]
        part = client.complete_json(
            rules + " Return alignments ONLY for required_obligations. "
            "The whole graph is context, not additional IDs to report. "
            "Respect executor meanings literally even if the synthetic relations "
            "look unnatural in real life. An unchanged canonical clause is valid "
            "by construction; do not reinterpret it. Evaluate ALL candidate spans "
            "together to resolve pronouns and owners. Reasons: at most 12 words. "
            "Do not copy long question text into reasons. Return each required ID once.",
            request, response_schema=schema,
        )
        batches.append(part)
        part_items, part_extras = part.get("alignments"), part.get("extra_constraints")
        expected_batch = {o["id"] for o in required}
        if (not isinstance(part_items, list)
                or not all(isinstance(v, dict) for v in part_items)
                or len(part_items) != len(expected_batch)
                or {v.get("id") for v in part_items} != expected_batch
                or not isinstance(part_extras, list)):
            items = None
            break
        items.extend(part_items)
        extras.extend(part_extras)
    verdict = {"alignments": items, "extra_constraints": extras, "batches": batches}
    expected = {o["id"] for o in contract["obligations"]}
    allowed = {s["id"] for s in spans}
    valid = (
        isinstance(items, list) and isinstance(extras, list)
        and len(items) == len(expected)
        and all(isinstance(v, dict) for v in items)
        and {v.get("id") for v in items} == expected
        and all(v.get("status") in ("match", "mismatch", "uncertain")
                and isinstance(v.get("reason"), str) and len(v["reason"]) <= 160
                and isinstance(v.get("spans"), list) and v["spans"]
                and all(type(s) is int and s in allowed for s in v["spans"])
                for v in items)
        and all(isinstance(v, dict) and type(v.get("span")) is int
                and v["span"] in allowed and isinstance(v.get("reason"), str)
                and 0 < len(v["reason"]) <= 160 for v in extras)
    )
    issues = ([v for v in items if v["status"] != "match"] + extras) if valid else [
        "invalid_alignment_coverage",
    ]
    eligible = valid and not issues
    return {
        "eligible": eligible, "reviewed": True,
        "decision": "eligible" if eligible else (
            "mismatch" if valid else "review_error"
        ),
        "method": "llm_scoped_programir_screen" if executor_view else "llm_contract_alignment", "issues": issues,
        **({"review_failure_kind": "invalid_alignment_coverage"} if not valid else {}),
        "question_sha256": digest(question), "program_sha256": digest(world["program"]),
        "alignment_version": ALIGNMENT_VERSION, "contract_sha256": digest(contract),
        "candidate_spans": spans, "verdict": verdict,
    }
