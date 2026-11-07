"""Candidate-only, question-answer probes of PUBLIC Stage-2 semantics.

The compiler derives expected answers from the frozen public ScopedPlan, but
the probing model sees ONLY the natural candidate plus short questions.
Probes are conservative targeted diagnostics, not an English equivalence proof.
High-risk owner and conditional probes survive paraphrases without group labels.
"""
from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

from .scoped_surface import ScopedPlan

_ORDINALS = ("first", "second", "third", "fourth", "fifth", "sixth",
             "seventh", "eighth")
_ORDINAL = "|".join(_ORDINALS)
_ALLOWED = frozenset(("AND", "OR", "SELF", "MEMBER",
                      "ALL_MUST", "NOT_ALL", "SELECTED", "RELATED",
                      "CONDITIONAL", "ASSUMED", "UNCLEAR",
                      "FORWARD", "BACKWARD", "DIRECT", "OTHER",
                      "COUNT", "ENTITIES", "ATTRIBUTE", "RELATIONS",
                      "RANK", "VERIFY"))
_EVIDENCE_FIELDS = {"evidence", "subject", "object", "subject_role",
                    "object_role", "owner"}
_EVIDENCE_SYSTEM = """You are an independent reader of candidate text only.
The probe anchors locate clauses; they are NOT instructions to repair the text.
Extract what the candidate actually says, even if it is inconsistent.
For a relation, quote the actual subject, object and complete local clause.
Subject means predicate FROM endpoint, object means predicate TO endpoint.
For passive 'X is a target of r from Y', subject=Y and object=X.
Map noun phrases to the LOCAL A/B endpoints of that particular hop.
Never substitute the final-answer entity for an intermediate endpoint.
For attribute ownership, quote the owning noun phrase and the condition
including its attachment. Never infer ownership from likely entity types.
Return exact contiguous verbatim quotes, with no ellipses or paraphrasing.
The evidence quote MUST contain your quoted subject/object or owner noun
phrases, not only the verb or number. Use a longer span across sentences
when a group name is defined earlier.
For owner probes, the evidence must also contain the condition's actual
value (allowing equivalent number/date notation). A later group definition
alone does not establish an earlier property: include that earlier clause.
The owner quote must be the minimal entity noun phrase, excluding relative
clauses such as "whose ...", "that has ..." and the attribute condition.
If either the clause, owner or endpoint is not recoverable, answer UNCLEAR.
Answer type (entities/count/attribute/relations/boolean) is independent of
ranking (ARGMAX/ARGMIN) and explicit inclusion of ties.
Return JSON {"answers":[{"id":"P1","answer":"...","evidence":"..."}]}.
Relation probes additionally require subject, object, subject_role and
object_role (A/B). Owner probes additionally require owner. Use empty strings
for unrecoverable quotes. Every non-UNCLEAR answer needs evidence.
"""


def _evidence_schema(probes):
    """Per-kind fields prevent small readers treating owner probes as hops."""
    alternatives = []
    labels = {
        "relation_path": ["A_TO_B", "B_TO_A", "UNCLEAR"],
        "condition_owner": ["A", "B", "UNCLEAR"],
        "answer_type": ["COUNT", "ENTITIES", "ATTRIBUTE", "RELATIONS", "VERIFY", "UNCLEAR"],
        "selection_operation": ["ARGMAX", "ARGMIN", "ALL", "THRESHOLD", "UNCLEAR"],
        "ranking_ties": ["INCLUDE", "EXCLUDE", "UNCLEAR"],
    }
    for p in probes:
        properties = {"id": {"const": p["id"]}, "answer": {"type": "string"},
                      "evidence": {"type": "string"}}
        if p["kind"] in labels:
            properties["answer"] = {"enum": labels[p["kind"]]}
        if p["kind"] == "relation_path":
            properties.update({key: {"type": "string"} for key in
                               ("subject", "object", "subject_role", "object_role")})
        elif p["kind"] == "condition_owner":
            properties["owner"] = {"type": "string"}
        alternatives.append({"type": "object", "properties": properties,
                             "required": list(properties), "additionalProperties": False})
    return {"type": "object", "properties": {"answers": {
        "type": "array", "minItems": len(probes), "maxItems": len(probes),
        "items": {"anyOf": alternatives}}},
        "required": ["answers"], "additionalProperties": False}
_PROBE_SYSTEM = """You are a candidate-only semantic reader.
Read ONLY the candidate question below. You are NOT given the reference,
gold answer, expected probe answers or private world. For each requested
probe, determine what the CANDIDATE really says, regardless of what seems
logical or likely to have been intended. Distinguish membership of the
result entity from membership of one of its members. For an attribute-owner
question, distinguish a property of the SELECTED entity from a property
of a RELATED entity reached by a named relation. Do not infer ownership
from which entity type is most likely to have that attribute.
'Either/or' permits either branch; 'both' on the SAME entity requires
the intersection. For relation-path questions, FORWARD means selected
entities are TARGETS of a predicate from another entity, BACKWARD means
selected entities have the predicate pointing TO another entity; DIRECT
means an attribute/name filter with no directed predicate, OTHER means
a weaker/different predicate. Infer from the candidate, not from the
question's alternative wording. For answer-type questions, select
COUNT, ENTITIES, ATTRIBUTE, RELATIONS, RANK, VERIFY or UNCLEAR.
If the candidate does not clearly specify the requested distinction, use
UNCLEAR instead of guessing. Return JSON only with exactly one answer
per probe ID, as {"answers":[{"id":"P1","answer":"AND"}]}.
Do not give explanations, intermediate reasoning or extra fields.
"""
_PROBE_ITEM_SCHEMA = {
    "type": "object",
    "properties": {
        "id": {"type": "string"},
        "answer": {"enum": sorted(_ALLOWED)},
    },
    "required": ["id", "answer"],
    "additionalProperties": False,
}
_PROBE_SCHEMA = {
    "type": "object",
    "properties": {"answers": {"type": "array", "items": _PROBE_ITEM_SCHEMA}},
    "required": ["answers"],
    "additionalProperties": False,
}


class CandidateProbeFormatError(ValueError):
    """A candidate-only probe response is missing or malformed."""

    def __init__(self, message: str, *, expected=(), received=()):
        super().__init__(message)
        self.diagnostic = {
            "expected_ids": list(expected),
            "received_ids": list(received),
            "received_count": len(received),
        }


def _candidate_defines_group(candidate: str, ordinal: str) -> bool:
    idx = _ORDINALS.index(ordinal) + 1
    label = rf"(?:the\s+)?(?:{ordinal}\s+group|group\s*{idx})"
    return bool(re.search(
        rf"\b{label}\s+(?:consists?|contains?|includes?|comprises?|"
        rf"is\s+defined|to\s+include|to\s+consist)\b",
        candidate, flags=re.I,
    ))


def _surface_join(node: Mapping[str, Any]) -> str | None:
    """Peel unary public filters, never traverse a relation to guess a join."""
    current = node
    for _ in range(5):
        op = str(current.get("definition", ""))
        if op in ("INTERSECT", "UNION"):
            return "AND" if op == "INTERSECT" else "OR"
        if not op.startswith("FILTER_"):
            return None
        inputs = current.get("inputs") or []
        if len(inputs) != 1 or not isinstance(inputs[0], Mapping):
            return None
        current = inputs[0]
    return None


def _has_relation(node: Mapping[str, Any], relation: str, *, depth=4) -> bool:
    """Search a bounded public-meaning subtree; never inspect World/Gold."""
    if depth < 0 or not isinstance(node, Mapping):
        return False
    if (node.get("definition") == "RELATE"
            and str(node.get("relation", "")).casefold() == relation.casefold()):
        return True
    return any(
        _has_relation(child, relation, depth=depth - 1)
        for child in node.get("inputs", ())
        if isinstance(child, Mapping)
    )


def _condition_at_selected_branch(node: Mapping[str, Any]) -> list[dict]:
    """Top-level filter restrictions on a branch's returned entity set.

    Never walk through RELATE to infer a child's owner: relation endpoints
    would then be conflated. A unary FILTER chain retains the same owner.
    """
    found = []
    current = node
    for _ in range(5):
        if current.get("definition") == "RELATE":
            break
        for cond in current.get("conditions", ()):
            if (isinstance(cond, Mapping)
                    and cond.get("attribute")
                    and isinstance(cond.get("value"), Mapping)
                    and cond["value"].get("type") in ("number", "year", "date")):
                found.append(cond)
        op = str(current.get("definition", ""))
        if not op.startswith("FILTER_"):
            break
        children = current.get("inputs") or []
        if len(children) != 1 or not isinstance(children[0], Mapping):
            break
        current = children[0]
    return found


def _owner_probes(
    meaning: Mapping[str, Any], ordinal: str,
) -> list[dict[str, str]]:
    """Public-plan owner probes for a joined result and a related location.

    Only assert SELECTED when the condition is on the top level of an
    intersected branch and an alternative branch has a location relation.
    This is deliberately narrower than general semantic readback.
    """
    current = meaning
    for _ in range(4):
        if current.get("definition") == "INTERSECT":
            break
        if not str(current.get("definition", "")).startswith("FILTER_"):
            return []
        children = current.get("inputs") or []
        if len(children) != 1 or not isinstance(children[0], Mapping):
            return []
        current = children[0]
    if current.get("definition") != "INTERSECT":
        return []
    branches = [b for b in current.get("inputs", ())
                if isinstance(b, Mapping)]
    if len(branches) < 2 or not any(_has_relation(b, "location") for b in branches):
        return []
    results = []
    seen = set()
    for branch in branches:
        for cond in _condition_at_selected_branch(branch):
            attribute = str(cond["attribute"])
            cmp = str(cond.get("comparison", "equal to"))
            value = str(cond["value"].get("value", ""))
            identifier = (attribute.casefold(), cmp.casefold(), value)
            if identifier in seen or not value:
                continue
            seen.add(identifier)
            results.append({
                "kind": "condition_owner", "group": ordinal,
                "attribute": attribute,
                "comparison": cmp, "value": value,
                "expected": "SELECTED",
                "question": (
                    f"In the candidate question, when determining entities "
                    f"using the location relation and the condition "
                    f"'{attribute} {cmp} {value}', which entity must satisfy "
                    f"that condition? Is it the entity being SELECTED for "
                    "the result set (SELECTED), or a different place/area "
                    "reached via the location relation (RELATED)? "
                    "Reply SELECTED, RELATED or UNCLEAR."
                
                ),
            })
    return results[:2]


def compile_candidate_probes(
    plan: ScopedPlan, reference: str, candidate: str,
) -> list[dict[str, str]]:
    """Generate a bounded number of *public* contrast questions.

    Owner/conditional probes bind to PUBLIC attribute/relation anchors, not
    fragile candidate group labels. Generic joins still require identifiable
    group descriptions. Expected answers NEVER enter a model request.
    """
    if not isinstance(plan, ScopedPlan):
        raise TypeError("candidate_probes_need_public_scoped_plan")
    candidates = []
    for group, meaning in zip(plan, plan.meanings, strict=True):
        if not group or not isinstance(meaning, Mapping):
            continue
        excerpt = str(group[0])
        found = re.match(
            rf"Let the ({_ORDINAL}) group consist\b", excerpt, flags=re.I,
        )
        if not found:
            continue
        ordinal = found.group(1).lower()
        # Owner checks must run even for natural "Second, define a set..."
        # rewrites without the mechanical words "second group".
        candidates.extend(_owner_probes(meaning, ordinal))
        if not _candidate_defines_group(candidate, ordinal):
            continue

        join = _surface_join(meaning)
        if join:
            candidates.append({
                "kind": "group_join", "group": ordinal,
                "expected": join,
                "question": (
                    f"When the candidate defines the {ordinal} group, does "
                    "the SAME entity have to satisfy BOTH branch conditions "
                    "(AND), or is satisfying EITHER branch enough (OR)? "
                    "Reply AND, OR or UNCLEAR."
                ),
            })

        # Guard a specific scoping contrast: the entity ITSELF belongs
        # to the second intersected group vs one of its members belongs.
        m = re.search(
            rf"\bentities in both entities with at least one member in the "
            rf"({_ORDINAL}) group and the ({_ORDINAL}) group\b",
            excerpt, flags=re.I,
        )
        if m:
            member_group, own_group = m.group(1).lower(), m.group(2).lower()
            candidates.append({
                "kind": "member_owner", "group": ordinal,
                "expected": "SELF",
                "question": (
                    f"When defining the {ordinal} group, must the candidate "
                    f"entity ITSELF belong to the {own_group} group (SELF), "
                    f"or is it enough for one of its members to belong to "
                    f"the {own_group} group (MEMBER)? Reply SELF, MEMBER "
                    "or UNCLEAR."
                ),
            })

    # A union with one branch derived from Group N and another genuinely
    # independent branch: do not accidentally restrict the second branch
    # to Group N. This probe is intentionally limited to explicit syntax.
    final = re.search(
        rf"\bAmong\b[^\?]{{0,300}}?\bdrawn from the ({_ORDINAL}) "
        rf"group or entities in both\b",
        reference, flags=re.I,
    )
    if final:
        ordinal = final.group(1).lower()
        if re.search(rf"\b(?:the\s+)?{ordinal}\s+group\b", candidate, re.I):
            candidates.append({
                "kind": "final_union_scope", "group": ordinal,
                "expected": "NOT_ALL",
                "question": (
                    f"In the final selection, MUST EVERY eligible candidate "
                    f"belong to the {ordinal} group (ALL_MUST), or can the "
                    f"alternative OR branch admit entities that are not in "
                    f"that group (NOT_ALL)? Reply ALL_MUST, NOT_ALL or UNCLEAR."
                ),
            })

    # Boundary check: a restriction taken from Group 1 to construct Group 2
    # must NOT be attached to Group 1 itself. This is grounded in the PUBLIC
    # reference wording, not any hidden entity or execution result.
    first_and_second = re.search(
        r"Let the first group consist of (?P<first>.*?)\. "
        r"Let the second group consist of (?P<second>.*?)\.",
        reference, re.I | re.S,
    )
    if first_and_second:
        first_text = first_and_second.group("first")
        second_text = first_and_second.group("second")
        pop = re.search(
            r"with a population number of (?P<value>[\d,]+) drawn from "
            r"the first group", second_text, re.I,
        )
        if (pop and not re.search(r"population number", first_text, re.I)
                and re.search(r"\bfirst group\b", candidate, re.I)):
            candidates.append({
                "kind": "group_identity", "group": "first",
                "expected": "NOT_ALL", "value": pop.group("value"),
                "question": (
                    "Does membership in the FIRST group itself require "
                    f"population {pop.group('value')} (ALL_MUST), or is "
                    "that population restriction applied later to a subset "
                    "drawn FROM the first group (NOT_ALL)? "
                    "Reply ALL_MUST, NOT_ALL or UNCLEAR."
                ),
            })

    # An explicit branch that can be empty is NOT a promise that the anchor
    # satisfies its condition. This probe does not require group labels in
    # the naturalized candidate.
    conditional = re.search(
        r"Let the (?P<group>\w+) group consist of (?P<anchor>.+?) "
        r"if it has an? (?P<attr>[\w\s]+?) of (?P<value>[\w+.-]+)\. "
        r"Otherwise, this group is empty\.",
        reference, re.I,
    )
    if conditional:
        candidates.append({
            "kind": "conditional_scope", "group": conditional.group("group"),
            "expected": "CONDITIONAL",
            "anchor": conditional.group("anchor"),
            "attribute": conditional.group("attr"),
            "value": conditional.group("value"),
            "question": (
                f"Regarding {conditional.group('anchor')} having "
                f"{conditional.group('attr')} {conditional.group('value')}, "
                "does the candidate retain a genuine IF check that allows "
                "the eligible set to be empty when the condition fails "
                "(CONDITIONAL), or does it assume the condition is true "
                "without handling failure (ASSUMED)? "
                "Reply CONDITIONAL, ASSUMED or UNCLEAR."
            ),
        })

    # Prefer scope-sensitive probes over generic join questions when bounded.
    rank = {"conditional_scope": 0, "condition_owner": 1,
            "group_identity": 2, "member_owner": 3,
            "final_union_scope": 4, "group_join": 5}
    chosen = sorted(candidates, key=lambda item: (rank[item["kind"]], item["group"]))[:6]
    return [{"id": f"P{i}", **item} for i, item in enumerate(chosen, start=1)]


def explicit_probe_contradictions(
    probes: list[dict[str, str]], candidate: str,
) -> list[str]:
    """Catch only HIGH-CONFIDENCE scope flips in a labeled candidate.

    These checks are intentionally narrower than a full semantic parser.
    An unsupported paraphrase is left to the candidate-only probe reader.
    """
    issues = []
    for probe in probes:
        ordinal = probe["group"]
        kind = probe["kind"]
        if (kind == "condition_owner" and probe.get("expected") == "B"
                and probe.get("direction") in ("forward", "backward")
                and not any(other.get("kind") == "condition_owner"
                            and other.get("expected") == "A"
                            and all(other.get(key) == probe.get(key) for key in
                                    ("relation", "direction", "attribute", "comparison",
                                     "value_type", "canonical_value"))
                            for other in probes)
                and _related_relative_owns_condition(probe, candidate)):
            issues.append(f"{probe['id']}:result_filter_moved_to_input_endpoint")
        if kind == "conditional_scope" and probe["expected"] == "CONDITIONAL":
            if (re.search(r"\bassuming\b", candidate, re.I)
                    and not re.search(
                        r"\b(?:otherwise|if not|unless|failing that|"
                        r"if (?:it|the airport) does not)\b",
                        candidate, re.I,
                    )):
                issues.append(f"{probe['id']}:conditional_replaced_by_assumption")
        elif kind == "group_identity" and probe["expected"] == "NOT_ALL":
            value = re.escape(str(probe["value"]))
            # The exact v18 task-5 drift: a new population filter is used
            # *to form the first group* even though it belongs downstream.
            if re.search(
                rf"\bpopulation\s+(?:number\s+)?(?:of\s+)?"
                rf"{value}\b[^.;!?]{{0,70}}\bto form the first group\b",
                candidate, re.I,
            ):
                issues.append(f"{probe['id']}:derived_filter_moved_to_first_group")
        elif kind == "condition_owner" and probe["expected"] == "SELECTED":
            # v17 #3: reference intersects (a) a location-relation result
            # with (b) an entity filter, while candidate says that entities
            # are LOCATED IN BOTH kinds of administrative areas. This moves
            # the second filter from result entities to locations.
            matched = re.search(
                rf"\b(?:the\s+)?{ordinal}\s+group\s+"
                rf"(?:consists?|includes?|contains?)\b"
                rf"(?P<body>[^.;!?]{{0,260}})",
                candidate, flags=re.I,
            )
            misplaced_labelled = bool(
                matched and re.search(
                    r"\bentities\s+located\s+in\s+both\b",
                    matched.group("body"), re.I,
                )
            )
            # v18 real task 3: "Second, define a set of entities that
            # exist in administrative areas meeting both ... population
            # < 1,765." No artificial 'second group' label survives.
            # Read the nearby numeric constraint as an additional anchor
            # so an unrelated 'in both' clause cannot trip this guard.
            value = str(probe.get("value", ""))
            numeric = re.escape(value.replace(",", ""))
            label_free = re.search(
                r"\b(?:entities\s+that\s+)?(?:exist|are\s+located)\s+in\s+"
                r"(?:administrative\s+)?areas\s+(?:meeting|satisfying)\s+both\b"
                r"(?P<body>[^.!?]{0,300})",
                candidate, re.I,
            )
            misplaced_unlabelled = bool(
                label_free and numeric and re.search(
                    rf"\b{numeric}\b",
                    label_free.group("body").replace(",", ""),
                    re.I,
                )
            )
            if misplaced_labelled or misplaced_unlabelled:
                issues.append(
                    f"{probe['id']}:selected_filter_moved_to_related_location"
                )
        elif kind == "group_join" and probe["expected"] == "AND":
            # Conjoining two entity noun phrases lists sets, rather than
            # asserting that the SAME entity satisfies both restrictions.
            # Keep decimal points inside the bounded definition clause.
            definition = re.search(
                rf"\b(?:the\s+)?{ordinal}\s+group\s+"
                rf"(?:includes?|consists? of|contains?|comprises?)\s+",
                candidate, flags=re.I,
            )
            if definition:
                body = re.split(r"[;!?]|\.(?!\d)",
                                candidate[definition.end():definition.end() + 800],
                                maxsplit=1)[0]
                entity_head = (
                    r"(?:entities|people|organizations|countries|cities|places|"
                    r"events|ways|administrative areas|creative works|TV series|"
                    r"Award entities)"
                )
                explicit_same_entity = re.search(
                    r"\bboth\b|\b(?:each|every|same)\s+(?:result\s+)?entit(?:y|ies)\b",
                    body, flags=re.I,
                )
                if (not explicit_same_entity
                        and re.match(rf"(?:the\s+)?{entity_head}\b", body, re.I)
                        and re.search(
                            rf"\b(?:AND|as well as)\s+(?:the\s+)?{entity_head}\b",
                            body, flags=re.I)):
                    issues.append(f"{probe['id']}:intersection_replaced_by_entity_list")
            # v16 #3: "second group includes areas that are either ... or ...".
            # Do not search outside the one local group-definition clause.
            m = re.search(
                rf"\b(?:the\s+)?{ordinal}\s+group\s+"
                rf"(?:includes?|consists? of|to include)\b"
                rf"(?P<body>[^.;!?]{{0,200}})",
                candidate, flags=re.I,
            )
            if m and re.search(r"\beither\b[^.;!?]{0,130}\bor\b",
                               m.group("body"), flags=re.I):
                issues.append(f"{probe['id']}:explicit_intersection_to_either_or")
        elif kind == "member_owner" and probe["expected"] == "SELF":
            # v16 #8: "entity belongs to group 4" became "one member
            # belongs to both group 3 and group 4".
            group = rf"(?:the\s+)?{ordinal}\s+group"
            start = re.search(
                rf"\b{group}\s+(?:consists?|includes?|contains?)\b"
                rf"(?P<body>[^.;!?]{{0,260}})", candidate, flags=re.I,
            )
            if start and re.search(
                rf"\bmembers?\s+in\s+both\s+(?:the\s+)?(?:{_ORDINAL}) "
                rf"group\s+and\s+(?:the\s+)?(?:{_ORDINAL}) group\b",
                start.group("body"), flags=re.I,
            ):
                issues.append(f"{probe['id']}:member_scope_moved_into_intersection")
        elif kind == "final_union_scope" and probe["expected"] == "NOT_ALL":
            # v16 #5: a final OR over two branches became a selection
            # exclusively among entities already in Group 2.
            if re.search(
                rf"\bAmong entities in (?:the\s+)?{ordinal}\s+group that\b",
                candidate, flags=re.I,
            ):
                issues.append(f"{probe['id']}:union_branch_restricted_to_group")
    return issues


def _contains_condition_value(probe, text):
    if probe.get("value_type") in ("number", "year", "date"):
        from .semantic_checks import missing_typed_literals
        tiny = {"definition": "FIND_ALL", "inputs": [], "conditions": [{
            "value": {"type": probe["value_type"], "value": probe["canonical_value"]}}]}
        return not missing_typed_literals(ScopedPlan([], [], tiny), text)
    raw = str(probe.get("canonical_value", ""))
    return bool(raw and re.search(r"(?<![\w+])" + re.escape(raw) + r"(?!\w)", text, re.I))


def _related_relative_owns_condition(probe, candidate):
    """Only recognize an explicit local 'input that has ... AND attr value'.

    End at sentence/clause boundaries or a new entity subject; unsupported
    English remains the reader's responsibility.
    """
    relation = re.escape(probe["relation"])
    if probe["direction"] == "backward":
        prefix = rf"(?:recorded\s+)?['\"]?{relation}['\"]?\s+points?\s+to\s+"
    else:
        prefix = rf"targets?\s+of\s+(?:the\s+)?['\"]?{relation}['\"]?\s+(?:relation\s+)?from\s+"
    pattern = prefix + r"[^.;!?,]{1,120}?\s+that\s+has\s+"
    for match in re.finditer(pattern, candidate, re.I):
        tail = candidate[match.end():match.end() + 350]
        body = re.split(r"[;!?]|\.(?!\d)|(?<!\d),(?!\d)|"
                        r"\b(?:and|but)\s+(?:are|have|they|these|those|themselves)\b",
                        tail, maxsplit=1, flags=re.I)[0]
        attribute = re.escape(probe["attribute"])
        for cond in re.finditer(
                rf"(?:^|\bAND\s+(?:an?\s+)?)(?:recorded\s+)?['\"]?{attribute}['\"]?\s+",
                body, re.I):
            span = re.split(r"\bAND\b", body[cond.end():], maxsplit=1, flags=re.I)[0]
            if _contains_condition_value(probe, span):
                return True
    return False


def _grounded_answer(probe, item, candidate):
    """Evidence integrity check; endpoint interpretation remains a model judgment."""
    answer = item["answer"]
    if answer == "UNCLEAR":
        return answer
    evidence = item.get("evidence")
    if not isinstance(evidence, str) or not evidence.strip() or evidence not in candidate:
        return "UNCLEAR"
    if probe["kind"] == "ranking_attribute":
        # Strip selection wording, not attribute semantics; no expected label
        # is used to repair or constrain the independent reader's answer.
        answer = re.sub(r"^(?:(?:the|largest|smallest|greatest|least|highest|lowest|"
                        r"maximum|minimum|recorded)\s+)+", "", answer.strip(), flags=re.I)
        answer = answer.strip("'\"")
        if not answer or not re.search(r"(?<!\w)" + re.escape(answer) + r"(?!\w)", evidence, re.I):
            return "UNCLEAR"
    if probe["kind"] == "relation_path":
        roles = (item.get("subject_role"), item.get("object_role"))
        derived = {("A", "B"): "A_TO_B", ("B", "A"): "B_TO_A"}.get(roles)
        if derived != answer:
            return "UNCLEAR"
        for key in ("subject", "object"):
            quote = item.get(key)
            if not isinstance(quote, str) or not quote.strip() or quote not in evidence:
                return "UNCLEAR"
        if item["subject"] == item["object"]:
            return "UNCLEAR"
    if probe["kind"] == "condition_owner" and probe.get("evidence_required"):
        owner = item.get("owner")
        if (answer not in ("A", "B") or not isinstance(owner, str)
                or not owner.strip() or owner not in evidence):
            return "UNCLEAR"
        if re.search(r"\b(?:whose|that has|which has|who has)\b", owner, re.I):
            return "UNCLEAR"
        if not _contains_condition_value(probe, evidence):
            return "UNCLEAR"
    return answer


# v23: explicitly separate model semantic answers from quote grounding.
# These fields are candidate-only audit data, never private World/Gold.
_DIAGNOSTIC_FIELDS = {"raw_answer", "evidence_status", "evidence_reason"}


def _evidence_failure_reason(probe, item, candidate):
    """Return a bounded local formatting/grounding reason, not a semantic vote."""
    if item.get("answer") == "UNCLEAR":
        return "model_uncertain"
    quote = item.get("evidence")
    if not isinstance(quote, str) or not quote.strip():
        return "evidence_missing"
    if quote not in candidate:
        return "evidence_not_verbatim"
    kind = probe["kind"]
    if kind == "relation_path":
        roles = (item.get("subject_role"), item.get("object_role"))
        mapped = {("A", "B"): "A_TO_B", ("B", "A"): "B_TO_A"}.get(roles)
        if mapped != item["answer"]:
            return "relation_role_label_inconsistent"
        for name in ("subject", "object"):
            value = item.get(name)
            if not isinstance(value, str) or not value.strip() or value not in quote:
                return f"relation_{name}_quote_invalid"
        if item.get("subject") == item.get("object"):
            return "identical_relation_endpoints"
    elif kind == "condition_owner":
        owner = item.get("owner")
        if item.get("answer") not in ("A", "B"):
            return "owner_role_label_invalid"
        if not isinstance(owner, str) or not owner.strip() or owner not in quote:
            return "owner_quote_invalid"
        if re.search(r"\b(?:whose|that has|which has|who has)\b", owner, re.I):
            return "owner_quote_contains_condition"
        if not _contains_condition_value(probe, quote):
            return "owner_condition_value_not_in_evidence"
    elif kind == "ranking_attribute":
        answer = item.get("answer", "").strip("'\\" ")
        if not answer or not re.search(
                r"(?<!\w)" + re.escape(answer) + r"(?!\w)", quote, re.I):
            return "ranking_attribute_not_in_evidence"
    return "evidence_grounding_unverified"


def _semantic_evidence_status(probe, item, candidate, grounded):
    """Classify independently of a final approve/reject decision."""
    if item["answer"] == "UNCLEAR":
        return "model_uncertain", "model_returned_unclear"
    if grounded == "UNCLEAR":
        return "evidence_invalid", _evidence_failure_reason(probe, item, candidate)
    if grounded != probe["expected"]:
        return "semantic_mismatch", "model_claim_differs_from_public_expected_label"
    return "verified", "model_claim_and_evidence_format_passed"


def validate_candidate_probe_records(probes: list[dict], records: Any, *,
                                     candidate: str | None = None) -> None:
    """Fail closed if the saved v17 probe certificate differs from public plan."""
    if not isinstance(records, list) or len(records) != len(probes):
        raise ValueError("candidate_probe_certificate_incomplete")
    for probe, record in zip(probes, records, strict=True):
        basic_fields = ({"id", "kind", "expected", "answer"} |
                        (_EVIDENCE_FIELDS if probe.get("evidence_required") else set()))
        actual_fields = set(record) if isinstance(record, dict) else set()
        if (not isinstance(record, dict)
                or actual_fields not in (basic_fields, basic_fields | _DIAGNOSTIC_FIELDS)
                or record["id"] != probe["id"]
                or record["kind"] != probe["kind"]
                or record["expected"] != probe["expected"]
                or record["answer"] != probe["expected"]):
            raise ValueError("candidate_probe_certificate_mismatch")
        if probe.get("evidence_required"):
            if (any(not isinstance(record[key], str) for key in _EVIDENCE_FIELDS)
                    or candidate is None
                    or _grounded_answer(probe, record, candidate) != record["answer"]):
                raise ValueError("candidate_probe_certificate_ungrounded")
            if _DIAGNOSTIC_FIELDS.issubset(record):
                raw = {**record, "answer": record["raw_answer"]}
                normalized = _grounded_answer(probe, raw, candidate)
                status, reason = _semantic_evidence_status(
                    probe, raw, candidate, normalized,
                )
                if (record["answer"] != normalized
                        or record["evidence_status"] != status
                        or record["evidence_reason"] != reason):
                    raise ValueError("candidate_probe_certificate_diagnostic_mismatch")


def probe_candidate_semantics(
    backend: Any, reference: str, candidate: str,
    probes: list[dict[str, str]], *, use_schema: bool = True, focused: bool = False,
    claimed_answers: dict[str, str] | None = None,
) -> list[dict[str, str]]:
    """Read candidate only; reference argument is deliberately not transmitted.

    The common review-retry wrapper accepts reference for all review roles.
    Expected answers are held here on the teacher side and never in the
    model request. Return ordered observed answers for audit/certification.
    """
    del reference
    if not callable(getattr(backend, "complete_json", None)):
        raise CandidateProbeFormatError("candidate_probe_backend_unavailable")
    evidence_mode = any(p.get("evidence_required") for p in probes)
    public = [{"id": p["id"], "question": p["question"],
               **({"kind": p["kind"]} if evidence_mode else {}),
               **({"initial_model_answer": claimed_answers[p["id"]]}
                  if claimed_answers and p["id"] in claimed_answers else {})}
              for p in probes]
    system = _EVIDENCE_SYSTEM if evidence_mode else _PROBE_SYSTEM
    schema = _PROBE_SCHEMA
    if evidence_mode:
        system += (f"\nReturn exactly {len(probes)} answer objects, one for EACH "
                   f"of these IDs: {', '.join(p['id'] for p in probes)}. "
                   "All probe kinds require a response, including output/conditional probes. "
                   "Set answer to UNCLEAR when unsure; never return an empty array.")
        system += ("\nRespect each probe's kind: condition_owner answers ONLY A/B/UNCLEAR "
                   "and requires owner; relation_path answers A_TO_B/B_TO_A/UNCLEAR "
                   "and requires subject, object and roles. Other kinds require only "
                   "id, answer and evidence. Do not turn owner probes into relation probes.")
        schema = _evidence_schema(probes)
    if focused:
        system += ("\nThis is the ONE focused rereading of disputed clauses. "
                   "Check local grammatical attachments and passive relation endpoints "
                   "carefully. A new answer requires fresh verbatim evidence. "
                   "Remain UNCLEAR if the text does not resolve the distinction.")
    if claimed_answers:
        # These are the model's OWN previous claims, never teacher expected labels.
        system += (
            "\nEVIDENCE-ONLY REPAIR: the initial_model_answer was your own "
            "previous candidate reading. Supply a short EXACT contiguous "
            "candidate quote and its actual noun phrases for that reading. "
            "Do not invent a quote or paraphrase the candidate. Do not "
            "change the semantic answer just to make a quote fit; answer "
            "UNCLEAR if a faithful quote cannot support your own claim."
        )
    raw = backend.complete_json(
        system,
        json.dumps({"candidate_question": candidate, "probes": public},
                   ensure_ascii=False),
        response_schema=schema if use_schema else None,
    )
    answers = raw.get("answers") if isinstance(raw, dict) else None
    ids = [p["id"] for p in probes]
    got = ([x.get("id") for x in answers if isinstance(x, dict)]
           if isinstance(answers, list) else [])
    if (not isinstance(answers, list) or len(answers) != len(probes)
            or any(not isinstance(identifier, str) for identifier in got)
            or len(set(got)) != len(ids) or set(got) != set(ids)):
        raise CandidateProbeFormatError(
            "candidate_probe_answers_incomplete", expected=ids, received=got,
        )
    by_id = {a["id"]: a for a in answers}
    for item in answers:
        if (not isinstance(item, dict)
                or (evidence_mode and (not {"id", "answer", "evidence"}.issubset(item)
                    or not set(item).issubset({"id", "answer"} | _EVIDENCE_FIELDS)))
                or (not evidence_mode and set(item) != {"id", "answer"})
                or (not evidence_mode and item["answer"] not in _ALLOWED)
                or (evidence_mode and any(not isinstance(v, str) for v in item.values()))):
            raise CandidateProbeFormatError(
                "candidate_probe_answer_invalid", expected=ids, received=got,
            )
    records = []
    for p in probes:
        item = by_id[p["id"]]
        answer = _grounded_answer(p, item, candidate) if evidence_mode else item["answer"]
        status, reason = _semantic_evidence_status(
            p, item, candidate, answer,
        ) if evidence_mode else (
            "verified" if answer == p["expected"] else "semantic_mismatch",
            "model_label_only_without_grounding",
        )
        records.append({
            "id": p["id"], "kind": p["kind"], "expected": p["expected"],
            "answer": answer,
            **({key: item.get(key, "") for key in _EVIDENCE_FIELDS} if evidence_mode else {}),
            **({"raw_answer": item["answer"], "evidence_status": status,
                "evidence_reason": reason} if evidence_mode else {}),
        })
    return records


__all__ = [
    "CandidateProbeFormatError", "compile_candidate_probes",
    "probe_candidate_semantics", "validate_candidate_probe_records",
    "explicit_probe_contradictions",
]
