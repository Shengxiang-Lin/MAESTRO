"""Candidate-only extraction for the offline Stage 2B shadow audit."""
from __future__ import annotations

from .backend import BackendError
from .question_review import _invoke, ReviewFormatError
from .semantic_equivalence import InvalidSemanticGraph, compare_graphs

SCHEMA_VERSION = "maestro.stage2b.shadow_graph.v1"
_KINDS = {"type": "TYPE", "text": "STRING", "number": "NUMBER", "year": "YEAR", "date": "DATE"}
_OPERATIONS = {
    "ALL": "FIND_ALL", "NAMED": "FIND", "ONE": "FIND", "EDGE": "RELATE",
    "AND": "INTERSECT", "OR": "UNION", "COUNT": "COUNT", "NAMES": "QUERY_NAME",
    "ATTRIBUTE": "QUERY_ATTR", "RELATIONS": "QUERY_RELATION",
}
_ARGUMENTS = {
    "ALL": 0, "NAMED": 1, "ONE": 1, "EDGE": 2, "FILTER": 0, "AND": 0, "OR": 0,
    "COUNT": 0, "NAMES": 0, "ATTRIBUTE": 1, "RELATIONS": 0, "MAX": 1, "MIN": 1, "VERIFY": 3,
}

_SYSTEM = """Translate candidate_question into the compact JSON below.
Describe only the request made by the wording. Keep every relation hop and
condition on its actual owner. Give ambiguous wording or unsupported logic
in unresolved. Do not answer the question or add missing facts.

Steps are indexed from zero. inputs lists earlier step indexes. The last step
returns the answer. Each step has op, inputs, args, and conditions.
ALL searches the entities in the task evidence. NAMED uses args [name].
ONE uses args [entity type] and conditions to identify a single entity.
EDGE has one input and args [forward or backward, relation label]. Forward
takes a subject as input and returns objects. Backward takes an object
and returns subjects. Write the label in the subject-to-object direction.
FILTER adds conditions to one input. AND and OR intersect or unite inputs.
COUNT counts distinct matching entities. NAMES returns matching entities.
ATTRIBUTE returns the values of args [attribute]. RELATIONS returns
relations from the first input (subjects) to the second (objects).
MAX and MIN rank the union of their inputs with all ties, args [attribute].
VERIFY takes an ATTRIBUTE input and asks whether any value meets args
[kind, comparison, value]. Use unresolved for other quantifiers.

Conditions restrict entities returned by their own step. Each condition is
{kind:"type", args:[entity type]} or {kind:"text|number|year|date",
args:[attribute, comparison, exact value with unit]}. Use text for postal
codes and other identifiers, year for a year, and number for a quantity.
Keep conditions separate, with their own matching value. Use unresolved if
one value must simultaneously satisfy multiple comparisons. Comparisons
use =, !=, <, >, <= or >=. Use singular types and ordinary English labels.
Leave arrays empty when unused. Put only meaningful ambiguities in unresolved.

Example: How many distinct bicycles under 15 kg are owned by Alex?
{"steps":[
 {"op":"NAMED","inputs":[],"args":["Alex"],"conditions":[]},
 {"op":"EDGE","inputs":[0],"args":["forward","owns"],"conditions":[
   {"kind":"type","args":["bicycle"]},{"kind":"number","args":["mass","<","15 kg"]}]},
 {"op":"COUNT","inputs":[1],"args":[],"conditions":[]}
],"unresolved":[]}
Return one JSON object.
"""


def _object(fields):
    return {
        "type": "object", "properties": fields, "required": list(fields),
        "additionalProperties": False,
    }


def _array(items):
    return {"type": "array", "items": items}


_STRINGS = _array({"type": "string"})
_STEP = _object({
    "op": {"enum": sorted(_ARGUMENTS)}, "inputs": _array({"type": "integer"}),
    "args": _STRINGS,
    "conditions": _array(_object({"kind": {"enum": list(_KINDS)}, "args": _STRINGS})),
})
_SCHEMA = _object({"steps": _array(_STEP), "unresolved": _STRINGS})


def _args(item, count):
    args = item.get("args")
    if not isinstance(args, list) or len(args) != count or any(
        not isinstance(arg, str) or not arg.strip() for arg in args
    ):
        raise InvalidSemanticGraph("invalid_parser_arguments")
    return args


def _condition(item):
    if not isinstance(item, dict) or item.keys() != {"kind", "args"}:
        raise InvalidSemanticGraph("invalid_parser_condition")
    kind = item["kind"]
    if not isinstance(kind, str) or kind not in _KINDS:
        raise InvalidSemanticGraph("invalid_parser_condition_type")
    op = "FILTER_" + _KINDS[kind]
    if kind == "type":
        return {"operation": op, "entity_type": _args(item, 1)[0]}
    attribute, comparison, value = _args(item, 3)
    return {"operation": op, "attribute": attribute, "comparison": comparison,
            "public_value": value}


def _parse_graph(steps):
    if not isinstance(steps, list) or not 0 < len(steps) <= 48:
        raise InvalidSemanticGraph("invalid_parser_steps")
    scopes = []
    for index, step in enumerate(steps):
        if not isinstance(step, dict) or step.keys() != _STEP["properties"].keys():
            raise InvalidSemanticGraph("invalid_parser_step")
        op, inputs, conditions = step["op"], step["inputs"], step["conditions"]
        if (not isinstance(op, str) or op not in _ARGUMENTS
                or not isinstance(conditions, list)
                or not isinstance(inputs, list)
                or any(type(i) is not int or not 0 <= i < index for i in inputs)):
            raise InvalidSemanticGraph("invalid_parser_dependencies_or_operation")
        args = _args(step, _ARGUMENTS[op])
        sid = str(index)
        deps = [str(i) for i in inputs]
        scope = {"id": sid, "operation": _OPERATIONS.get(op), "inputs": deps,
                 "conditions": [_condition(item) for item in conditions]}
        if op == "NAMED":
            scope["public_anchor"] = args[0]
        elif op == "ONE":
            scope["identify_one_by_its_conditions"] = args[0]
        elif op == "EDGE":
            if len(deps) != 1 or args[0] not in {"forward", "backward"}:
                raise InvalidSemanticGraph("invalid_parser_edge")
            subject, obj = (deps[0], sid) if args[0] == "forward" else (sid, deps[0])
            scope.update(subject=subject, object=obj, relation=args[1])
        elif op == "FILTER":
            if not scope["conditions"]:
                raise InvalidSemanticGraph("empty_parser_filter")
            scope["operation"] = scope["conditions"][0]["operation"]
        elif op == "COUNT":
            scope["distinct"] = True
        elif op == "ATTRIBUTE":
            scope["output_attribute"] = args[0]
        elif op == "RELATIONS" and len(deps) == 2:
            scope["subject"], scope["object"] = deps
        elif op in {"MAX", "MIN"}:
            scope.update(
                operation="SELECT_BETWEEN" if len(deps) == 2 else "SELECT_AMONG",
                ranking="largest" if op == "MAX" else "smallest",
                rank_attribute=args[0], include_ties=True,
            )
        elif op == "VERIFY":
            if args[0] not in _KINDS or args[0] == "type":
                raise InvalidSemanticGraph("invalid_parser_verify_type")
            scope.update(operation="VERIFY_" + _KINDS[args[0]], comparison=args[1],
                         public_value=args[2], quantifier="any recorded value")
        scopes.append(scope)
    return {"answer_scope": scopes[-1]["id"], "scopes": scopes}


def verify_candidate(backend, candidate, gold):
    """Compare the independently extracted graph; never send Gold to the model."""
    try:
        parsed = _invoke(backend, _SYSTEM, {"candidate_question": candidate}, schema=_SCHEMA)
    except (BackendError, ReviewFormatError) as exc:
        return {"status": "unresolved", "reason": "parser_protocol", "detail": str(exc)[:200]}
    if parsed.keys() != _SCHEMA["properties"].keys():
        return {"status": "unresolved", "reason": "invalid_parser_fields"}
    unresolved = parsed["unresolved"]
    if not isinstance(unresolved, list) or any(not isinstance(item, str) for item in unresolved):
        return {"status": "unresolved", "reason": "invalid_parser_uncertainty"}
    if unresolved:
        return {"status": "unresolved", "reason": "parser_uncertainty", "parser_output": parsed}
    try:
        graph = _parse_graph(parsed["steps"])
    except InvalidSemanticGraph as exc:
        return {"status": "unresolved", "reason": str(exc), "parser_output": parsed}
    return {**compare_graphs(gold, graph), "parsed_graph": graph}
