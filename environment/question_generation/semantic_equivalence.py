"""Conservative structural comparison of public, typed question graphs."""
from __future__ import annotations

from datetime import datetime

from .contracts import normalize_value
from ..task_construction.rendering.naming import comparison_phrase

FILTERS = frozenset({
    "FILTER_TYPE", "FILTER_STRING", "FILTER_NUMBER", "FILTER_YEAR", "FILTER_DATE",
})
VERIFY = frozenset("VERIFY_" + suffix for suffix in ("STRING", "NUMBER", "YEAR", "DATE"))
RANK = frozenset({"SELECT_AMONG", "SELECT_BETWEEN"})
SETS = FILTERS | RANK | {"FIND", "FIND_ALL", "RELATE", "INTERSECT", "UNION"}
OPERATIONS = SETS | VERIFY | {"COUNT", "QUERY_NAME", "QUERY_ATTR", "QUERY_RELATION"}

_COMPARISONS = {
    "equal to", "different from", "less than", "greater than", "at least", "at most",
}
_FIELDS = {
    "FIND": {"public_anchor", "candidate_type", "identify_one_by_its_conditions"},
    "FIND_ALL": {"candidate_range"},
    "RELATE": {"relation", "subject", "object"},
    "COUNT": {"distinct"},
    "QUERY_ATTR": {"output_attribute"},
    "QUERY_RELATION": {"subject", "object"},
    **{op: {"ranking", "rank_attribute", "include_ties", "candidate_pool"} for op in RANK},
    **{op: {"comparison", "public_value", "quantifier"} for op in VERIFY},
}
_ARITIES = {
    "FIND": 0, "FIND_ALL": 0, "QUERY_RELATION": 2, "SELECT_BETWEEN": 2,
    "INTERSECT": None, "UNION": None,
}


class InvalidSemanticGraph(ValueError):
    """The graph is incomplete, inconsistent, or outside the supported grammar."""


def _text(item, field):
    value = item.get(field)
    if not isinstance(value, str) or not value.strip():
        raise InvalidSemanticGraph(f"missing_{field}")
    return value.strip()


def _label(item, field):
    return " ".join(_text(item, field).casefold().replace("-", " ").split())


def _comparison(item):
    raw = _text(item, "comparison").casefold()
    value = {"before": "less than", "after": "greater than"}.get(raw, comparison_phrase(raw))
    if value not in _COMPARISONS:
        raise InvalidSemanticGraph("unsupported_comparison")
    return value


def _value(item, op):
    raw = _text(item, "public_value")
    value = normalize_value(raw, op)
    if op.endswith("_DATE") and value["type"] != "date":
        try:
            date = datetime.strptime(raw.title(), "%B %d, %Y").date()
        except ValueError as exc:
            raise InvalidSemanticGraph("unsupported_date_literal") from exc
        return "date", date.isoformat(), ""
    if op.endswith(("_NUMBER", "_YEAR")) and value["type"] == "string":
        raise InvalidSemanticGraph("unsupported_numeric_literal")
    return value["type"], value["value"], value.get("unit", "").strip()


def _conditions(scope):
    conditions = scope.get("conditions", [])
    if not isinstance(conditions, list):
        raise InvalidSemanticGraph("invalid_conditions")
    atoms = []
    for condition in conditions:
        if not isinstance(condition, dict):
            raise InvalidSemanticGraph("invalid_condition")
        op = _text(condition, "operation")
        if op not in FILTERS:
            raise InvalidSemanticGraph("unsupported_filter")
        fields = {"entity_type"} if op == "FILTER_TYPE" else {
            "attribute", "comparison", "public_value",
        }
        if condition.keys() - fields - {"id", "operation"}:
            raise InvalidSemanticGraph("unsupported_condition_fields")
        if op == "FILTER_TYPE":
            atoms.append((op, _label(condition, "entity_type")))
        else:
            atoms.append((
                op, _label(condition, "attribute"), _comparison(condition), _value(condition, op),
            ))
    return atoms


def _ordered_unique(values):
    return tuple(sorted(set(values), key=repr))


def _filter(base, atoms):
    # Each atom tests for any matching recorded value independently.
    if not atoms:
        return base
    if base[0] == "filter":
        _, base, prior = base
        atoms = (*prior, *atoms)
    return ("filter", base, _ordered_unique(atoms))


def _join(op, children):
    terms = _ordered_unique(
        term for child in children for term in (child[1] if child[0] == op else (child,))
    )
    if len(terms) == 1:
        return terms[0]
    if op == "INTERSECT":
        parts = [(term[1], term[2]) if term[0] == "filter" else (term, ()) for term in terms]
        restricted = {base for base, _ in parts} - {("universe",)}
        # A relation endpoint from the task evidence is in the universe.
        # An intersection with filters over that universe constrains it directly.
        if len(restricted) == 1 and next(iter(restricted))[0] == "RELATE":
            return _filter(next(iter(restricted)), [atom for _, atoms in parts for atom in atoms])
        if all(base == parts[0][0] for base, _ in parts):
            return _filter(parts[0][0], [atom for _, atoms in parts for atom in atoms])
    return (op, terms)


def _expression(scope, op, deps):
    if op == "FIND_ALL":
        if scope.get("candidate_range", "bounded task evidence") != "bounded task evidence":
            raise InvalidSemanticGraph("unsupported_candidate_range")
        return ("universe",)
    if op == "FIND":
        anchors = set(scope) & _FIELDS["FIND"]
        if len(anchors) != 1:
            raise InvalidSemanticGraph("ambiguous_find_anchor")
        anchor = next(iter(anchors))
        if anchor == "public_anchor":
            return ("named", _text(scope, anchor))
        return _filter(("universe",), [("FILTER_TYPE", _label(scope, anchor))])
    if op == "RELATE":
        endpoints = scope.get("subject"), scope.get("object")
        if endpoints == (scope["inputs"][0], scope["id"]):
            direction = "forward"
        elif endpoints == (scope["id"], scope["inputs"][0]):
            direction = "backward"
        else:
            raise InvalidSemanticGraph("unbound_relation_endpoint")
        return (op, _label(scope, "relation"), direction, deps[0])
    if op in FILTERS:
        return deps[0]
    if op in {"UNION", "INTERSECT"}:
        return _join(op, deps)
    if op in RANK:
        rank = _text(scope, "ranking")
        if rank not in {"largest", "smallest"} or scope.get("include_ties") is not True:
            raise InvalidSemanticGraph("unresolved_ranking")
        if op == "SELECT_BETWEEN" and scope.get("candidate_pool", "union of inputs") != "union of inputs":
            raise InvalidSemanticGraph("unsupported_ranking_pool")
        pool = _join("UNION", deps) if op == "SELECT_BETWEEN" else deps[0]
        return ("rank", pool, rank, _label(scope, "rank_attribute"))
    if op == "COUNT" and scope.get("distinct") is not True:
        raise InvalidSemanticGraph("unresolved_distinct_count")
    if op == "QUERY_ATTR":
        return (op, deps[0], _label(scope, "output_attribute"))
    if op == "QUERY_RELATION" and [scope.get("subject"), scope.get("object")] != scope["inputs"]:
        raise InvalidSemanticGraph("unbound_query_relation_endpoint")
    if op in VERIFY:
        if (deps[0][0] != "QUERY_ATTR"
                or scope.get("quantifier") != "any recorded value"):
            raise InvalidSemanticGraph("unresolved_verification_quantifier")
        return (op, deps[0], _comparison(scope), _value(scope, op))
    return (op, *deps)


def canonical_expression(graph):
    """Resolve variable names and normalize supported set expressions."""
    if not isinstance(graph, dict) or not isinstance(graph.get("scopes"), list):
        raise InvalidSemanticGraph("invalid_scope_graph")
    by_id = {}
    for scope in graph["scopes"]:
        if not isinstance(scope, dict):
            raise InvalidSemanticGraph("invalid_scope")
        sid = _text(scope, "id")
        if sid in by_id:
            raise InvalidSemanticGraph("duplicate_scope_id")
        by_id[sid] = scope

    active, visited = set(), set()
    budget = 256

    def visit(sid):
        nonlocal budget
        budget -= 1
        if budget < 0 or len(active) >= 48:
            raise InvalidSemanticGraph("scope_graph_too_complex")
        if sid not in by_id or sid in active:
            raise InvalidSemanticGraph("unbound_or_cyclic_scope")
        scope = by_id[sid]
        op = _text(scope, "operation")
        if op not in OPERATIONS:
            raise InvalidSemanticGraph("unsupported_operation")
        if scope.keys() - _FIELDS.get(op, set()) - {"id", "operation", "inputs", "conditions"}:
            raise InvalidSemanticGraph("unsupported_scope_fields")
        inputs = scope.get("inputs")
        if not isinstance(inputs, list) or any(not isinstance(dep, str) for dep in inputs):
            raise InvalidSemanticGraph("invalid_scope_inputs")
        arity = _ARITIES.get(op, 1)
        if (arity is None and len(inputs) < 2) or (arity is not None and len(inputs) != arity):
            raise InvalidSemanticGraph("invalid_scope_arity")
        active.add(sid)
        deps = [visit(dep) for dep in inputs]
        active.remove(sid)
        visited.add(sid)
        input_ops = [by_id[dep]["operation"] for dep in inputs]
        if op in VERIFY:
            valid_inputs = input_ops == ["QUERY_ATTR"]
        else:
            valid_inputs = all(dep_op in SETS for dep_op in input_ops)
        if not valid_inputs:
            raise InvalidSemanticGraph("invalid_scope_input_type")
        atoms = _conditions(scope)
        if (atoms and op not in SETS) or (op in FILTERS and not atoms):
            raise InvalidSemanticGraph("invalid_scope_conditions")
        result = _filter(_expression(scope, op, deps), atoms)
        if "identify_one_by_its_conditions" in scope:
            if not atoms:
                raise InvalidSemanticGraph("missing_identity_clue")
            return ("identify_one", result)
        return result

    output = _text(graph, "answer_scope")
    result = visit(output)
    if visited != by_id.keys():
        raise InvalidSemanticGraph("unused_scope")
    if by_id[output]["operation"] in SETS - RANK:
        raise InvalidSemanticGraph("missing_answer_operation")
    return result


def compare_graphs(gold, candidate):
    """Report a canonical match or a discrepancy for independent inspection."""
    expected = canonical_expression(gold)
    try:
        observed = canonical_expression(candidate)
    except (InvalidSemanticGraph, ArithmeticError, ValueError) as exc:
        return {"status": "unresolved", "reason": str(exc)}
    if expected == observed:
        return {"status": "match"}
    return {"status": "mismatch", "expected_expression": expected, "observed_expression": observed}
