"""Source-scoped, executable Reasoning Program IR for an isolated V2 pilot.

Unlike the legacy executor, downstream operations consume typed upstream
results, never a complete Teacher World or rehydrated source observations.
"""
from __future__ import annotations

from dataclasses import dataclass
import json


class ProgramError(ValueError):
    pass


class MissingSourceError(ProgramError):
    pass


@dataclass(frozen=True)
class Relation:
    columns: tuple[str, ...]
    rows: tuple[dict, ...]


_ARITY = {
    "scan": 0, "filter_eq": 1, "filter_gte": 1,
    "project": 1, "unique": 1, "join": 2, "count": 1,
}


def validate_program(program: dict) -> list[str]:
    if not isinstance(program, dict) or not isinstance(program.get("nodes"), list):
        return ["invalid_program_structure"]
    issues, seen = [], {}
    for node in program["nodes"]:
        if not isinstance(node, dict):
            issues.append("invalid_node")
            continue
        nid, op, parents = node.get("id"), node.get("op"), node.get("inputs")
        if not isinstance(nid, str) or not nid or nid in seen:
            issues.append("invalid_or_duplicate_node_id")
            continue
        if op not in _ARITY or not isinstance(parents, list):
            issues.append("unknown_op_or_inputs")
            continue
        if len(parents) != _ARITY[op] or any(
            not isinstance(parent, str) or parent not in seen for parent in parents
        ):
            issues.append("invalid_arity_or_non_topological_dependency")
        if op == "scan" and not isinstance(node.get("source_id"), str):
            issues.append("scan_missing_source_id")
        if op in {"filter_eq", "filter_gte"} and not isinstance(node.get("field"), str):
            issues.append("filter_missing_field")
        if op == "filter_gte" and (
            isinstance(node.get("value"), bool) or
            not isinstance(node.get("value"), (float, int))
        ):
            issues.append("filter_gte_missing_numeric_value")
        if op == "project" and (
            not isinstance(node.get("fields"), list) or
            not node["fields"] or
            any(not isinstance(x, str) for x in node["fields"]) or
            len(set(node["fields"])) != len(node["fields"])
        ):
            issues.append("project_invalid_fields")
        if op == "join" and any(
            not isinstance(node.get(key), str)
            for key in ("left_on", "right_on")
        ):
            issues.append("join_missing_keys")
        seen[nid] = node
    output = program.get("output")
    if output not in seen:
        issues.append("missing_output")
    else:
        reachable = set()
        def visit(nid):
            if nid in reachable:
                return
            reachable.add(nid)
            for parent in seen[nid].get("inputs", []):
                if parent in seen:
                    visit(parent)
        visit(output)
        if reachable != set(seen):
            issues.append("unused_node_not_required_for_output")
    if not any(node.get("op") == "scan" for node in seen.values()):
        issues.append("program_has_no_sources")
    return sorted(set(issues))


def _check(relation: Relation, field: str):
    if field not in relation.columns:
        raise ProgramError("field_not_in_input_schema:" + field)


def _scan(sid: str, source: dict) -> Relation:
    columns, observations = source.get("schema"), source.get("observations")
    if not isinstance(columns, list) or not columns or not isinstance(observations, list):
        raise ProgramError("invalid_source_observation_schema:" + sid)
    if len(set(columns)) != len(columns):
        raise ProgramError("duplicate_source_columns:" + sid)
    rows = []
    for observation in observations:
        values, refs = observation.get("values"), observation.get("fact_refs")
        if not isinstance(values, dict) or set(values) != set(columns):
            raise ProgramError("invalid_source_row:" + sid)
        if not isinstance(refs, list) or not refs or any(
            not isinstance(ref, str) or not ref.startswith(sid + ":")
            for ref in refs
        ):
            raise ProgramError("invalid_source_fact_refs:" + sid)
        rows.append({"values": dict(values), "support": tuple(sorted(set(refs)))})
    return Relation(tuple(columns), tuple(rows))


def _apply(node: dict, inputs: list[Relation]) -> Relation:
    op = node["op"]
    if op in {"filter_eq", "filter_gte"}:
        relation = inputs[0]
        field = node["field"]
        _check(relation, field)
        if op == "filter_eq":
            predicate = lambda x: x == node.get("value")
        else:
            threshold = node["value"]
            def predicate(x):
                if isinstance(x, bool) or not isinstance(x, (int, float)):
                    raise ProgramError("non_numeric_input:" + field)
                return x >= threshold
        return Relation(relation.columns, tuple(
            row for row in relation.rows if predicate(row["values"][field])
        ))
    if op == "join":
        left, right = inputs
        _check(left, node["left_on"])
        _check(right, node["right_on"])
        columns = tuple(dict.fromkeys((*left.columns, *right.columns)))
        rows = []
        for a in left.rows:
            for b in right.rows:
                if a["values"][node["left_on"]] != b["values"][node["right_on"]]:
                    continue
                if any(a["values"][col] != b["values"][col]
                       for col in set(left.columns) & set(right.columns)):
                    continue
                rows.append({
                    "values": {**a["values"], **b["values"]},
                    "support": tuple(sorted(set(a["support"]) | set(b["support"]))),
                })
        return Relation(columns, tuple(rows))
    if op == "project":
        src = inputs[0]
        for field in node["fields"]:
            _check(src, field)
        return Relation(tuple(node["fields"]), tuple({
            "values": {field: row["values"][field] for field in node["fields"]},
            "support": row["support"],
        } for row in src.rows))
    if op == "unique":
        src, grouped = inputs[0], {}
        for row in src.rows:
            key = json.dumps(row["values"], sort_keys=True, ensure_ascii=False)
            if key not in grouped:
                grouped[key] = {"values": dict(row["values"]), "support": set()}
            grouped[key]["support"].update(row["support"])
        return Relation(src.columns, tuple({
            "values": item["values"],
            "support": tuple(sorted(item["support"])),
        } for _, item in sorted(grouped.items())))
    if op == "count":
        src = inputs[0]
        support = sorted({ref for row in src.rows for ref in row["support"]})
        return Relation(("count",), ({
            "values": {"count": len(src.rows)}, "support": tuple(support)
        },))
    raise ProgramError("unsupported_op:" + op)


def evaluate_program(program: dict, *, sources: dict) -> dict:
    problems = validate_program(program)
    if problems:
        raise ProgramError("invalid_program:" + ",".join(problems))
    products, trace = {}, []
    for node in program["nodes"]:
        if node["op"] == "scan":
            sid = node["source_id"]
            if sid not in sources:
                raise MissingSourceError("source_not_available:" + sid)
            product = _scan(sid, sources[sid])
        else:
            product = _apply(node, [products[key] for key in node["inputs"]])
        products[node["id"]] = product
        trace.append({
            "node_id": node["id"],
            "operation": node["op"],
            "input_nodes": list(node["inputs"]),
            "row_count": len(product.rows),
            "schema": list(product.columns),
            "source_fact_refs": sorted({
                ref for row in product.rows for ref in row["support"]
            }),
        })
    result = products[program["output"]]
    answer = [{
        "values": dict(row["values"]),
        "source_fact_refs": list(row["support"]),
    } for row in result.rows]
    answer.sort(key=lambda row: json.dumps(row["values"], sort_keys=True))
    return {"answer": answer, "trace": trace, "products": products}


def answer_values(execution: dict) -> list[dict]:
    return [entry["values"] for entry in execution["answer"]]


def program_complexity(program: dict) -> dict:
    problems = validate_program(program)
    if problems:
        raise ProgramError("invalid_program:" + ",".join(problems))
    depth = {}
    for node in program["nodes"]:
        depth[node["id"]] = (
            0 if node["op"] == "scan"
            else 1 + max(depth[parent] for parent in node["inputs"])
        )
    return {
        "reasoning_depth": depth[program["output"]],
        "operator_count": sum(node["op"] != "scan" for node in program["nodes"]),
        "join_count": sum(node["op"] == "join" for node in program["nodes"]),
        "source_count": len({
            node["source_id"] for node in program["nodes"] if node["op"] == "scan"
        }),
        "operation_sequence": [
            node["op"] for node in program["nodes"] if node["op"] != "scan"
        ],
    }
