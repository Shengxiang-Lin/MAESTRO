"""Symbolic executor for bound synthetic-world programs."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Set

from .ir import ProgramIR, ProgramNode
from .world import SyntheticWorld


class ExecutionError(RuntimeError):
    pass


def _as_set(value: Any) -> Set[Any]:
    if isinstance(value, set):
        return value
    if isinstance(value, (list, tuple)):
        return set(value)
    return {value}


def _parse_control(inputs: Sequence[str]) -> tuple[str | None, str | None]:
    relation = inputs[0] if inputs else None
    direction = None
    for item in inputs[1:]:
        if item in {"forward", "backward"}:
            direction = item
    return relation, direction


def _compare(left: Any, op: str, right: Any) -> bool:
    try:
        l_num = float(left); r_num = float(right); left, right = l_num, r_num
    except (TypeError, ValueError):
        left, right = str(left), str(right)
    if op in {"=", "==", "<CMP:=>", "<CMP:==>"}: return left == right
    if op in {"!=", "<CMP:!=>"}: return left != right
    if op in {">", "greater", "<CMP:>>", "<CMP:greater>"}: return left > right
    if op in {"<", "less", "<CMP:<>", "<CMP:less>"}: return left < right
    if op in {">=", "<CMP:>=>"}: return left >= right
    if op in {"<=", "<CMP:<=>"}: return left <= right
    raise ExecutionError(f"unsupported comparison operator: {op}")


def _select_minimum(mode: str) -> bool:
    return mode.lower() in {"min", "smallest", "less", "<", "<cmp:less>", "<cmp:<>"}


class SymbolicExecutor:
    def execute_trace(self, program: ProgramIR, world: SyntheticWorld) -> Dict[int, Any]:
        values: Dict[int, Any] = {}; nodes = program.node_map(); order = program.topological_order()
        if not program.nodes: raise ExecutionError("empty program")
        for node_id in order:
            node = nodes[node_id]; deps = [values[dep] for dep in node.dependencies]; values[node_id] = self._execute_node(node, deps, world)
        return values

    def execute(self, program: ProgramIR, world: SyntheticWorld) -> Any:
        values = self.execute_trace(program, world); return values[program.topological_order()[-1]]

    def _execute_node(self, node: ProgramNode, deps: List[Any], world: SyntheticWorld) -> Any:
        op = node.op; inputs = list(node.inputs)
        if op == "FIND":
            if not inputs: raise ExecutionError("FIND needs entity:<id> or type:<name>")
            token = inputs[0]
            if token.startswith("entity:"): return {token.split(":", 1)[1]}
            if token.startswith("type:"): return world.entities_of_type(token.split(":", 1)[1])
            return {token}
        if op == "FIND_ALL": return set(world.entities)
        if op == "FILTER_TYPE":
            if len(deps) != 1 or not inputs: raise ExecutionError("FILTER_TYPE needs one dependency and a type")
            entity_type = inputs[0].split(":", 1)[-1]
            return {entity for entity in _as_set(deps[0]) if world.entities.get(str(entity)) == entity_type}
        if op == "RELATE":
            if len(deps) != 1 or not inputs: raise ExecutionError("RELATE needs one dependency and a relation")
            relation, direction = _parse_control(inputs); return world.related(_as_set(deps[0]), str(relation), direction or "forward")
        if op.startswith("FILTER_") and not op.startswith("FILTER_QUALIFIER"):
            if len(deps) != 1 or len(inputs) < 3: raise ExecutionError(f"{op} needs attr, comparator, value")
            attr, comparator, target = inputs[0], inputs[1], inputs[2]; candidates = _as_set(deps[0]); values = world.attribute_values((str(x) for x in candidates), attr)
            return {entity for entity, observed in values.items() if any(_compare(value, comparator, target) for value in observed)}
        if op == "INTERSECT":
            if len(deps) < 2: raise ExecutionError("INTERSECT needs at least two dependencies")
            result = set(_as_set(deps[0]))
            for dep in deps[1:]: result &= _as_set(dep)
            return result
        if op == "UNION":
            result: Set[Any] = set()
            for dep in deps: result |= _as_set(dep)
            return result
        if op == "COUNT":
            if len(deps) != 1: raise ExecutionError("COUNT needs one dependency")
            return len(_as_set(deps[0]))
        if op == "QUERY_NAME":
            if len(deps) != 1: raise ExecutionError("QUERY_NAME needs one dependency")
            return sorted(str(x) for x in _as_set(deps[0]))
        if op == "QUERY_ATTR":
            if len(deps) != 1 or not inputs: raise ExecutionError("QUERY_ATTR needs one dependency and an attribute")
            values = world.attribute_values((str(x) for x in _as_set(deps[0])), inputs[0]); return [value for observed in values.values() for value in observed]
        if op == "QUERY_RELATION":
            if len(deps) != 2: raise ExecutionError("QUERY_RELATION needs two entity-set dependencies")
            left = {str(x) for x in _as_set(deps[0])}; right = {str(x) for x in _as_set(deps[1])}
            return sorted({fact.predicate for fact in world.facts if fact.object_kind == "entity" and fact.subject in left and str(fact.object) in right})
        if op == "SELECT_BETWEEN":
            if len(deps) != 2 or len(inputs) < 2: raise ExecutionError("SELECT_BETWEEN needs two dependencies plus attr and comparator")
            return self._select_by_attribute(_as_set(deps[0]) | _as_set(deps[1]), inputs[0], inputs[1], world)
        if op == "SELECT_AMONG":
            if len(deps) != 1 or len(inputs) < 2: raise ExecutionError("SELECT_AMONG needs one dependency plus attr and comparator")
            return self._select_by_attribute(_as_set(deps[0]), inputs[0], inputs[1], world)
        if op.startswith("VERIFY_"):
            if len(deps) != 1 or len(inputs) < 2: raise ExecutionError(f"{op} needs comparator and expected value")
            observed = list(_as_set(deps[0]))
            if not observed: raise ExecutionError(f"{op} cannot verify a claim without an observed value")
            comparator, target = inputs[0], inputs[1]; return any(_compare(value, comparator, target) for value in observed)
        if op.startswith("FILTER_QUALIFIER") or op in {"QUERY_ATTR_CONDITIONAL", "QUERY_ATTR_QUALIFIER", "QUERY_RELATION_QUALIFIER"}:
            raise ExecutionError(f"{op} requires fact qualifiers, which are intentionally not enabled in the first world schema")
        raise ExecutionError(f"operator not implemented in first executor: {op}")

    def _select_by_attribute(self, candidates: Set[Any], attr: str, mode: str, world: SyntheticWorld) -> Set[str]:
        values = world.attribute_values((str(x) for x in candidates), attr); scored = []
        for entity, observed in values.items():
            for value in observed:
                try: score = float(value)
                except (TypeError, ValueError): continue
                scored.append((score, entity))
        if not scored: return set()
        target_score = min(score for score, _ in scored) if _select_minimum(mode) else max(score for score, _ in scored)
        return {entity for score, entity in scored if score == target_score}
