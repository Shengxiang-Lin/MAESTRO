"""Compile source-scoped Reasoning Program operations into executable Macro Tasks.

The authorized source list of each task is either a singleton for scan or
empty; other tasks only receive typed upstream WorkProducts. No implicit
Teacher-World access is possible during execution.
"""
from __future__ import annotations

from envir.macro.graph import compile_waves
from envir.macro.program_ir import (
    MissingSourceError, ProgramError, Relation, _apply, _scan,
    answer_values, evaluate_program, validate_program,
)

VERSION = "maestro.dependency_macro_graph.v2"


def compile_macro_tasks(program: dict, execution_trace: list[dict],
                        *, compact: bool = False) -> dict:
    if compact:
        return _compile_compact_tasks(program, execution_trace)
    errors = validate_program(program)
    if errors:
        raise ProgramError("invalid_program:" + ",".join(errors))
    trace = {x["node_id"]: x for x in execution_trace}
    tasks = []
    for node in program["nodes"]:
        tid = "work_" + node["id"]
        depends_on = ["work_" + item for item in node["inputs"]]
        source_ids = [node["source_id"]] if node["op"] == "scan" else []
        tasks.append({
            "task_id": tid,
            "program_node_id": node["id"],
            "operation": node["op"],
            "operation_args": {
                key: value for key, value in node.items()
                if key not in {"id", "op", "inputs"}
            },
            "depends_on": depends_on,
            "source_ids": source_ids,
            "input_schema": {
                "input_" + str(i): list(trace[parent]["schema"])
                for i, parent in enumerate(node["inputs"])
            },
            "outputs": {"typed_relation": list(trace[node["id"]]["schema"])},
            "handoff_type": "SourceScopedRelationWithFactRefs",
            "dispatchable": True,
            "execution_policy": "upstream_derived_results_only",
        })
    waves = compile_waves(tasks)
    return {
        "version": VERSION,
        "tasks": tasks,
        "reference_waves": waves,
        "terminal_task_id": "work_" + program["output"],
        "macro_task_count": len(tasks),
        "critical_path_depth": _critical_path(tasks),
        "source_scope_policy": "only_scan_may_access_explicitly_granted_source",
    }


def _compile_compact_tasks(program: dict, execution_trace: list[dict]) -> dict:
    """Fuse ONLY linear unary dependency chains into executable work units.

    Every merged operation still runs and every source scan remains explicitly
    authorized. Branched DAG joins stay as separate typed handoffs.
    """
    errors = validate_program(program)
    if errors:
        raise ProgramError("invalid_program:" + ",".join(errors))
    trace = {row["node_id"]: row for row in execution_trace}
    uses = {}
    for node in program["nodes"]:
        for parent in node["inputs"]:
            uses[parent] = uses.get(parent, 0) + 1
    groups, owner = [], {}
    for node in program["nodes"]:
        parents = node["inputs"]
        merge = len(parents) == 1 and uses.get(parents[0], 0) == 1
        if merge:
            parent = parents[0]
            group_idx = owner[parent]
            group = groups[group_idx]
            merge = (
                group[-1]["id"] == parent
                and len(group) < 3
                and node["op"] != "scan"
            )
        if merge:
            groups[group_idx].append(node)
        else:
            group_idx = len(groups)
            groups.append([node])
        owner[node["id"]] = group_idx

    tasks = []
    for group in groups:
        local = set()
        external = []
        for node in group:
            for parent in node["inputs"]:
                if parent not in local and parent not in external:
                    external.append(parent)
            local.add(node["id"])
        terminal = group[-1]["id"]
        sources = [node["source_id"] for node in group if node["op"] == "scan"]
        # Fusion must never grant a source that an included scan does not use.
        if len(sources) > 1:
            raise ProgramError("compact_group_grants_multiple_sources")
        tasks.append({
            "task_id": "work_" + terminal,
            "program_node_id": terminal,
            "operation": group[0]["op"] if len(group) == 1 else "composite",
            "operation_args": {
                k: v for k, v in group[0].items()
                if k not in {"id", "op", "inputs"}
            } if len(group) == 1 else {},
            "operator_chain": [dict(node) for node in group],
            "external_inputs": external,
            "depends_on": [
                "work_" + groups[owner[parent]][-1]["id"]
                for parent in external
            ],
            "source_ids": sources,
            "input_schema": {
                "input_" + str(i): list(trace[parent]["schema"])
                for i, parent in enumerate(external)
            },
            "outputs": {"typed_relation": list(trace[terminal]["schema"])},
            "handoff_type": "SourceScopedRelationWithFactRefs",
            "dispatchable": True,
            "execution_policy": "typed_input_chain_no_teacher_world",
        })
    waves = compile_waves(tasks)
    return {
        "version": "maestro.dependency_macro_graph.v4",
        "tasks": tasks,
        "reference_waves": waves,
        "terminal_task_id": "work_" + groups[owner[program["output"]]][-1]["id"],
        "macro_task_count": len(tasks),
        "primitive_operator_count": len(program["nodes"]),
        "critical_path_depth": _critical_path(tasks),
        "source_scope_policy": "only_scans_grant_explicit_source",
        "coarsening": "bounded_linear_chain_fusion_max3",
    }


def _critical_path(tasks: list[dict]) -> int:
    depth = {}
    for task in tasks:
        depth[task["task_id"]] = 1 + max(
            (depth[p] for p in task["depends_on"]), default=0
        )
    return max(depth.values(), default=0)


def execute_task(task: dict, *, input_products: list[Relation],
                 visible_sources: dict) -> Relation:
    """One simulated Sub-Agent operation with strict data access controls."""
    if len(input_products) != len(task["depends_on"]):
        raise ProgramError("incorrect_dependency_arity:" + task["task_id"])
    for index, relation in enumerate(input_products):
        expected = task["input_schema"]["input_" + str(index)]
        if list(relation.columns) != expected:
            raise ProgramError("input_schema_mismatch:" + task["task_id"])
    if "operator_chain" in task:
        # The only information available is authorized scan data and
        # previously completed typed handoffs, never the private World.
        if set(visible_sources) != set(task["source_ids"]):
            raise ProgramError("compact_source_scope_mismatch")
        results = dict(zip(task["external_inputs"], input_products))
        for step in task["operator_chain"]:
            if step["op"] == "scan":
                sid = step["source_id"]
                if sid not in visible_sources:
                    raise MissingSourceError("compact_scan_not_granted:" + sid)
                product = _scan(sid, visible_sources[sid])
            else:
                product = _apply(
                    step, [results[parent] for parent in step["inputs"]]
                )
            results[step["id"]] = product
        output = results[task["program_node_id"]]
        if list(output.columns) != task["outputs"]["typed_relation"]:
            raise ProgramError("compact_output_schema_mismatch:" + task["task_id"])
        return output
    args = task["operation_args"]
    if task["operation"] == "scan":
        if task["depends_on"] or len(task["source_ids"]) != 1:
            raise ProgramError("invalid_scan_task")
        sid = task["source_ids"][0]
        if sid not in visible_sources:
            raise MissingSourceError("required_source_not_retrieved:" + sid)
        output = _scan(sid, visible_sources[sid])
    else:
        if task["source_ids"] or visible_sources:
            raise ProgramError("non_scan_task_must_not_read_sources")
        node = {"op": task["operation"], **args}
        output = _apply(node, input_products)
    if list(output.columns) != task["outputs"]["typed_relation"]:
        raise ProgramError("output_schema_mismatch:" + task["task_id"])
    return output


def run_macro_tasks(graph: dict, *, sources: dict,
                    acquired_ids: set[str] | None = None) -> dict:
    """Execute private oracle schedule, emitting handoff-only evidence.

    If acquired_ids is provided, no task can read a source not previously
    acquired by the caller. The default executes with all supplied sources.
    """
    allowed = set(sources) if acquired_ids is None else set(acquired_ids)
    by_id = {t["task_id"]: t for t in graph["tasks"]}
    products, feedback = {}, []
    for wave in graph["reference_waves"]:
        for tid in wave:
            task = by_id[tid]
            scope = task["source_ids"]
            if any(sid not in allowed for sid in scope):
                raise MissingSourceError("source_not_acquired:" + tid)
            inputs = [products[parent] for parent in task["depends_on"]]
            visible = {sid: sources[sid] for sid in scope if sid in sources}
            product = execute_task(
                task, input_products=inputs, visible_sources=visible
            )
            products[tid] = product
            feedback.append({
                "task_id": tid,
                "operation": task["operation"],
                "used_source_ids": scope,
                "used_upstream_handoffs": list(task["depends_on"]),
                "output_schema": list(product.columns),
                "result_rows": len(product.rows),
                "fact_refs": sorted({
                    ref for row in product.rows for ref in row["support"]
                }),
                "workproduct": [{
                    "values": dict(row["values"]),
                    "source_fact_refs": list(row["support"]),
                } for row in product.rows],
            })
    terminal = products[graph["terminal_task_id"]]
    answer_rows = [{
        "values": dict(row["values"]),
        "source_fact_refs": list(row["support"]),
    } for row in terminal.rows]
    from json import dumps
    answer_rows.sort(key=lambda item: dumps(item["values"], sort_keys=True))
    return {
        "answer": [item["values"] for item in answer_rows],
        "answer_with_provenance": answer_rows,
        "task_feedback": feedback,
        "completed_tasks": len(products),
    }


class V2Session:
    """Stateful, costed orchestration simulator with explicit source access.

    Teacher-owned graph is only accessible to the environment. Main-Agent
    receives public_view; only action results reveal source observations or
    downstream WorkProducts. This class is a deterministic integration tool,
    not a learned policy or an untrusted-subagent attack environment.
    """

    COSTS = {"read_source": 2, "run_task": 1, "verify": 2}

    def __init__(self, scene: dict):
        self._scene = scene
        self._sources = scene["world"]["sources"]
        self._tasks = {x["task_id"]: x for x in scene["macro_graph"]["tasks"]}
        self._products = {}
        self._acquired = set()
        self.action_log = []
        self.total_cost = 0

    @property
    def public_view(self) -> dict:
        from copy import deepcopy
        return deepcopy(self._scene["public"])

    @property
    def completed_ids(self) -> list[str]:
        return sorted(self._products)

    def read_source(self, source_id: str) -> dict:
        from copy import deepcopy
        if source_id not in self._sources:
            raise MissingSourceError("unknown_source_id")
        self._acquired.add(source_id)
        self.total_cost += self.COSTS["read_source"]
        self.action_log.append({"action": "read_source", "source_id": source_id})
        return deepcopy(self._sources[source_id])

    def run_task(self, task_id: str) -> dict:
        if task_id not in self._tasks:
            raise ProgramError("unknown_macro_task:" + task_id)
        if task_id in self._products:
            raise ProgramError("duplicate_macro_task:" + task_id)
        task = self._tasks[task_id]
        if any(parent not in self._products for parent in task["depends_on"]):
            raise ProgramError("missing_upstream_workproduct:" + task_id)
        if any(sid not in self._acquired for sid in task["source_ids"]):
            raise MissingSourceError("source_must_be_read_first:" + task_id)
        product = execute_task(
            task, input_products=[
                self._products[parent] for parent in task["depends_on"]
            ],
            visible_sources={
                sid: self._sources[sid] for sid in task["source_ids"]
            },
        )
        self._products[task_id] = product
        self.total_cost += self.COSTS["run_task"]
        self.action_log.append({"action": "run_task", "task_id": task_id})
        return {
            "task_id": task_id,
            "schema": list(product.columns),
            "rows": [{
                "values": dict(row["values"]),
                "source_fact_refs": list(row["support"]),
            } for row in product.rows],
        }

    def verify(self, task_id: str) -> dict:
        """Evidence-closure check only; no second independent observation."""
        if task_id not in self._products:
            raise ProgramError("verify_before_execution:" + task_id)
        available = {
            fact for sid in self._acquired
            for observation in self._sources[sid]["observations"]
            for fact in observation["fact_refs"]
        }
        product = self._products[task_id]
        refs = {ref for row in product.rows for ref in row["support"]}
        self.total_cost += self.COSTS["verify"]
        self.action_log.append({"action": "verify", "task_id": task_id})
        return {
            "valid_evidence_closure": refs <= available,
            "checked_fact_ref_count": len(refs),
            "independent_source_verification": False,
        }

    def final_answer(self) -> list[dict]:
        tid = self._scene["macro_graph"]["terminal_task_id"]
        if tid not in self._products:
            raise ProgramError("terminal_task_not_completed")
        rows = [dict(row["values"]) for row in self._products[tid].rows]
        from json import dumps
        return sorted(rows, key=lambda value: dumps(value, sort_keys=True))


def oracle_schedule(scene: dict) -> dict:
    """Reproduce the reference DAG using legal read/execute actions."""
    session = V2Session(scene)
    for wave in scene["macro_graph"]["reference_waves"]:
        for tid in wave:
            task = next(t for t in scene["macro_graph"]["tasks"]
                        if t["task_id"] == tid)
            for sid in task["source_ids"]:
                if sid not in session._acquired:
                    session.read_source(sid)
            session.run_task(tid)
    return {
        "answer": session.final_answer(),
        "action_log": session.action_log,
        "total_cost": session.total_cost,
        "completed_tasks": len(session.completed_ids),
    }
