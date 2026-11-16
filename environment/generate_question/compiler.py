"""Compile source-scoped Reasoning Program operations into executable Macro Tasks.

The authorized source list of each task is either a singleton for scan or
empty; other tasks only receive typed upstream WorkProducts. No implicit
Teacher-World access is possible during execution.
"""
from __future__ import annotations

from .dag import compile_waves
from .evidence_tools import tools_from_primitive_steps
from .program_ir import (
    MissingSourceError, ProgramError, Relation, _apply, _scan, _lookup_by_key,
    answer_values, evaluate_program, validate_program,
)

def compile_macro_tasks(program: dict, execution_trace: list[dict]) -> dict:
    """Compile the sole supported executable, bounded Macro TaskGraph."""
    return _compile_compact_tasks(program, execution_trace)


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
                and node["op"] not in {"scan", "lookup_by_key"}
                and group[-1]["op"] != "lookup_by_key"
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
        sources = [node["source_id"] for node in group
                   if node["op"] in {"scan", "lookup_by_key"}]
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
        "source_scope_policy": (
            "scan_or_keyed_lookup_only_explicit_source"
            if any(node["op"] == "lookup_by_key" for node in program["nodes"])
            else "only_scans_grant_explicit_source"
        ),
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
        grants = {
            step["source_id"] for step in task["operator_chain"]
            if step["op"] in {"scan", "lookup_by_key"}
        }
        if (set(task["source_ids"]) != grants
                or set(visible_sources) != grants):
            raise ProgramError("compact_source_scope_mismatch")
        results = dict(zip(task["external_inputs"], input_products))
        for step in task["operator_chain"]:
            if step["op"] in {"scan", "lookup_by_key"}:
                sid = step["source_id"]
                if sid not in visible_sources:
                    raise MissingSourceError("compact_source_not_granted:" + sid)
                product = (
                    _scan(sid, visible_sources[sid])
                    if step["op"] == "scan" else
                    _lookup_by_key(
                        step, results[step["inputs"][0]],
                        visible_sources[sid],
                    )
                )
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


class QuestionSession:
    """Stateful, costed orchestration simulator with explicit source access.

    Teacher-owned graph is only accessible to the environment. Main-Agent
    receives public_view; only action results reveal source observations or
    downstream WorkProducts. This class is a deterministic integration tool,
    not a learned policy or an untrusted-subagent attack environment.
    """

    COSTS = {"read_source": 2, "lookup_by_key": 2, "run_task": 1, "verify": 2}

    def __init__(self, scene: dict):
        self._scene = scene
        self._sources = scene["world"]["sources"]
        self._tasks = {x["task_id"]: x for x in scene["macro_graph"]["tasks"]}
        self._products = {}
        self._acquired = set()
        self._keyed_fact_refs = set()
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
        if self._sources[source_id].get("access_policy", {}).get(
            "mode", "browse"
        ) != "browse":
            raise MissingSourceError("full_source_read_forbidden:" + source_id)
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
        for sid in task["source_ids"]:
            mode = self._sources[sid].get(
                "access_policy", {}
            ).get("mode", "browse")
            if mode == "browse" and sid not in self._acquired:
                raise MissingSourceError("source_must_be_read_first:" + task_id)
            if mode not in {"browse", "keyed_lookup"}:
                raise MissingSourceError("unknown_source_access_mode:" + sid)
        input_products = [
            self._products[parent] for parent in task["depends_on"]
        ]
        product = execute_task(
            task, input_products=input_products,
            visible_sources={
                sid: self._sources[sid] for sid in task["source_ids"]
            },
        )
        self._products[task_id] = product
        # Access is keyed by an actual upstream product and granted only
        # for the requested records. Full keyed stores never enter memory.
        for step in task.get("operator_chain", []):
            if step["op"] != "lookup_by_key":
                continue
            sid = step["source_id"]
            source_keys = [row["values"][step["input_key_field"]]
                           for relation in input_products
                           for row in relation.rows
                           if step["input_key_field"] in relation.columns]
            self.action_log.append({
                "action": "lookup_by_key", "source_id": sid,
                "keys": source_keys,
            })
            self.total_cost += self.COSTS["lookup_by_key"]
            self._keyed_fact_refs.update({
                ref for row in product.rows for ref in row["support"]
                if ref.startswith(sid + ":")
            })
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
        available.update(self._keyed_fact_refs)
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
    session = QuestionSession(scene)
    for wave in scene["macro_graph"]["reference_waves"]:
        for tid in wave:
            task = next(t for t in scene["macro_graph"]["tasks"]
                        if t["task_id"] == tid)
            for sid in task["source_ids"]:
                policy = session._sources[sid].get(
                    "access_policy", {}
                ).get("mode", "browse")
                if policy == "browse" and sid not in session._acquired:
                    session.read_source(sid)
            session.run_task(tid)
    return {
        "answer": session.final_answer(),
        "action_log": session.action_log,
        "total_cost": session.total_cost,
        "completed_tasks": len(session.completed_ids),
    }


def compile_frozen_macro_plan(spec: dict, program: dict,
                              execution_trace: list[dict]) -> dict:
    """Compile pre-sampled Macro contracts WITHOUT re-partitioning operations.

    This function preserves every task ID and edge from macro_plan.
    Primitive nodes may only read an explicitly granted local source or
    a declared parent Macro's terminal WorkProduct. Any hidden cross-Macro
    primitive dependency fails closed.
    """
    if spec.get("reasoning_pattern") == "task_plan":
        from .task_plan import validate_spec as validate_macro_first_spec
        from .task_plan_runtime import compile_macro_solvers
    elif spec.get("reasoning_pattern") == "task_graph_first_v2":
        from .task_graph_first_v2 import (
            validate_spec as validate_macro_first_spec,
            compile_macro_solvers,
        )
    elif spec.get("reasoning_pattern") == "task_graph_first_v1":
        from .task_graph_first import (
            validate_spec as validate_macro_first_spec,
            compile_macro_solvers,
        )
    elif spec.get("reasoning_pattern") == "typed_knowledge_v4":
        from .hybrid_world_v4 import (
            validate_spec as validate_macro_first_spec,
            compile_macro_solvers,
        )
    elif spec.get("reasoning_pattern") == "typed_knowledge_v3":
        from .hybrid_world import (
            validate_spec as validate_macro_first_spec,
            compile_macro_solvers,
        )
    elif spec.get("reasoning_pattern") == "typed_knowledge_v2":
        from .knowledge_macro_v2 import (
            validate_spec as validate_macro_first_spec,
            compile_macro_solvers,
        )
    elif spec.get("reasoning_pattern") == "typed_knowledge_v1":
        from .knowledge_macro import (
            validate_spec as validate_macro_first_spec,
            compile_macro_solvers,
        )
    else:
        from .macro_planning import (
            validate_spec as validate_macro_first_spec,
            compile_macro_solvers,
        )
    errors = validate_macro_first_spec(spec)
    if errors:
        raise ProgramError("invalid_frozen_macro_plan:" + ",".join(errors))
    if program != compile_macro_solvers(spec):
        raise ProgramError("macro_solver_not_frozen")
    problems = validate_program(program)
    if problems:
        raise ProgramError("invalid_local_macro_solvers:" + ",".join(problems))
    plan = spec["macro_plan"]
    groups = program["macro_solver_groups"]
    source_ids = {role["source_id"] for role in spec["sources"]}
    by_node = {node["id"]: node for node in program["nodes"]}
    trace = {entry["node_id"]: entry for entry in execution_trace}
    if len(groups) != len(plan["macro_tasks"]):
        raise ProgramError("macro_solver_group_count_mismatch")

    tasks, owned = [], set()
    for macro, group in zip(plan["macro_tasks"], groups):
        mid = macro["macro_id"]
        if group["macro_id"] != mid:
            raise ProgramError("macro_group_order_mismatch")
        ids = group["node_ids"]
        if not ids or any(nid in owned or nid not in by_node for nid in ids):
            raise ProgramError("macro_solver_node_partition_invalid")
        if group["output_node_id"] != ids[-1]:
            raise ProgramError("macro_solver_terminal_invalid")
        parent_tasks = macro["depends_on"]
        available = {
            previous["task_id"].removeprefix("work_"): previous
            for previous in tasks
        }
        if any(pid not in available for pid in parent_tasks):
            raise ProgramError("macro_dependency_not_topological")
        prior_output_ids = {
            previous["task_id"].removeprefix("work_"): previous["program_node_id"]
            for previous in tasks
        }
        external = [prior_output_ids[parent] for parent in parent_tasks]
        if group["external_input_nodes"] != external:
            raise ProgramError("macro_external_input_mismatch")
        local = set()
        for nid in ids:
            node = by_node[nid]
            for parent in node["inputs"]:
                if parent not in local and parent not in external:
                    raise ProgramError("undeclared_cross_macro_primitive_dependency")
            local.add(nid)
        accessed = [
            by_node[nid]["source_id"] for nid in ids
            if by_node[nid]["op"] in {"scan", "lookup_by_key"}
        ]
        if (accessed != macro["source_scope"]
                or any(sid not in source_ids for sid in accessed)):
            raise ProgramError("macro_local_source_scope_mismatch")
        # A typed relation schema is a map, not an ordered field list.
        # Frozen Spec JSON can normalize dict key order (for example:
        # {"metric_a": "int", "record_key_2": "str"}), while the program
        # projects ["record_key_2", "metric_a"]. Comparing list(dict)
        # silently rejected valid computation Macros and drove quota refill.
        if set(trace[ids[-1]]["schema"]) != set(macro["output_schema"]):
            raise ProgramError("macro_output_type_contract_mismatch:" + mid)
        for parent, nid in zip(parent_tasks, external):
            upstream = available[parent]["outputs"]["typed_relation"]
            if list(trace[nid]["schema"]) != upstream:
                raise ProgramError("macro_input_type_contract_mismatch")
            # Frozen typed relation contracts are mappings; JSON object
            # ordering is not meaningful. Verify *both* the field names and
            # their declared types against the frozen parent output schema,
            # not only a list of ordered keys.
            declared = macro["input_contract"][parent]
            frozen_parent = next(
                prior["output_schema"] for prior in plan["macro_tasks"]
                if prior["macro_id"] == parent
            )
            if (not isinstance(declared, dict)
                    or declared != frozen_parent
                    or set(declared) != set(upstream)):
                raise ProgramError("macro_declared_input_type_mismatch")
        chain = [dict(by_node[nid]) for nid in ids]
        tool_fields = (
            {"required_tools": tools_from_primitive_steps(chain)}
            if spec.get("evidence_tool_contract_version") else {}
        )
        tasks.append({
            "task_id": "work_" + mid,
            "program_node_id": ids[-1],
            "operation": "composite" if len(ids) > 1 else chain[0]["op"],
            "operation_args": {},
            "operator_chain": chain,
            "external_inputs": external,
            "depends_on": ["work_" + p for p in parent_tasks],
            "source_ids": accessed,
            "input_schema": {
                "input_" + str(i): list(trace[nid]["schema"])
                for i, nid in enumerate(external)
            },
            "outputs": {"typed_relation": list(trace[ids[-1]]["schema"])},
            "handoff_type": "SourceScopedRelationWithFactRefs",
            "dispatchable": True,
            "execution_policy": "frozen_macro_contract_local_gold_solver",
            "macro_role": macro["role"],
            **tool_fields,
        })
        owned.update(local)
    if owned != set(by_node):
        raise ProgramError("orphan_primitive_not_owned_by_macro")
    if (tasks[-1]["task_id"] != "work_" + plan["terminal_macro_id"]
            or tasks[-1]["program_node_id"] != program["output"]):
        raise ProgramError("macro_final_output_mismatch")
    waves = compile_waves(tasks)
    return {
        "version": "maestro.dependency_macro_graph.v4",
        "tasks": tasks,
        "reference_waves": waves,
        "terminal_task_id": tasks[-1]["task_id"],
        "macro_task_count": len(tasks),
        "primitive_operator_count": len(program["nodes"]),
        "critical_path_depth": _critical_path(tasks),
        "source_scope_policy": "frozen_macro_local_source_permissions",
        "coarsening": "none_predefined_macro_boundaries",
        "plan_origin": "macro_first",
    }
