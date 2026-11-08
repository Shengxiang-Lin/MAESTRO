"""Type-safe work-package DAG, atomic-to-macro coarsening and reference waves.

Only a handoff result is externally addressable. All media/OCR/search/KQA
operations stay inside one macro package and may appear later in summary_trace.
"""
from __future__ import annotations

from collections import defaultdict
from itertools import permutations, product

MAX_CONCURRENCY = 10


class MacroPlanError(ValueError):
    pass


def compile_waves(tasks: list[dict], *, max_concurrency: int = MAX_CONCURRENCY) -> list[list[str]]:
    if max_concurrency != MAX_CONCURRENCY:
        raise MacroPlanError("this experiment fixes maximum concurrency at 10")
    by_id = {t["task_id"]: t for t in tasks}
    if len(by_id) != len(tasks):
        raise MacroPlanError("duplicate macro task IDs")
    completed, waves = set(), []
    while len(completed) < len(tasks):
        ready = sorted(
            tid for tid, item in by_id.items()
            if tid not in completed and set(item["depends_on"]) <= completed
        )
        if not ready:
            raise MacroPlanError("missing dependency or cycle")
        for start in range(0, len(ready), max_concurrency):
            wave = ready[start:start + max_concurrency]
            waves.append(wave)
            completed.update(wave)
    return waves


def build_atomic_draft(packages: list[dict], edges: list[dict]) -> dict:
    """Model fine-grained work as PRIVATE atomic steps before coarsening."""
    steps, handoffs = [], {}
    for package in packages:
        pid = package["task_id"]
        actions = package["internal_actions"]
        if len(actions) < 2:
            raise MacroPlanError("macro work requires at least two internal actions")
        for index, action in enumerate(actions):
            aid = f"{pid}.internal.{index + 1}"
            steps.append({
                "atomic_id": aid, "owner_task_id": pid,
                "action": action,
                "depends_on": ([f"{pid}.internal.{index}"] if index else []),
                "is_handoff": index == len(actions) - 1,
            })
        handoffs[pid] = steps[-1]["atomic_id"]
    step_by_id = {step["atomic_id"]: step for step in steps}
    for edge in edges:
        source, dest = edge["from"], edge["to"]
        start = f"{dest}.internal.1"
        if source not in handoffs or start not in step_by_id:
            raise MacroPlanError("unknown atomic cross-package endpoint")
        step_by_id[start]["depends_on"].append(handoffs[source])
    return {"packages": packages, "steps": steps, "edges": edges}


def coarsen(draft: dict) -> tuple[list[dict], dict]:
    """Collapse all internal steps to their owner work package, preserving
    cross-package boundary dependencies and typed handoffs. Refuse hidden
    cross-package access to intermediate (non-handoff) results.
    """
    packages = {p["task_id"]: p for p in draft["packages"]}
    steps = {s["atomic_id"]: s for s in draft["steps"]}
    owners = defaultdict(list)
    for step in draft["steps"]:
        pid = step["owner_task_id"]
        if pid not in packages or step["atomic_id"] in owners[pid]:
            raise MacroPlanError("invalid micro operation ownership")
        owners[pid].append(step["atomic_id"])
    handoffs = {pid: [sid for sid in ids if steps[sid]["is_handoff"]]
                for pid, ids in owners.items()}
    if any(len(ids) != 1 for ids in handoffs.values()):
        raise MacroPlanError("each work package needs exactly one public handoff")
    coarse_pairs = set()
    for step in steps.values():
        for upstream in step["depends_on"]:
            if upstream not in steps:
                raise MacroPlanError("unknown atomic dependency")
            producer = steps[upstream]
            if producer["owner_task_id"] != step["owner_task_id"]:
                if not producer["is_handoff"]:
                    raise MacroPlanError("cross-package dependency exposes an internal step")
                coarse_pairs.add((producer["owner_task_id"], step["owner_task_id"]))
    expected_pairs = {(e["from"], e["to"]) for e in draft["edges"]}
    if coarse_pairs != expected_pairs:
        raise MacroPlanError("atomic dependencies disagree with handoff contracts")
    coarse = []
    for pid, package in packages.items():
        item = {k: v for k, v in package.items() if k != "internal_actions"}
        item["internal_execution_outline"] = [
            steps[sid]["action"] for sid in owners[pid]
        ]
        item["depends_on"] = sorted({source for source, dest in coarse_pairs if dest == pid})
        coarse.append(item)
    stats = {"atomic_steps": len(steps), "macro_tasks": len(coarse),
             "collapsed_internal_steps": len(steps) - len(coarse),
             "external_handoff_edges_preserved": len(coarse_pairs)}
    return coarse, stats


def compile_contracts(tasks: list[dict], edges: list[dict]) -> list[dict]:
    """Compile input slots from exact producer output fields and types."""
    by_id = {t["task_id"]: t for t in tasks}
    for t in tasks:
        t["inputs"] = {}
    for edge in edges:
        src, dst = edge["from"], edge["to"]
        if src == dst or src not in by_id or dst not in by_id:
            raise MacroPlanError("invalid cross-package endpoint")
        producer = by_id[src]
        output = edge["output_slot"]
        input_slot = edge["input_slot"]
        out_type = producer["outputs"].get(output)
        if not out_type:
            raise MacroPlanError("no such produced output field")
        expected_type = by_id[dst]["input_schema"].get(input_slot)
        if expected_type != out_type:
            raise MacroPlanError("typed input/output mismatch")
        if input_slot in by_id[dst]["inputs"]:
            raise MacroPlanError("duplicate consumer slot")
        by_id[dst]["inputs"][input_slot] = {
            "from_task": src, "output_slot": output,
            "type": out_type, "when": "producer_completed",
        }
    for task in tasks:
        if set(task["inputs"]) != set(task["input_schema"]):
            raise MacroPlanError("unbound required input slots")
        actual = {binding["from_task"] for binding in task["inputs"].values()}
        if actual != set(task["depends_on"]):
            raise MacroPlanError("input bindings disagree with topology")
    compile_waves(tasks)
    return tasks


def complexity(tasks: list[dict], waves: list[list[str]], stats: dict) -> dict:
    depths = {}
    by_id = {t["task_id"]: t for t in tasks}
    for wave in waves:
        for tid in wave:
            deps = by_id[tid]["depends_on"]
            depths[tid] = 1 + max((depths[d] for d in deps), default=0)
    return {
        "macro_task_count": len(tasks),
        "macro_dependency_depth": max(depths.values(), default=0),
        "max_ready_parallelism": max(map(len, waves), default=0),
        "parallel_wave_count": sum(len(wave) > 1 for wave in waves),
        "handoff_count": sum(len(t["inputs"]) for t in tasks),
        "internal_operation_count": stats["atomic_steps"],
        "collapsed_internal_steps": stats["collapsed_internal_steps"],
        "concurrency_limit": MAX_CONCURRENCY,
    }


def canonical_topology(tasks: list[dict], edges: list[dict],
                       waves: list[list[str]]) -> str:
    """Exact unlabeled DAG canonical key for bounded Macro graphs.

    The iterative topological-wave partition is isomorphism-invariant.
    For every permutation within each wave, encode directed adjacency,
    taking its lexicographically minimal representation. With 3-9 Macro
    Tasks and at most four independent source roots this is inexpensive.
    Crucially, Work Types and source identities do not enter the key.
    """
    node_count = len(tasks)
    if node_count > 9:
        raise MacroPlanError("canonical topology supports at most nine tasks")
    ids = {task["task_id"] for task in tasks}
    if set(sum(waves, [])) != ids:
        raise MacroPlanError("topology waves do not cover the task graph")
    edge_set = {(edge["from"], edge["to"]) for edge in edges}
    best = None
    for grouped in product(*(permutations(wave) for wave in waves)):
        order = [node for wave in grouped for node in wave]
        bits = "".join("1" if (src, dst) in edge_set else "0"
                       for src in order for dst in order)
        if best is None or bits < best:
            best = bits
    return f"{node_count}:{best}"
