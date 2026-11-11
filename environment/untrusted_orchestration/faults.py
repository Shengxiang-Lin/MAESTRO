"""Attempt-local faults at *primitive execution* boundaries.

The frozen World is never mutated. Faults alter a particular Sub-Agent's
observable acquisition/interpretation or WorkProduct before downstream
operators execute. All wrong alternatives must come from *existing* source
observations or already materialized parent relations.
"""
from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
import random

from environment.generate_question.compiler import execute_task
from environment.generate_question.program_ir import (
    Relation, ProgramError, _scan, _lookup_by_key, _apply,
)


class FaultUnsupported(ValueError):
    """The frozen evidence has no plausible realization of this fault."""


@dataclass(frozen=True)
class Fault:
    task_id: str
    mode: str                         # missing | partial | wrong
    seed: int = 0
    persistence: str = "once"        # once | same_path

    def __post_init__(self):
        if self.mode not in {"missing", "partial", "wrong"}:
            raise ValueError("unsupported_fault_mode")
        if self.persistence not in {"once", "same_path"}:
            raise ValueError("unsupported_fault_persistence")


def relation_rows(product: Relation | None) -> list[dict]:
    if product is None:
        return []
    return [{
        "values": deepcopy(row["values"]),
        "source_fact_refs": list(row["support"]),
    } for row in product.rows]


def _primitive(step: dict, products: dict, sources: dict) -> Relation:
    if step["op"] == "scan":
        return _scan(step["source_id"], sources[step["source_id"]])
    if step["op"] == "lookup_by_key":
        return _lookup_by_key(
            step, products[step["inputs"][0]], sources[step["source_id"]],
        )
    return _apply(step, [products[name] for name in step["inputs"]])


def _different(a: Relation, b: Relation) -> bool:
    return (a.columns != b.columns or relation_rows(a) != relation_rows(b))


def _wrong_join(step: dict, products: dict, clean: Relation) -> tuple[Relation, dict] | None:
    left, right = (products[step["inputs"][0]],
                   products[step["inputs"][1]])
    if not left.rows or not right.rows or not clean.rows:
        return None
    right_keys = {
        row["values"][step["right_on"]] for row in right.rows
    }
    # Choose a real but *nonmatching* left candidate. Its mismatch with
    # the right-side record is the evidence-level join interpretation error.
    alternatives = [
        row for row in left.rows
        if row["values"][step["left_on"]] not in right_keys
    ]
    if not alternatives:
        return None
    a, b = alternatives[0], right.rows[0]
    joined = {**b["values"], **a["values"]}
    if set(joined) != set(clean.columns):
        return None
    provenance = sorted(set(a["support"]) | set(b["support"]))
    mutated = Relation(clean.columns, ({
        "values": {name: joined[name] for name in clean.columns},
        "support": tuple(provenance),
    },))
    if not _different(mutated, clean):
        return None
    return mutated, {
        "primitive": step["id"],
        "kind": "misjoined_real_candidates",
        "left_key": a["values"][step["left_on"]],
        "right_key": b["values"][step["right_on"]],
        "source_fact_refs": provenance,
    }


def _wrong_row(
    step: dict, clean: Relation, products: dict, sources: dict,
) -> tuple[Relation, dict] | None:
    if not clean.rows:
        return None
    # A genuine distractor is required; never generate an arbitrary
    # replacement scalar or label. Prefer existing source observations,
    # then preceding upstream rows already visible to this Macro.
    candidates = []
    for source in sources.values():
        for obs in source.get("observations", []):
            values = obs["values"]
            if set(clean.columns) <= set(values):
                candidates.append((
                    {c: values[c] for c in clean.columns},
                    tuple(obs["fact_refs"]),
                ))
    for relation in products.values():
        for row in relation.rows:
            if set(clean.columns) <= set(row["values"]):
                candidates.append((
                    {c: row["values"][c] for c in clean.columns},
                    tuple(row["support"]),
                ))
    original = [row["values"] for row in clean.rows]
    alternate = next((
        (values, refs) for values, refs in candidates
        if values not in original and refs
    ), None)
    if alternate is None:
        return None
    values, refs = alternate
    mutated = Relation(clean.columns, ({
        "values": dict(values), "support": tuple(sorted(set(refs))),
    },))
    return mutated, {
        "primitive": step["id"], "kind": "selected_real_distractor",
        "source_fact_refs": list(refs),
    }


def run_attempt(
    task: dict, *, parents: list[Relation], sources: dict,
    fault: Fault | None = None,
) -> tuple[Relation | None, dict]:
    """Execute actual primitive operations and optionally alter one step.

    clean returns exactly compiler.execute_task's relation. A poisoned
    WorkProduct flows through every *subsequent* operator; the next Macro
    then consumes this resulting Relation as its actual input.
    """
    grants = set(task["source_ids"])
    if set(sources) != grants:
        raise ProgramError("fault_run_source_grant_mismatch")
    if len(parents) != len(task["depends_on"]):
        raise ProgramError("fault_run_dependency_arity_mismatch")
    if fault is None:
        result = execute_task(
            task, input_products=parents, visible_sources=sources,
        )
        return result, {
            "status": "done", "available_fields": list(result.columns),
            "primitive": None, "source_fact_refs": sorted({
                ref for row in result.rows for ref in row["support"]
            }),
        }
    if fault.task_id != task["task_id"]:
        raise ValueError("fault_target_does_not_match_task")
    if "operator_chain" not in task:
        raise FaultUnsupported("fault_requires_primitive_operator_chain")
    # Validate using the unmodified compiler first (also protects against
    # accidentally bypassing frozen source grants and input schemas).
    correct = execute_task(task, input_products=parents, visible_sources=sources)
    prefix = dict(zip(task["external_inputs"], parents))
    clean_steps = []
    for step in task["operator_chain"]:
        relation = _primitive(step, prefix, sources)
        prefix[step["id"]] = relation
        clean_steps.append((step, relation))
    assert not _different(prefix[task["program_node_id"]], correct)

    if fault.mode == "missing":
        first = next((
            step for step, _ in clean_steps
            if step["op"] in {"scan", "lookup_by_key"}
        ), clean_steps[0][0])
        return None, {
            "status": "missing", "primitive": first["id"],
            "available_fields": [], "source_fact_refs": [],
            "failure": "acquisition_or_execution_returned_no_deliverable",
        }

    if fault.mode == "partial":
        if not correct.rows or (len(correct.columns) <= 1
                                and len(correct.rows) <= 1):
            raise FaultUnsupported("no_partial_deliverable_available")
        if len(correct.columns) > 1:
            # Keep schema fixed for typed handoff but set this one field
            # missing; do not fabricate a value.
            field = correct.columns[-1]
            rows = tuple({
                "values": {**row["values"], field: None},
                "support": tuple(row["support"]),
            } for row in correct.rows)
            result = Relation(correct.columns, rows)
            available = [c for c in correct.columns if c != field]
            missing = [field]
        else:
            result = Relation(correct.columns, correct.rows[:-1])
            available = list(correct.columns)
            missing = ["remaining_rows"]
        return result, {
            "status": "partial",
            "primitive": task["program_node_id"],
            # Truncating rows is not the same as omitting a field. Even if
            # the remaining rows contain all fields, counting or selection
            # over an incomplete set is unsafe.
            "rows_complete": len(correct.columns) > 1,
            "missing_fields": missing,
            "available_fields": available,
            "source_fact_refs": sorted({
                ref for row in result.rows for ref in row["support"]
            }),
        }

    # Wrong-result faults are evaluated at the *earliest viable semantic
    # operation* rather than rewriting the final Sub-Agent text.
    ordered = [
        (step, relation) for step, relation in clean_steps
        if step["op"] == "join"
    ] + [
        (step, relation) for step, relation in reversed(clean_steps)
        if step["op"] != "join"
    ]
    chosen = None
    for step, relation in ordered:
        altered = (_wrong_join(step, prefix, relation)
                   if step["op"] == "join" else
                   _wrong_row(step, relation, prefix, sources))
        if altered is not None:
            chosen = step, altered[0], altered[1]
            break
    if chosen is None:
        raise FaultUnsupported("no_real_evidence_distractor")
    target, poisoned, meta = chosen
    products = dict(zip(task["external_inputs"], parents))
    applied = False
    for step in task["operator_chain"]:
        result = _primitive(step, products, sources)
        if step["id"] == target["id"]:
            result = poisoned
            applied = True
        products[step["id"]] = result
    if not applied:
        raise AssertionError("fault_primitive_not_reached")
    final = products[task["program_node_id"]]
    if list(final.columns) != task["outputs"]["typed_relation"]:
        raise FaultUnsupported("fault_changed_frozen_output_schema")
    if not _different(final, correct) or not final.rows:
        raise FaultUnsupported("fault_did_not_propagate_to_workproduct")
    return final, {
        "status": "done",
        "available_fields": list(final.columns),
        "primitive": target["id"],
        "source_fact_refs": sorted({
            ref for row in final.rows for ref in row["support"]
        }),
        **meta,
    }


def probe_eligible_fault_modes(
    scene: dict, modes: list[str],
) -> dict[str, list[str]]:
    """Preflight real fault execution before selecting a branch target.

    Execute the clean frozen DAG once and probe each allowed fault at the
    actual task boundary. Impossible single-field partials, nonexistent
    distractors and faults that cannot propagate are excluded *before*
    assigning a mode, rather than being counted as runtime failures.
    All probes use the frozen World and already computed parent Relations;
    the probes do not mutate the Scene or any published report.
    """
    tasks = {task["task_id"]: task for task in scene["macro_graph"]["tasks"]}
    clean: dict[str, Relation] = {}
    eligible: dict[str, list[str]] = {}
    for wave in scene["macro_graph"]["reference_waves"]:
        for tid in wave:
            task = tasks[tid]
            parents = [clean[p] for p in task["depends_on"]]
            sources = {sid: scene["world"]["sources"][sid]
                       for sid in task["source_ids"]}
            clean[tid], _ = run_attempt(
                task, parents=parents, sources=sources,
            )
            eligible[tid] = []
            for mode in modes:
                try:
                    injected, meta = run_attempt(
                        task, parents=parents, sources=sources,
                        fault=Fault(tid, mode),
                    )
                except (FaultUnsupported, ProgramError, ValueError,
                        TypeError, KeyError, IndexError, ZeroDivisionError):
                    continue
                if mode == "missing" and injected is not None:
                    continue
                if mode == "partial" and meta["status"] != "partial":
                    continue
                if mode == "wrong" and (
                    injected is None
                    or not _different(injected, clean[tid])
                ):
                    continue
                eligible[tid].append(mode)
    return eligible


def choose_targets(wave: list[str], tasks: dict, *,
                   seed: int, multiple: bool = False) -> list[str]:
    """Reproducible, downstream-impact-aware selection in the *last* batch."""
    if not wave:
        raise ValueError("fault_wave_is_empty")
    descendants = {}
    for tid in wave:
        reached, fringe = set(), [tid]
        while fringe:
            node = fringe.pop()
            children = [t["task_id"] for t in tasks.values()
                        if node in t["depends_on"]]
            for child in children:
                if child not in reached:
                    reached.add(child)
                    fringe.append(child)
        descendants[tid] = len(reached)
    rng = random.Random(seed)
    ordered = sorted(wave, key=lambda tid: (-descendants[tid], tid))
    # Prefer high-impact targets but retain variety between branches.
    weights = [1 + 2 * descendants[tid] for tid in ordered]
    chosen = rng.choices(ordered, weights=weights, k=1)[0]
    if multiple and len(wave) > 1:
        other = rng.choice([tid for tid in ordered if tid != chosen])
        return [chosen, other]
    return [chosen]
