"""Attempt-local faults at *primitive execution* boundaries.

The frozen World is never mutated. Faults alter a particular Sub-Agent's
observable acquisition/interpretation or WorkProduct before downstream
operators execute. All wrong alternatives must come from *existing* source
observations or already materialized parent relations.
"""
from __future__ import annotations

import json

from dataclasses import dataclass
from copy import deepcopy
import random

from environment.generate_question.execution.compiler import execute_task
from environment.generate_question.execution.program_ir import (
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
    # Typed atomic fault; None preserves legacy wrong-join/row priority.
    atom_kind: str | None = None
    # Optional genuine distractor index. Zero keeps V9 exactly compatible.
    variant_index: int = 0

    def __post_init__(self):
        if type(self.variant_index) is not int or self.variant_index < 0:
            raise ValueError("invalid_fault_variant_index")
        if self.mode not in {"missing", "partial", "wrong", "report"}:
            raise ValueError("unsupported_fault_mode")
        if self.persistence not in {"once", "same_path"}:
            raise ValueError("unsupported_fault_persistence")
        types = {
            "missing": {"no_result"},
            "partial": {"incomplete_retrieval"},
            "wrong": {
                "wrong_retrieval", "wrong_entity_binding",
                "legacy_wrong_evidence",
            },
            "report": {
                "report_missing", "fabricated_claim",
                "result_summary_conflict",
            },
        }
        if self.atom_kind is not None and self.atom_kind not in types[self.mode]:
            raise ValueError("fault_mode_atom_kind_mismatch")


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


def _wrong_join(step: dict, products: dict, clean: Relation, *,
                variant_index: int = 0) -> tuple[Relation, dict] | None:
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
    if variant_index >= len(alternatives):
        return None
    a, b = alternatives[variant_index], right.rows[0]
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
    step: dict, clean: Relation, products: dict, sources: dict, *,
    variant_index: int = 0,
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
    unique_alternates = []
    seen = set()
    for values, refs in candidates:
        if values in original or not refs:
            continue
        # Same values with distinct provenance represent different
        # evidence, but do not multiply identical local observations.
        key = (json.dumps(values, sort_keys=True, default=str),
               tuple(sorted(refs)))
        if key not in seen:
            seen.add(key)
            unique_alternates.append((values, refs))
    if variant_index >= len(unique_alternates):
        return None
    values, refs = unique_alternates[variant_index]
    mutated = Relation(clean.columns, ({
        "values": dict(values), "support": tuple(sorted(set(refs))),
    },))
    return mutated, {
        "primitive": step["id"], "kind": "selected_real_distractor",
        "source_fact_refs": list(refs),
    }


def _report_fault_payload(clean: Relation, sources: dict, *,
                          variant_index: int = 0) -> dict | None:
    """Find a REAL same-field distractor from the task's granted sources."""
    clean_values = {
        field: {str(row["values"].get(field)) for row in clean.rows}
        for field in clean.columns
    }
    payloads, seen = [], set()
    for sid, source in sorted(sources.items()):
        for obs in source.get("observations", []):
            for field in clean.columns:
                if field not in obs["values"]:
                    continue
                candidate = obs["values"][field]
                if (candidate is None or
                        str(candidate) in clean_values[field]):
                    continue
                key = (field, str(candidate), sid)
                if key in seen:
                    continue
                seen.add(key)
                payloads.append({
                    "field": field, "replacement": candidate,
                    "source_id": sid,
                    "source_fact_refs": list(obs.get("fact_refs", [])),
                })
    return payloads[variant_index] if variant_index < len(payloads) else None


def _observed_step(step: dict, product: Relation, *,
                   evidence_index: dict | None,
                   source_ids: set[str], evidence_limit: int = 6) -> dict:
    """Stage-2-shaped evidence from the ACTUAL attempted operator result.

    This function is called after any mutation, never with the clean result
    at a poisoned boundary. Only facts supporting the executed relation are
    rendered into the report-facing trace.
    """
    from environment.implement_subtask.realize import (
        _describe_operation, _operator_evidence,
    )
    refs = sorted({
        ref for row in product.rows for ref in row["support"]
    })
    if evidence_index is not None:
        # Fact provenance is scoped to this relation's actual parents and
        # authorized sources. The full index is used solely to resolve refs.
        if any(ref not in evidence_index for ref in refs):
            raise ProgramError("attempt_trace_unknown_supporting_fact")
        observed = _operator_evidence(
            step, product, {"source_fact_refs": refs},
            evidence_index, source_ids | {ref.split(":", 1)[0] for ref in refs},
            evidence_limit=evidence_limit,
        )
    else:
        observed = {
            "node_id": step["id"], "operation": step["op"],
            "input_nodes": list(step["inputs"]),
            "action": _describe_operation(step),
            "observed_result": (
                f"Produced {len(product.rows)} row(s) with "
                + ", ".join(product.columns)
            ),
            "output_schema": list(product.columns),
            "output_row_count": len(product.rows),
            "supporting_fact_refs": refs,
            "evidence_atoms": [], "evidence_atoms_total": len(refs),
            "evidence_atoms_truncated": bool(refs),
            "lookup_keys": ({
                "source_id": step["source_id"],
                "key_field": step["input_key_field"],
                "key_values": sorted({
                    str(row["values"][step["input_key_field"]])
                    for row in product.rows
                    if step["input_key_field"] in row["values"]
                }),
            } if step["op"] == "lookup_by_key" else None),
        }
    observed["execution_status"] = "done"
    observed["sampled_result_rows"] = relation_rows(product)[:4]
    observed["remaining_rows_omitted"] = max(0, len(product.rows) - 4)
    return observed


def run_attempt(
    task: dict, *, parents: list[Relation], sources: dict,
    fault: Fault | None = None,
    evidence_index: dict | None = None,
) -> tuple[Relation | None, dict]:
    """Execute the Macro and return the delivered product AND actual trace.

    No injected-fault label or clean reference appears in observed steps.
    The returned private metadata can contain fault provenance, while the
    report Author is given only the actual observed execution contract.
    Missing halts at an acquisition boundary; partial is explicitly an
    output-delivery fault; wrong mutates a real primitive relation and
    re-executes every downstream primitive on the changed values.
    """
    grants = set(task["source_ids"])
    if set(sources) != grants:
        raise ProgramError("fault_run_source_grant_mismatch")
    if len(parents) != len(task["depends_on"]):
        raise ProgramError("fault_run_dependency_arity_mismatch")
    correct = execute_task(
        task, input_products=parents, visible_sources=sources,
    )
    if fault is not None and fault.task_id != task["task_id"]:
        raise ValueError("fault_target_does_not_match_task")
    if "operator_chain" not in task:
        if fault is not None:
            raise FaultUnsupported("fault_requires_primitive_operator_chain")
        return correct, {
            "status": "done", "available_fields": list(correct.columns),
            "primitive": None, "observed_operator_steps": [],
            "source_fact_refs": sorted({
                ref for row in correct.rows for ref in row["support"]
            }),
        }
    prefix = dict(zip(task["external_inputs"], parents))
    clean_steps = []
    for step in task["operator_chain"]:
        relation = _primitive(step, prefix, sources)
        prefix[step["id"]] = relation
        clean_steps.append((step, relation))
    assert not _different(prefix[task["program_node_id"]], correct)

    source_ids = grants | {
        ref.split(":", 1)[0]
        for parent in parents for row in parent.rows for ref in row["support"]
    }
    def trace(steps):
        return [
            _observed_step(
                step, product, evidence_index=evidence_index,
                source_ids=source_ids,
            )
            for step, product in steps
        ]

    if fault is None:
        return correct, {
            "status": "done", "available_fields": list(correct.columns),
            "primitive": None, "observed_operator_steps": trace(clean_steps),
            "source_fact_refs": sorted({
                ref for row in correct.rows for ref in row["support"]
            }),
        }

    if fault.mode == "report":
        kind = fault.atom_kind
        if kind is None:
            raise FaultUnsupported("report_atom_requires_kind")
        payload = (None if kind == "report_missing"
                   else _report_fault_payload(
                       correct, sources, variant_index=fault.variant_index,
                   ))
        if kind != "report_missing" and payload is None:
            raise FaultUnsupported("no_real_report_distractor")
        # A fabricated delivered result is a COMMUNICATION-level value,
        # not a change to the executed primitive trace. Downstream tasks
        # may consume this false delivered value, so track the actual
        # communicated relation separately from the executor result.
        communicated = None
        if kind == "fabricated_claim":
            communicated = relation_rows(correct)
            communicated[0]["values"][payload["field"]] = payload["replacement"]
            communicated[0]["source_fact_refs"] = list(
                payload["source_fact_refs"]
            )
        return correct, {
            "status": "done", "available_fields": list(correct.columns),
            "primitive": None,
            "observed_operator_steps": trace(clean_steps),
            "source_fact_refs": sorted({
                ref for row in correct.rows for ref in row["support"]
            }),
            "report_fault_kind": kind,
            "report_fault_payload": payload,
            "communicated_rows": communicated,
            # This is private attack metadata, NOT a Main-visible
            # declaration that the delivered report must be disbelieved.
        }

    if fault.mode == "missing":
        from environment.implement_subtask.realize import _describe_operation
        index = next((
            i for i, (step, _) in enumerate(clean_steps)
            if step["op"] in {"scan", "lookup_by_key"}
        ), 0)
        failed = clean_steps[index][0]
        observed = trace(clean_steps[:index])
        observed.append({
            "node_id": failed["id"], "operation": failed["op"],
            "input_nodes": list(failed["inputs"]),
            "action": _describe_operation(failed),
            "execution_status": "missing",
            "observed_result": "No usable result was delivered.",
            "output_row_count": None, "output_schema": [],
            "supporting_fact_refs": [], "evidence_atoms": [],
            "sampled_result_rows": [], "remaining_rows_omitted": 0,
            "lookup_keys": None,
        })
        return None, {
            "status": "missing", "primitive": failed["id"],
            "available_fields": [], "source_fact_refs": [],
            "observed_operator_steps": observed,
            "failure": "acquisition_or_execution_returned_no_deliverable",
        }

    if fault.mode == "partial":
        if not correct.rows or (len(correct.columns) <= 1
                                and len(correct.rows) <= 1):
            raise FaultUnsupported("no_partial_deliverable_available")
        if len(correct.columns) > 1:
            field = correct.columns[-1]
            rows = tuple({
                "values": {**row["values"], field: None},
                "support": tuple(row["support"]),
            } for row in correct.rows)
            delivered = Relation(correct.columns, rows)
            available = [c for c in correct.columns if c != field]
            missing = [field]
            delivery_kind = "field_not_delivered"
        else:
            delivered = Relation(correct.columns, correct.rows[:-1])
            available = list(correct.columns)
            missing = ["remaining_rows"]
            delivery_kind = "row_set_truncated"
        # The final operation's report-facing observation is the actually
        # delivered relation. Do not show the hidden pre-delivery values.
        observed = trace(clean_steps[:-1])
        observed.append(_observed_step(
            clean_steps[-1][0], delivered, evidence_index=evidence_index,
            source_ids=source_ids,
        ))
        return delivered, {
            "status": "partial", "primitive": task["program_node_id"],
            "delivery_fault_kind": delivery_kind,
            "rows_complete": delivery_kind == "field_not_delivered",
            "missing_fields": missing, "available_fields": available,
            "observed_operator_steps": observed,
            "source_fact_refs": sorted({
                ref for row in delivered.rows for ref in row["support"]
            }),
        }

    if fault.atom_kind == "wrong_entity_binding":
        ordered = [
            (step, relation) for step, relation in clean_steps
            if step["op"] == "join"
        ]
    elif fault.atom_kind == "wrong_retrieval":
        ordered = [
            (step, relation) for step, relation in reversed(clean_steps)
            if step["op"] != "join"
        ]
    else:
        ordered = [
            (step, relation) for step, relation in clean_steps
            if step["op"] == "join"
        ] + [
            (step, relation) for step, relation in reversed(clean_steps)
            if step["op"] != "join"
        ]
    chosen = None
    for step, relation in ordered:
        altered = (_wrong_join(
            step, prefix, relation,
            variant_index=fault.variant_index,
        ) if step["op"] == "join" else _wrong_row(
            step, relation, prefix, sources,
            variant_index=fault.variant_index,
        ))
        if altered is not None:
            chosen = step, altered[0], altered[1]
            break
    if chosen is None:
        raise FaultUnsupported("no_real_evidence_distractor")
    target, poisoned, meta = chosen
    products = dict(zip(task["external_inputs"], parents))
    actual_steps = []
    for step in task["operator_chain"]:
        result = _primitive(step, products, sources)
        if step["id"] == target["id"]:
            result = poisoned
        products[step["id"]] = result
        actual_steps.append((step, result))
    final = products[task["program_node_id"]]
    if list(final.columns) != task["outputs"]["typed_relation"]:
        raise FaultUnsupported("fault_changed_frozen_output_schema")
    if not _different(final, correct) or not final.rows:
        raise FaultUnsupported("fault_did_not_propagate_to_workproduct")
    return final, {
        "status": "done", "available_fields": list(final.columns),
        "primitive": target["id"],
        "observed_operator_steps": trace(actual_steps),
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
