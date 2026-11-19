"""Zero-LLM enumeration of genuine Stage-4 atomic fault candidates.

The original Macro DAG is an executable *baseline*, not a verifier graph.
For every eligible trusted feedback round, try each actual returned
Sub-Agent task, supported atomic mechanism, and (optionally) several
different genuine distractors. Each candidate must pass a real executor
preflight on the frozen Scene. No random text or new World facts.

This is deliberately limited to ONE initial fault per candidate; a
multi-fault search is not required to obtain complex recovery trajectories.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from hashlib import sha256
import json
import random

from environment.generate_question.execution.program_ir import ProgramError
from environment.untrusted_orchestration.runtime.atomic_faults import AtomicFault, ATOM_TO_MODE
from environment.untrusted_orchestration.runtime.faults import FaultUnsupported, relation_rows, run_attempt
from .scenario_generator import make_scenario
from .handoff_conflicts import discover_handoffs, fault_breaks_handoff
from .same_source_recheck import discover_same_source_rechecks


_VARIANT_KINDS = {
    "wrong_retrieval", "wrong_entity_binding", "computation_error",
    "fabricated_claim", "result_summary_conflict",
    "legacy_wrong_evidence",
}


def _clean_products(scene: dict) -> dict:
    tasks = {row["task_id"]: row for row in scene["macro_graph"]["tasks"]}
    products = {}
    for wave in scene["macro_graph"]["reference_waves"]:
        for tid in wave:
            task = tasks[tid]
            products[tid], _ = run_attempt(
                task, parents=[products[pid] for pid in task["depends_on"]],
                sources={
                    sid: scene["world"]["sources"][sid]
                    for sid in task["source_ids"]
                },
            )
    return products


def _candidate_is_real(scene: dict, task: dict, kind: str, variant: int,
                       products: dict) -> bool:
    try:
        atom = AtomicFault(
            kind, task["task_id"], origin_round=1,
            variant_index=variant,
        )
        mutated, runtime = run_attempt(
            task, parents=[products[pid] for pid in task["depends_on"]],
            sources={
                sid: scene["world"]["sources"][sid]
                for sid in task["source_ids"]
            },
            fault=atom.as_legacy_fault(),
        )
    except (FaultUnsupported, ProgramError, ValueError, TypeError,
            KeyError, IndexError, ZeroDivisionError):
        return False
    if kind in {"no_result", "incomplete_retrieval",
                "done_but_incomplete"}:
        return (
            mutated is None if kind == "no_result"
            else runtime.get("status") == "partial"
            and (kind != "done_but_incomplete"
                 or runtime.get("reported_status") == "done")
        )
    if kind in {"wrong_retrieval", "wrong_entity_binding",
                "computation_error", "legacy_wrong_evidence"}:
        return (
            mutated is not None
            and relation_rows(mutated) != relation_rows(
                products[task["task_id"]]
            )
        )
    return runtime.get("report_fault_kind") == kind


def _stable_candidate_order(
    candidates: list[tuple[str, str, int]], seed: int
) -> list[tuple[str, str, int]]:
    """Round-robin among faults rather than accidentally favoring Missing.

    Deterministic seeded ordering maintains source/task variety under a
    caller-configured hard cap.
    """
    by_kind = defaultdict(list)
    for item in candidates:
        by_kind[item[1]].append(item)
    rng = random.Random(seed)
    keys = sorted(by_kind)
    rng.shuffle(keys)
    for key in keys:
        by_kind[key].sort()
        rng.shuffle(by_kind[key])
    output = []
    while any(by_kind.values()):
        for key in keys:
            if by_kind[key]:
                output.append(by_kind[key].pop())
    return output


def enumerate_round_candidates(
    scene: dict, round_plan: dict, *,
    atomic_fault_kinds: list[str],
    max_candidates: int | None,
    max_distractors: int | None,
    max_fault_atoms: int = 1,
    delayed_evidence_probability: float = 0.0,
    seed: int = 17,
    base_episode: dict | None = None,
    max_parallel_tasks: int = 4,
) -> list[dict]:
    """Return preflight-valid one-fault scenarios for one feedback round.

    The caller attaches stable *global* indices after all Scene/round
    candidates have been created. No language model is used here.
    """
    if ((max_candidates is not None and max_candidates < 1)
            or (max_distractors is not None and max_distractors < 1)):
        raise ValueError("candidate_exploration_limit_invalid")
    if any(kind not in ATOM_TO_MODE for kind in atomic_fault_kinds):
        raise ValueError("unsupported_candidate_kind")
    original_reason = round_plan["skip_reason"]
    if original_reason == "insufficient_recovery_round_budget":
        return [deepcopy(round_plan)]
    tasks = {t["task_id"]: t for t in scene["macro_graph"]["tasks"]}
    products = _clean_products(scene)
    eligible = []
    for tid in round_plan["incoming_task_ids"]:
        task = tasks[tid]
        for kind in atomic_fault_kinds:
            if kind not in _VARIANT_KINDS:
                variants = 1
            elif max_distractors is not None:
                variants = max_distractors
            elif kind == "computation_error":
                variants = 2
            else:
                # Exhaust finite *real* alternatives from already granted
                # observations and upstream products. More source records
                # may yield more valid variants, but never invented values.
                source_rows = sum(
                    len(scene["world"]["sources"][sid].get("observations", []))
                    for sid in task.get("source_ids", [])
                )
                parent_rows = sum(
                    len(relation_rows(products[pid]))
                    for pid in task.get("depends_on", [])
                )
                variants = max(1, (
                    source_rows + parent_rows
                ) * max(1, len(task["outputs"]["typed_relation"])))
            for variant in range(variants):
                if _candidate_is_real(
                    scene, tasks[tid], kind, variant, products
                ):
                    eligible.append((tid, kind, variant))
    # Method B: if a prior source-backed observation is already visible,
    # a later Main may legitimately delegate another observation of that
    # same source. Its result can disagree after a real one-atom fault.
    # The extra report MUST have an accompanying Main assignment.
    rechecks = discover_same_source_rechecks(
        scene, round_plan, products, base_episode=base_episode,
        atomic_fault_kinds=atomic_fault_kinds,
        max_parallel_tasks=max_parallel_tasks,
    )
    # Method C: one old report is corrupted, actual descendants execute
    # BEFORE a clean late recheck reveals the discrepancy. Only candidates
    # with an already-dispatched intermediate consumer are meaningful.
    waves = scene["macro_graph"]["reference_waves"]
    delayed = []
    for repeat in rechecks:
        origin = repeat["older_wave_index"]
        newer = repeat["new_wave_index"]
        old_id = repeat["older_task_id"]
        middle = {
            tid for wave in waves[origin + 1:newer] for tid in wave
        }
        if not middle:
            continue
        queue, descendants = [old_id], set()
        while queue:
            current = queue.pop()
            for task in tasks.values():
                tid = task["task_id"]
                if current in task["depends_on"] and tid not in descendants:
                    descendants.add(tid)
                    queue.append(tid)
        if not descendants & middle:
            continue
        if not _candidate_is_real(
            scene, tasks[old_id], repeat["fault_kind"],
            repeat["fault_variant"], products,
        ):
            continue
        delayed.append({
            **repeat,
            "construction": "delayed_consumed_error_recovery",
            "feeds_future_tasks": [],  # Keep the ORIGINAL dependency chain.
            "consumers_before_discovery": sorted(descendants & middle),
        })
    # Keep an explicit denominator when none of the three genuine causal
    # construction routes can execute.
    if not eligible and not rechecks and not delayed:
        skipped = deepcopy(round_plan)
        skipped["faults"] = []
        skipped["fault_mode"] = None
        skipped["skip_reason"] = "no_executable_fault_on_returned_batch"
        skipped["sample_id"] += "-candidate-none"
        skipped["candidate_count_total"] = 0
        skipped["candidate_count_pruned"] = 0
        skipped["candidate_variant_index"] = None
        skipped["scenario_plan"] = make_scenario(
            scene, skipped, mode="event_driven",
            max_fault_atoms=1, delayed_evidence_probability=0.0,
        )
        return [skipped]

    local_seed = (
        seed + round_plan["scene_index"] * 1009
        + round_plan["branch_round"] * 92821
    )
    ordered = _stable_candidate_order(eligible, local_seed)
    # True history-first discovery: anchor previous public keys before
    # deciding which erroneous newest records to synthesize. Reserve at
    # most one B candidate per feedback round, retaining A fault variety.
    handoffs = discover_handoffs(
        scene, round_plan, products, base_episode=base_episode,
    )
    history = {}
    for tid, kind, variant in ordered:
        plan = handoffs.get(tid)
        if not plan or kind not in {"wrong_retrieval", "wrong_entity_binding",
                                  "fabricated_claim", "legacy_wrong_evidence"}:
            continue
        fault = AtomicFault(
            kind, tid, origin_round=round_plan["branch_round"] - 1,
            variant_index=variant,
        ).as_legacy_fault()
        if fault_breaks_handoff(scene, plan, fault, products):
            history[(tid, kind, variant)] = plan
    first_history = next((row for row in ordered if row in history), None)
    prioritized = ([first_history] if first_history else []) + [
        row for row in ordered if row != first_history
    ]
    if max_candidates is None:
        sampled = prioritized
    else:
        repeat_budget = min(len(rechecks), max_candidates // 2)
        sampled = prioritized[:max_candidates - repeat_budget]
    result = []
    for slot, (tid, kind, variant) in enumerate(sampled, 1):
        item = deepcopy(round_plan)
        item["sample_id"] = (
            round_plan["sample_id"] + f"-candidate-{slot:03d}"
        )
        item["candidate_count_total"] = len(ordered) + len(rechecks)
        item["candidate_count_pruned"] = 0
        item["candidate_variant_index"] = variant
        item["skip_reason"] = None
        item["fault_mode"] = kind
        item["faults"] = [
            AtomicFault(
                kind, tid,
                origin_round=item["branch_round"] - 1,
                seed=local_seed + slot,
                variant_index=variant,
            ).as_legacy_fault()
        ]
        item["scenario_plan"] = make_scenario(
            scene, item, mode="event_driven",
            max_fault_atoms=max_fault_atoms,
            delayed_evidence_probability=delayed_evidence_probability,
        )
        if (tid, kind, variant) in history:
            item["scenario_plan"]["history_first_plan"] = history[
                (tid, kind, variant)
            ]
        result.append(item)

    if max_candidates is None:
        extra = rechecks
    else:
        by_history = {}
        for repeat in rechecks:
            gap = repeat["new_wave_index"] - repeat["older_wave_index"]
            by_history.setdefault(gap, []).append(repeat)
        for repeats in by_history.values():
            repeats.sort(key=lambda repeat: not bool(repeat.get("feeds_future_tasks")))
        diverse = []
        while any(by_history.values()):
            for gap in sorted(by_history, reverse=True):
                if by_history[gap]:
                    diverse.append(by_history[gap].pop(0))
        extra = diverse[:max(0, max_candidates - len(result))]
    for patch in extra:
        slot = len(result) + 1
        item = deepcopy(round_plan)
        item["sample_id"] += f"-candidate-{slot:03d}"
        item["candidate_count_total"] = len(ordered) + len(rechecks)
        item["candidate_variant_index"] = patch["fault_variant"]
        item["skip_reason"] = None
        item["fault_mode"] = patch["fault_kind"]
        item["incoming_task_ids"] = [
            *item["incoming_task_ids"], patch["new_task"]["task_id"]
        ]
        item["faults"] = [AtomicFault(
            patch["fault_kind"], patch["new_task"]["task_id"],
            origin_round=item["branch_round"] - 1,
            seed=local_seed + slot,
            variant_index=patch["fault_variant"],
        ).as_legacy_fault()]
        item["scenario_plan"] = make_scenario(
            scene, item, mode="event_driven",
            max_fault_atoms=max_fault_atoms,
            delayed_evidence_probability=0.0,
        )
        item["scenario_plan"]["same_source_recheck_plan"] = patch
        result.append(item)
    # Method C does not mutate or rewrite old feedback. The same clean
    # Stage-3 dispatch sequence is replayed using the EARLY poisoned
    # WorkProduct and actual downstream operator execution, then a later
    # clean recheck is delegated before the fork decision.
    for patch in delayed:
        if max_candidates is not None and len(result) >= max_candidates:
            break
        slot = len(result) + 1
        item = deepcopy(round_plan)
        item["sample_id"] += f"-candidate-{slot:03d}"
        item["candidate_variant_index"] = patch["fault_variant"]
        item["skip_reason"] = None
        item["fault_mode"] = patch["fault_kind"]
        item["incoming_task_ids"] = [
            *item["incoming_task_ids"], patch["new_task"]["task_id"]
        ]
        old_id = patch["older_task_id"]
        item["faults"] = [AtomicFault(
            patch["fault_kind"], old_id,
            origin_round=patch["older_wave_index"] + 1,
            seed=local_seed + slot,
            variant_index=patch["fault_variant"],
        ).as_legacy_fault()]
        item["scenario_plan"] = make_scenario(
            scene, item, mode="event_driven",
            max_fault_atoms=1, delayed_evidence_probability=0.0,
        )
        item["scenario_plan"]["blueprint"] = (
            "delayed_consumed_error_recovery"
        )
        item["scenario_plan"]["fault_atoms"][0]["origin_round"] = (
            patch["older_wave_index"] + 1
        )
        item["scenario_plan"]["same_source_recheck_plan"] = patch
        result.append(item)
    total = len(ordered) + len(rechecks) + len(delayed)
    for item in result:
        item["candidate_count_total"] = total
        item["candidate_count_pruned"] = total - len(result)
    return result


def expand_all_rounds(
    scenes: list[dict], round_plans: list[dict], *,
    atomic_fault_kinds: list[str],
    max_candidates_per_round: int | None,
    max_distractors_per_fault: int | None,
    max_fault_atoms: int,
    delayed_evidence_probability: float,
    seed: int,
    base: dict[str, dict] | None = None,
    max_parallel_tasks: int = 4,
) -> list[dict]:
    """Expand each historical feedback round into eligible independent trials."""
    expanded = []
    for original in round_plans:
        scene = scenes[original["scene_index"]]
        children = enumerate_round_candidates(
            scene, original, atomic_fault_kinds=atomic_fault_kinds,
            max_candidates=max_candidates_per_round,
            max_distractors=max_distractors_per_fault,
            max_fault_atoms=max_fault_atoms,
            delayed_evidence_probability=delayed_evidence_probability,
            seed=seed,
            base_episode=(base or {}).get(scene["task_id"]),
            max_parallel_tasks=max_parallel_tasks,
        )
        for child in children:
            child["index"] = len(expanded)
            # The scenario is a private plan; the candidate's identity
            # must match its final stable round identifier.
            child["scenario_plan"]["scenario_id"] = child["sample_id"]
            expanded.append(child)
    return expanded
