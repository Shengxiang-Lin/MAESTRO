"""Enumerate executable one-atom, Main-visible scenario targets.

The frozen Stage-3 history and Macro DAG are never rewritten. A candidate
becomes positive only when the Stage-4A public-witness check succeeds.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import random

from environment.generate_question.execution.program_ir import ProgramError
from environment.untrusted_orchestration.runtime.atomic_faults import AtomicFault, ATOM_TO_MODE
from environment.untrusted_orchestration.runtime.faults import FaultUnsupported, relation_rows, run_attempt
from .scenario_generator import make_scenario
from .observable_scenarios import (
    CONSTRUCTION_ORDER, feasible_fault_targets,
)


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



def _variant_count(
    scene: dict, task: dict, kind: str, products: dict,
    max_distractors: int | None,
) -> int:
    """One shared finite enumeration bound for both generation paths."""
    if kind not in _VARIANT_KINDS:
        return 1
    if max_distractors is not None:
        return max_distractors
    if kind == "computation_error":
        return 4
    source_rows = sum(
        len(scene["world"]["sources"][sid].get("observations", []))
        for sid in task.get("source_ids", [])
    )
    parent_rows = sum(
        len(relation_rows(products[pid]))
        for pid in task.get("depends_on", [])
    )
    return max(1, (source_rows + parent_rows) *
               max(1, len(task["outputs"]["typed_relation"])))


def _enumerate_scenario_first(
    scene: dict, round_plan: dict, *,
    atomic_fault_kinds: list[str], max_candidates: int | None,
    max_distractors: int | None, seed: int, base_episode: dict | None,
) -> list[dict]:
    """Current Stage-4 main path: A/B1/B2/B3 -> legal operator -> public audit.

    No historical augmentation, extra Main delegation, future recovery
    rollout, or secret Gold is used to establish the *target* scenario.
    A target is only a plan until 4A independently verifies public issues.
    """
    targets = feasible_fault_targets(
        scene, round_plan, kinds=atomic_fault_kinds,
        base_episode=base_episode,
    )
    tasks = {task["task_id"]: task for task in scene["macro_graph"]["tasks"]}
    products = _clean_products(scene)
    seed_local = (
        seed + round_plan["scene_index"] * 1009
        + round_plan["branch_round"] * 92821
    )
    eligible = defaultdict(list)
    for target, tid, kind in targets:
        for variant in range(_variant_count(
            scene, tasks[tid], kind, products, max_distractors,
        )):
            if _candidate_is_real(scene, tasks[tid], kind, variant, products):
                eligible[target].append((tid, kind, variant))
    ordered = [
        (target, tid, kind, variant)
        for target in CONSTRUCTION_ORDER
        for tid, kind, variant in _stable_candidate_order(
            eligible[target], seed_local + CONSTRUCTION_ORDER.index(target),
        )
    ]
    if not ordered:
        skipped = deepcopy(round_plan)
        skipped.update({
            "faults": [], "fault_mode": None,
            "skip_reason": "no_executable_scenario_target_on_returned_batch",
            "sample_id": round_plan["sample_id"] + "-candidate-none",
            "candidate_count_total": 0, "candidate_count_pruned": 0,
            "candidate_variant_index": None, "target_scenario": None,
        })
        skipped["scenario_plan"] = make_scenario(scene, skipped)
        return [skipped]
    sampled = ordered if max_candidates is None else ordered[:max_candidates]
    trials = []
    for slot, (target, tid, kind, variant) in enumerate(sampled, 1):
        item = deepcopy(round_plan)
        item.update({
            "sample_id": round_plan["sample_id"] + f"-candidate-{slot:03d}",
            "target_scenario": target,
            "candidate_count_total": len(ordered),
            "candidate_count_pruned": len(ordered) - len(sampled),
            "candidate_variant_index": variant,
            "skip_reason": None, "fault_mode": kind,
            "faults": [AtomicFault(
                kind, tid, origin_round=round_plan["branch_round"] - 1,
                seed=seed_local + slot, variant_index=variant,
            ).as_legacy_fault()],
        })
        item["scenario_plan"] = make_scenario(scene, item)
        item["scenario_plan"]["target_scenario"] = target
        trials.append(item)
    return trials


def enumerate_round_candidates(
    scene: dict, round_plan: dict, *,
    atomic_fault_kinds: list[str],
    max_candidates: int | None = None,
    max_distractors: int | None = None,
    seed: int = 17,
    base_episode: dict | None = None,
) -> list[dict]:
    """Enumerate one-atom variants for a frozen latest-feedback batch.

    Stage 4 has no delayed faults, historical prefix rewrites, synthetic
    witnesses, or multi-round continuation. Stage 4A separately determines
    whether each target is actually observable to Main.
    """
    if ((max_candidates is not None and max_candidates < 1)
            or (max_distractors is not None and max_distractors < 1)):
        raise ValueError("candidate_exploration_limit_invalid")
    if any(kind not in ATOM_TO_MODE for kind in atomic_fault_kinds):
        raise ValueError("unsupported_candidate_kind")
    if round_plan["skip_reason"]:
        return [deepcopy(round_plan)]
    return _enumerate_scenario_first(
        scene, round_plan, atomic_fault_kinds=atomic_fault_kinds,
        max_candidates=max_candidates, max_distractors=max_distractors,
        seed=seed, base_episode=base_episode,
    )


def expand_all_rounds(
    scenes: list[dict], round_plans: list[dict], *,
    atomic_fault_kinds: list[str],
    max_candidates_per_round: int | None = None,
    max_distractors_per_fault: int | None = None,
    seed: int = 17,
    base: dict[str, dict] | None = None,
) -> list[dict]:
    """Expand frozen feedback rounds without mutating their Stage-3 prefix."""
    expanded = []
    for original in round_plans:
        scene = scenes[original["scene_index"]]
        children = enumerate_round_candidates(
            scene, original, atomic_fault_kinds=atomic_fault_kinds,
            max_candidates=max_candidates_per_round,
            max_distractors=max_distractors_per_fault,
            seed=seed, base_episode=(base or {}).get(scene["task_id"]),
        )
        for child in children:
            child["index"] = len(expanded)
            child["scenario_plan"]["scenario_id"] = child["sample_id"]
            expanded.append(child)
    return expanded
