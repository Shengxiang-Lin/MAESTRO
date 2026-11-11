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

from environment.generate_question.program_ir import ProgramError
from .atomic_faults import AtomicFault, ATOM_TO_MODE
from .faults import FaultUnsupported, relation_rows, run_attempt
from .scenario_generator import make_scenario


_VARIANT_KINDS = {
    "wrong_retrieval", "wrong_entity_binding",
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
    if kind in {"no_result", "incomplete_retrieval"}:
        return (
            mutated is None if kind == "no_result"
            else runtime.get("status") == "partial"
        )
    if kind in {"wrong_retrieval", "wrong_entity_binding",
                "legacy_wrong_evidence"}:
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
    max_candidates: int,
    max_distractors: int,
    max_fault_atoms: int = 1,
    delayed_evidence_probability: float = 0.0,
    seed: int = 17,
) -> list[dict]:
    """Return preflight-valid one-fault scenarios for one feedback round.

    The caller attaches stable *global* indices after all Scene/round
    candidates have been created. No language model is used here.
    """
    if max_candidates < 1 or max_distractors < 1:
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
        for kind in atomic_fault_kinds:
            variants = max_distractors if kind in _VARIANT_KINDS else 1
            for variant in range(variants):
                if _candidate_is_real(
                    scene, tasks[tid], kind, variant, products
                ):
                    eligible.append((tid, kind, variant))
    # A single unsupported placeholder preserves per-round coverage
    # and a truthful denominator in the Stage-4A audit.
    if not eligible:
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
    sampled = ordered[:max_candidates]
    result = []
    for slot, (tid, kind, variant) in enumerate(sampled, 1):
        item = deepcopy(round_plan)
        item["sample_id"] = (
            round_plan["sample_id"] + f"-candidate-{slot:03d}"
        )
        item["candidate_count_total"] = len(ordered)
        item["candidate_count_pruned"] = max(0, len(ordered) - max_candidates)
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
        result.append(item)
    return result


def expand_all_rounds(
    scenes: list[dict], round_plans: list[dict], *,
    atomic_fault_kinds: list[str],
    max_candidates_per_round: int,
    max_distractors_per_fault: int,
    max_fault_atoms: int,
    delayed_evidence_probability: float,
    seed: int,
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
        )
        for child in children:
            child["index"] = len(expanded)
            # The scenario is a private plan; the candidate's identity
            # must match its final stable round identifier.
            child["scenario_plan"]["scenario_id"] = child["sample_id"]
            expanded.append(child)
    return expanded
