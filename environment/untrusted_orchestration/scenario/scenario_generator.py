"""Constraint-guided event scenario compiler.

Every ordinary v8 latest-feedback fork is a one-atom ScenarioPlan. Opt-in
event-driven sampling may use a specific atomic fault, an additional
future-execution atom, and/or delay an INDEPENDENT not-yet-executed
investigation. Neither Gold nor a desired recovery action is put into
the Main-visible prompt.

The compiler is intentionally conservative: it does not invent new
sources, mutate a historical Main action, or pretend an unknown report
arrival happened without a corresponding executable task.
"""
from __future__ import annotations

from copy import deepcopy
import random

from ..atomic_faults import AtomicFault, legacy_atom
from ..event_scheduler import EvidenceEvent


def _dependents(tasks: dict[str, dict], tid: str) -> set[str]:
    reached, queue = set(), [tid]
    while queue:
        node = queue.pop(0)
        for child in tasks.values():
            cid = child["task_id"]
            if cid not in reached and node in child["depends_on"]:
                reached.add(cid)
                queue.append(cid)
    return reached


def _source_lineage(tasks: dict[str, dict], tid: str) -> set[str]:
    """Direct plus inherited source grants for independent-source checks."""
    visited, stack, grants = set(), [tid], set()
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        task = tasks[node]
        grants.update(task.get("source_ids", []))
        stack.extend(task.get("depends_on", []))
    return grants


def delayed_event_candidate(scene: dict, *, fork_round: int,
                            initial_fault_task: str) -> dict | None:
    """Look for two legal post-fork investigations that can run separately.

    The anchor must use the faulty branch and be ready at the fork.
    The held investigation must be independent of the anchor and share
    a downstream consumer. If there is no such structure in Stage 1,
    an unsupported delayed scenario is NOT synthesized.
    """
    waves = scene["macro_graph"]["reference_waves"]
    tasks = {t["task_id"]: t for t in scene["macro_graph"]["tasks"]}
    prefix = {
        tid for wave in waves[:fork_round - 1] for tid in wave
    }
    if initial_fault_task not in prefix:
        return None
    ready = [
        tid for tid, t in tasks.items()
        if tid not in prefix and set(t["depends_on"]) <= prefix
    ]
    descendants = _dependents(tasks, initial_fault_task)
    anchors = [tid for tid in ready if tid in descendants]
    others = [
        tid for tid in ready if tid not in descendants
        and tid != initial_fault_task
    ]
    for anchor in sorted(anchors):
        anchor_children = _dependents(tasks, anchor)
        for held in sorted(others):
            if (held in anchor_children
                    or anchor in _dependents(tasks, held)):
                continue
            if not anchor_children.intersection(_dependents(tasks, held)):
                continue
            left_sources = _source_lineage(tasks, initial_fault_task)
            right_sources = _source_lineage(tasks, held)
            # Independent DAG positions are not sufficient if both
            # reports ultimately cite the SAME source material.
            if (not left_sources or not right_sources
                    or left_sources & right_sources):
                continue
            return EvidenceEvent(
                kind="hold_independent_evidence",
                task_id=held,
                earliest_round=fork_round + 1,
                after_task_id=anchor,
            ).to_private_dict()
    return None


def make_scenario(
    scene: dict, plan: dict, *, mode: str = "round_fork",
    eligible_atoms: dict[str, list[str]] | None = None,
    max_fault_atoms: int = 1,
    delayed_evidence_probability: float = 0.0,
) -> dict:
    """Compile one reproducible, causally bound private scenario plan."""
    if mode not in {"round_fork", "event_driven"}:
        raise ValueError("unknown_event_scenario_mode")
    if type(max_fault_atoms) is not int or not 1 <= max_fault_atoms <= 3:
        raise ValueError("invalid_max_atomic_faults")
    if not isinstance(delayed_evidence_probability, (int, float)) or (
        not 0 <= delayed_evidence_probability <= 1
    ):
        raise ValueError("invalid_delayed_evidence_probability")
    rng = random.Random(
        plan["index"] * 1000003 + plan["branch_round"] * 7919
        + plan["scene_index"] * 53
    )
    initial = [
        legacy_atom(
            fault, origin_round=plan["branch_round"] - 1,
            atom_id=f"F{index + 1:03d}",
        ) for index, fault in enumerate(plan.get("faults", []))
    ]
    atoms = list(initial)
    events = []
    requested_delayed = False
    if mode == "event_driven" and initial:
        if (len(atoms) < max_fault_atoms and eligible_atoms is not None):
            waves = scene["macro_graph"]["reference_waves"]
            later = [
                (wave_index + 1, tid, kind)
                for wave_index, wave in enumerate(waves)
                if wave_index >= plan["branch_round"] - 1
                for tid in wave
                for kind in eligible_atoms.get(tid, [])
                if tid not in {item.task_id for item in initial}
            ]
            if later:
                rnd, tid, kind = rng.choice(sorted(later))
                atoms.append(AtomicFault(
                    kind=kind, task_id=tid,
                    origin_round=rnd, seed=plan["index"] + 1009,
                    atom_id=f"F{len(atoms) + 1:03d}",
                ))
        requested_delayed = (
            rng.random() < delayed_evidence_probability
        )
        if requested_delayed:
            # Delaying independent corroboration is meaningful when a
            # plausible wrong value can still drive real downstream
            # work. A missing/partial handoff generally cannot do so;
            # attempting to hold its other prerequisite just deadlocks.
            eligible_for_delay = initial[0].kind in {
                "legacy_wrong_evidence", "wrong_retrieval",
                "wrong_entity_binding", "fabricated_claim",
            }
            option = (
                delayed_event_candidate(
                    scene, fork_round=plan["branch_round"],
                    initial_fault_task=initial[0].task_id,
                ) if eligible_for_delay else None
            )
            if option is not None:
                events.append(option)

    return {
        "version": "maestro.event_scenario.v1",
        "scenario_id": plan["sample_id"],
        "question_id": plan["task_id"],
        "origin_feedback_round": plan["branch_round"],
        "mode": mode,
        "blueprint": (
            "delayed_independent_evidence" if events
            else "compound_atomic_faults" if len(atoms) > 1
            else "immediate_feedback_fault"
        ),
        "fault_atoms": [item.to_private_dict() for item in atoms],
        "evidence_events": deepcopy(events),
        "delayed_requested_but_unsupported": (
            requested_delayed and not events
        ),
        "acceptance": {
            "no_future_evidence": True,
            "real_source_distractors_only": True,
            "no_implicit_new_macro": True,
            "actual_invalidations_audited": True,
        },
    }
