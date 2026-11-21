"""Private one-atom plan for the latest frozen Sub-Agent feedback batch.

The source Macro DAG and all earlier Stage-3 reports are immutable. No
future fault events, delayed witnesses, multiple atoms or recovery rollout.
"""
from __future__ import annotations

from environment.untrusted_orchestration.runtime.atomic_faults import legacy_atom


def make_scenario(scene: dict, plan: dict) -> dict:
    """Build one latest-feedback fault; 4A must confirm public observability."""
    faults = plan.get("faults", [])
    if len(faults) > 1:
        raise ValueError("single_turn_requires_one_atomic_fault")
    atoms = [
        legacy_atom(
            fault, origin_round=plan["branch_round"] - 1,
            atom_id=f"F{index + 1:03d}",
        ).to_private_dict()
        for index, fault in enumerate(faults)
    ]
    return {
        "version": "maestro.single_turn_scenario.v1",
        "scenario_id": plan["sample_id"],
        "question_id": plan["task_id"],
        "origin_feedback_round": plan["branch_round"],
        "mode": "single_turn",
        "blueprint": "immediate_feedback_fault",
        "fault_atoms": atoms,
        "evidence_events": [],
        "acceptance": {
            "no_future_evidence": True,
            "real_source_distractors_only": True,
            "no_implicit_new_macro": True,
        },
    }
