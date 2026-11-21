"""Offline Stage-1 -> Stage-4 independent-source B3 pilot.

No LLM services, dataset surgery, historical feedback edits or oracle-based
visibility labels. This intentionally samples *new* Stage-1 scenes and only
changes one latest received Sub-Agent report in each counterfactual.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

from environment.tests.current_stage1 import current_scene
from environment.implement_subtask.realize import realize_scene
from environment.implement_subtask.reporting import code_owned_result
from environment.base_orchestration.trajectory_builder import build_episode
from environment.untrusted_orchestration.runtime.faults import Fault
from environment.untrusted_orchestration.trajectory_builder import build_branch
from environment.untrusted_orchestration.scenario.single_turn_selection import (
    current_public_witness,
)
from environment.untrusted_orchestration.scenario.observable_scenarios import (
    classify_observable_scenario, comparable_incoming_tasks,
)


def _reviewed_teacher_reports(scene: dict) -> list[dict]:
    """Keep exact code-owned typed values, full citations and Stage-2 scope."""
    rows = []
    for record in realize_scene(scene)["macro_tasks"]:
        row = deepcopy(record)
        row["finish_result"] = {
            "status": "done",
            "result": code_owned_result(record),
            "summary": "Completed the assigned source-scoped evidence work.",
        }
        row["report_authoring"] = {"status": "accepted"}
        rows.append(row)
    return rows


def run_pilot(*, target_scenes: int = 2, max_scene_scan: int = 50,
              max_fault_variants: int = 24) -> dict:
    results = []
    checked, corroborated, planned = 0, 0, 0
    for ordinal in range(max_scene_scan):
        scene = current_scene(ordinal)
        checked += 1
        plan = scene["problem_spec"]["macro_plan"]
        if not plan["information_requirements"]["requested"][
            "independent_corroboration"
        ]:
            continue
        corroborated += 1
        siblings = [
            task for task in scene["macro_graph"]["tasks"]
            if task.get("macro_role") == "association"
            and len(task.get("source_ids", [])) == 1
            and task["source_ids"][0] in {"link_a", "link_b"}
            and set(task["outputs"]["typed_relation"])
                == {"route_key", "entity_0"}
        ]
        if len(siblings) != 2:
            results.append({
                "question_id": scene["task_id"], "status": "no_sibling_macros",
            })
            continue
        reports = _reviewed_teacher_reports(scene)
        base, _, _, status = build_episode(
            scene, reports, freeze_dispatch=True,
        )
        if status["status"] != "accepted":
            results.append({
                "question_id": scene["task_id"],
                "status": "stage3_" + str(status["status"]),
                "detail": status,
            })
            continue
        planned += 1
        waves = scene["macro_graph"]["reference_waves"]
        wave_index = next((
            i for i, wave in enumerate(waves)
            if all(t["task_id"] in wave for t in siblings)
        ), None)
        if wave_index is None:
            results.append({
                "question_id": scene["task_id"],
                "status": "sibling_reports_not_same_wave",
            })
            continue
        round_number = wave_index + 2
        incoming = [x["task_id"] for x in siblings]
        comparable = comparable_incoming_tasks(
            scene, base, {
                "incoming_task_ids": incoming,
                "branch_round": round_number,
            },
        )
        if set(incoming) - comparable:
            results.append({
                "question_id": scene["task_id"],
                "status": "stage3_claims_not_comparable",
                "round": round_number,
                "comparable": sorted(comparable),
            })
            continue
        found = None
        attempts = 0
        errors = []
        for target in incoming:
            for variant in range(max_fault_variants):
                try:
                    _, turns, audit = build_branch(
                        scene, reports, base,
                        branch_round=round_number,
                        faults=[Fault(
                            target, "report", atom_kind="fabricated_claim",
                            variant_index=variant,
                        )],
                    )
                except (ValueError, RuntimeError) as exc:
                    errors.append(type(exc).__name__ + ":" + str(exc)[:110])
                    continue
                attempts += 1
                issues, conflicts = current_public_witness(
                    audit, round_number,
                )
                realized = classify_observable_scenario(issues, conflicts)
                if (realized["scenario_id"] == "B3"
                        and any(
                            c.get("kind")
                            == "explicit_independent_claim_disagreement"
                            for c in conflicts
                        )):
                    found = {
                        "question_id": scene["task_id"],
                        "status": "b3_observable",
                        "round": round_number,
                        "faulted_task": target,
                        "variant": variant,
                        "realized": realized,
                        "public_conflicts": conflicts,
                        "stage3_prior_history_preserved": audit.get(
                            "stage3_prior_history_preserved",
                        ),
                        "decision_unit": audit.get("decision_unit"),
                        "next_main_turns": len(turns),
                    }
                    break
            if found:
                break
        results.append(found or {
            "question_id": scene["task_id"],
            "status": "no_visible_b3_after_variants",
            "round": round_number,
            "candidate_attempts": attempts,
            "errors": errors[:5],
        })
        if len(results) >= target_scenes:
            break
    return {
        "pilot_type": "offline_synthetic_motif_not_llm_reviewed",
        "scene_scan_count": checked,
        "corroboration_scenes_seen": corroborated,
        "stage3_clean_episodes_built": planned,
        "requested_scenes": target_scenes,
        "observable_b3": sum(
            row["status"] == "b3_observable" for row in results
        ),
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-scenes", type=int, default=2)
    parser.add_argument("--max-scene-scan", type=int, default=50)
    parser.add_argument("--max-fault-variants", type=int, default=24)
    args = parser.parse_args()
    summary = run_pilot(
        target_scenes=args.target_scenes,
        max_scene_scan=args.max_scene_scan,
        max_fault_variants=args.max_fault_variants,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print("B3_PILOT_RESULTS=" + json.dumps({
        key: value for key, value in summary.items() if key != "results"
    }, ensure_ascii=False))
    for row in summary["results"]:
        print(row["question_id"] + ": " + row["status"])


if __name__ == "__main__":
    main()
