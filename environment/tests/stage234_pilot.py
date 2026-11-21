"""Small reproducible Stage-2/3/4 pairing pilot (no LLM endpoint).

Uses synthetic Task Plan fixtures, the real Macro Oracle and public Finish
Results. Stage 3 and Stage 4 use deterministic template-language mode so
CI can check evidence/decision contracts without API credentials.
This is NOT an LLM Author/Critic acceptance experiment.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

from environment.tests.current_stage1 import current_scene
from environment.implement_subtask.realize import realize_scene
from environment.implement_subtask.reporting import code_owned_result
from environment.base_orchestration.trajectory_builder import build_episode
from environment.untrusted_orchestration.generate_turns import generate_turns
from environment.data_export.export import export_dataset
from environment.data_export.source_io import digest


CASES = ((0, "medium"), (1, "hard"), (2, "medium"), (3, "hard"))
FAULTS = (
    "no_result", "incomplete_retrieval", "wrong_retrieval",
    "wrong_entity_binding", "report_missing", "result_summary_conflict",
    "fabricated_claim",
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(
        encoding="utf-8").splitlines() if line.strip()]


def run_pilot(output_dir: Path, *, count: int = 4,
              feedback_rounds: int = 2) -> dict:
    if count not in range(1, len(CASES) + 1):
        raise ValueError("stage234_pilot_scene_count_out_of_range")
    if feedback_rounds not in range(1, 5):
        raise ValueError("stage234_pilot_round_cap_out_of_range")
    output_dir.mkdir(parents=True, exist_ok=True)
    scenes, reports, episodes, turns = [], [], [], []
    workload = []
    for index, difficulty in CASES[:count]:
        scene = current_scene(index, difficulty)
        teacher = realize_scene(scene)
        current_reports = []
        for item in teacher["macro_tasks"]:
            delivered = {
                **deepcopy(item),
                "finish_result": {
                    "status": "done",
                    "result": code_owned_result(item),
                    "summary": (
                        "Returned the exact evidence for this delegated "
                        "Macro, without making downstream claims."
                    ),
                },
                "report_authoring": {
                    "status": "accepted",
                    "mode": "pilot_deterministic_not_llm_reviewed",
                },
            }
            current_reports.append(delivered)
        clean, clean_turns, _audit, status = build_episode(
            scene, current_reports, freeze_dispatch=True, max_attempts=16,
        )
        if clean is None or status.get("status") != "accepted":
            raise RuntimeError("stage234_pilot_clean_rejected:" + repr(status))
        scenes.append(scene)
        reports.extend(current_reports)
        episodes.append(clean)
        turns.extend(clean_turns)
        workload.append({
            "task_id": scene["task_id"], "difficulty": difficulty,
            "macros": len(current_reports),
            "sources": len(scene["world"]["sources"]),
            "clean_rounds": len(clean["trajectory"]),
            "clean_delegated_batches": sum(
                row["action"] == "delegate_task" for row in clean["trajectory"]
            ),
            "clean_reference_id": clean["clean_reference_id"],
        })

    _write_jsonl(output_dir / "scenes.jsonl", scenes)
    _write_jsonl(output_dir / "stage2" / "subtasks.jsonl", reports)
    _write_jsonl(output_dir / "stage3" / "episodes.jsonl", episodes)
    _write_jsonl(output_dir / "stage3" / "main_turns.jsonl", turns)
    # These frozen offline fixtures are generated directly through
    # build_episode rather than the Stage-3 CLI. Supply the matching
    # source manifest required by the canonical exporter, including
    # its scene digest and exact O1 prompt settings.
    (output_dir / "stage3" / "manifest.json").write_text(
        json.dumps({
            "scenes_sha256": digest(output_dir / "scenes.jsonl"),
            "max_attempts": 16,
            "sub_model": "qwen3-8b",
            "model_alias": "model_1",
        }, sort_keys=True),
        encoding="utf-8",
    )

    stage4 = generate_turns(
        scenes_path=output_dir / "scenes.jsonl",
        reports_path=output_dir / "stage2" / "subtasks.jsonl",
        clean_episodes_path=output_dir / "stage3" / "episodes.jsonl",
        output_dir=output_dir / "stage4",
        max_scenes=count, branches_per_scene=feedback_rounds,
        seed=17, decision_unit="single_turn",
        atomic_fault_kinds=list(FAULTS),
        # Exhaust feasible scenario targets in 4A; the smoke's expensive
        # language work is still bounded by one selected candidate per wave.
        max_candidates_per_round=None,
        max_distractors_per_fault=1,
        max_selected_per_round=1,
        max_language_attempts_per_round=1,
        max_rounds=16, max_retries=1, max_parallel_tasks=4,
        workers=2, no_progress=True, overwrite=True,
        main_mode="template", report_mode="template",
    )
    prefix_checks = _read_jsonl(
        output_dir / "stage4" / "private" / "trusted_prefix_checks.jsonl"
    )
    if len(prefix_checks) != count or any(
        row.get("status") != "accepted"
        or row.get("source") != "validated_stage3_clean_reference"
        for row in prefix_checks
    ):
        raise RuntimeError("stage234_pilot_stage3_reference_not_reused")

    pairs = _read_jsonl(
        output_dir / "stage4" / "private" / "paired_trajectories.jsonl"
    )
    known = {item["task_id"]: item["clean_reference_id"]
             for item in episodes}
    if len(pairs) != stage4["completed_branches"] or any(
        pair.get("clean_reference_id") != known[pair["scene_task_id"]]
        for pair in pairs
    ):
        raise RuntimeError("stage234_pilot_pairing_identity_mismatch")

    public = _read_jsonl(output_dir / "stage4" / "episodes.jsonl")
    if any(e.get("clean_reference_id") != known[e["task_id"]]
           for e in public):
        raise RuntimeError("stage234_pilot_public_episode_reference_mismatch")
    if (stage4["decision_unit"] != "single_turn"
            or not public
            or len(public) != stage4["single_turn_accepted"]
            or any(e.get("format") != "maestro_untrusted_single_turn.v1"
                   for e in public)
            or stage4["published_single_turn_stage3_prior_history"].get(
                "preserved", 0
            ) != len(public)):
        raise RuntimeError("stage234_pilot_single_turn_contract_failed")

    unified = export_dataset(
        scenes_path=output_dir / "scenes.jsonl",
        clean_episodes_path=output_dir / "stage3" / "episodes.jsonl",
        clean_turns_path=output_dir / "stage3" / "main_turns.jsonl",
        stage4_dir=output_dir / "stage4",
        output_dir=output_dir / "unified",
        overwrite=True,
    )
    if (unified["accepted_fault_cases"] != len(public)
            or unified["decision_sources"].get("untrusted") != len(public)):
        raise RuntimeError("stage234_pilot_single_turn_export_mismatch")

    summary = {
        "mode": "offline_deterministic_no_llm",
        "scene_count": len(scenes),
        "macro_count": len(reports),
        "clean_episode_count": len(episodes),
        "clean_turn_count": len(turns),
        "clean_references_verified": len(prefix_checks),
        "decision_unit": stage4["decision_unit"],
        "unified_export_decisions": unified["decision_count"],
        "unified_export_accepted_fault_cases": unified["accepted_fault_cases"],
        "fault_trials": stage4["candidate_trials"],
        "published_scenario_types": stage4.get("published_scenario_types", {}),
        "candidate_screening_outcomes": stage4.get(
            "candidate_screening_outcomes", {}
        ),
        "fault_feedback_rounds": stage4["rounds_extracted_from_base"],
        "paired_index_rows": len(pairs),
        "fault_kinds_planned": stage4.get("atomic_fault_kinds_planned", {}),
        "execution_outcomes": stage4.get("execution_outcomes", {}),
        "publication_outcomes": stage4.get("outcomes", {}),
        "published_robust_episodes": len(public),
        "published_robust_turns": stage4["main_training_turns"],
        "visible_claim_state_labels": stage4.get(
            "claim_state_status_distribution", {}
        ),
        "naive_outcomes": dict(Counter(
            pair.get("naive_execution_status") or "not_recorded"
            for pair in pairs
        )),
        "workload": workload,
    }
    with (output_dir / "stage234_pilot_summary.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print("STAGE234_PILOT_RESULTS=" + json.dumps(summary, ensure_ascii=False,
                                             sort_keys=True), flush=True)
    if not summary["fault_trials"] or not summary["fault_feedback_rounds"]:
        raise RuntimeError("stage234_pilot_missing_fault_trials")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--feedback-rounds", type=int, default=2)
    args = parser.parse_args()
    run_pilot(args.output_dir, count=args.count,
              feedback_rounds=args.feedback_rounds)


if __name__ == "__main__":
    main()
