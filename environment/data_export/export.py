"""Export MAESTRO method-agnostic paired Cases and Main decisions.

This environment layer publishes *unsplit* JSONL and source provenance only.
Every baseline owns its own filtering, Scene split and trainer-specific format.
Private gold/reward state never enters the policy prompt.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import shutil

from tqdm.auto import tqdm

from environment.base_orchestration.o1_compat import o1_prompt_from_turn
from environment.base_orchestration.trajectory_builder import _answer_text
from .source_io import digest, read_jsonl, settings_for
from .audit import audit_dataset, audit_output

CASE_SCHEMA = "maestro.paired_case.v1"
DECISION_SCHEMA = "maestro.main_decision.v1"
LABEL_SCHEMA = "maestro.decision_teacher.v1"
DATASET_SCHEMA = "maestro.dataset.v1"


def _sha(value) -> str:
    return sha256(json.dumps(
        value, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def _index(items, field, description):
    result = {}
    for row in items:
        key = row.get(field)
        if not isinstance(key, str) or not key or key in result:
            raise ValueError("missing_or_duplicate_" + description)
        result[key] = row
    return result


def _write_jsonl(path: Path, rows, *, progress: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for item in tqdm(rows, total=len(rows), desc=f"Write {path.name}",
                         unit="row", disable=not progress, leave=False):
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    tmp.replace(path)


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def _check_clean(clean: dict, scene_id: str, scene: dict):
    if (clean.get("task_id") != scene_id
            or clean.get("question") != scene.get("question")
            or clean.get("trusted_subagent_feedback") is not True
            or not isinstance(clean.get("trajectory"), list)
            or not clean["trajectory"]):
        raise ValueError("invalid_stage3_clean_episode:" + scene_id)
    ref = clean.get("clean_reference_id")
    if not isinstance(ref, str) or not ref:
        raise ValueError("missing_stage3_clean_reference_id")
    return ref


def _case_for_clean(scene: dict, episode: dict):
    return {
        "schema": CASE_SCHEMA, "case_id": "clean:" + scene["task_id"],
        "scene_id": scene["task_id"],
        "question": scene["question"],
        "clean_reference_id": episode["clean_reference_id"],
        "kind": "clean",
        "clean": deepcopy(episode),
        "untrusted": None,
        "fault": {"atoms": [], "applied_events": []},
        "recovery": {
            "publication_status": "accepted",
            "execution_outcome": "clean_completed",
            "globally_recovered": None,
            "observable_issues": [],
            "main_decisions": [],
            "invalidations": [],
            "revalidations": [],
        },
    }


def _case_for_branch(
    scene: dict, clean: dict, entry: dict, journal: dict, *,
    published: bool,
):
    scenario_id = entry["round_sample_id"]
    audit = journal.get("private_audit") or {}
    ep = journal.get("episode")
    if not isinstance(ep, dict):
        return None
    if (journal.get("task_id") != scene["task_id"]
            or journal.get("round_sample_id") != scenario_id
            or ep.get("task_id") != scene["task_id"]
            or ep.get("branch_round") != entry["main_decision_round"]):
        raise ValueError("stage4_journal_identity_mismatch:" + scenario_id)
    if ep.get("clean_reference_id") != clean["clean_reference_id"]:
        raise ValueError("stage4_clean_reference_mismatch:" + scenario_id)
    recorded = (audit.get("clean_reference_identity") or {}).get(
        "clean_reference_id"
    )
    if recorded is not None and recorded != clean["clean_reference_id"]:
        raise ValueError("stage4_private_clean_identity_mismatch")
    if published and journal.get("status") != "recovered":
        raise ValueError("stage4_published_case_not_recovered")
    if published and audit.get("parallel_batches_verified") is not True:
        raise ValueError("stage4_published_case_not_dependency_safe")
    invalidations = (audit.get("runtime") or {}).get("invalidations") or []
    return {
        "schema": CASE_SCHEMA,
        "case_id": scenario_id,
        "scene_id": scene["task_id"],
        "question": scene["question"],
        "clean_reference_id": clean["clean_reference_id"],
        "kind": "fault_injected",
        "clean": deepcopy(clean),
        "untrusted": {
            "episode": deepcopy(ep),
            "fork_round": entry["main_decision_round"],
            "scenario_blueprint": entry.get("scenario_blueprint"),
            "publication_status": journal.get("status"),
        },
        # All of this belongs in the private paired Case, not in prompts.
        "fault": {
            "atoms": deepcopy(
                (audit.get("event_scenario") or {}).get("fault_atoms")
                or audit.get("faults") or []
            ),
            "applied_events": deepcopy(audit.get("fault_events_applied") or []),
            "injection_details": deepcopy(audit.get("faults") or []),
        },
        "recovery": {
            "publication_status": journal.get("status"),
            "execution_outcome": audit.get("execution_outcome"),
            "globally_recovered": published,
            "counterfactual_unrepaired": deepcopy(
                audit.get("counterfactual_unrepaired")
            ),
            "observable_issues": deepcopy(
                audit.get("observable_issue_snapshots") or []
            ),
            "main_decisions": [{
                "round": row.get("round"),
                "decision_kind": row.get("decision_kind"),
                "task_ids": row.get("task_ids", []),
                "mode": row.get("mode"),
            } for row in audit.get("decision_audit") or []],
            "invalidations": deepcopy(invalidations),
            "revalidations": deepcopy(
                (audit.get("runtime") or {}).get("revalidations") or []
            ),
            "delayed_causal_audit": deepcopy(
                audit.get("delayed_causal_audit")
            ),
        },
    }


def _stage4_cases(stage4_dir: Path, scenes: dict, clean: dict,
                  *, progress: bool = False):
    index_path = stage4_dir / "round_index.jsonl"
    if not index_path.is_file():
        raise FileNotFoundError("stage4_round_index_missing:" + str(index_path))
    published = _index(
        read_jsonl(stage4_dir / "episodes.jsonl", progress=progress,
                   desc="Read Stage4 episodes"),
        "round_sample_id", "stage4_published_episode",
    )
    cases, seen = [], set()
    status_counts = Counter()
    entries = read_jsonl(index_path, progress=progress, desc="Read Stage4 index")
    bar = tqdm(enumerate(entries), total=len(entries), desc="Stage4 Cases",
               unit="case", disable=not progress, mininterval=0.5)
    for number, item in bar:
        case_id = item["round_sample_id"]
        if case_id in seen:
            raise ValueError("duplicate_stage4_round_index_case")
        seen.add(case_id)
        journal_path = stage4_dir / "journal" / f"{number:06d}.json"
        if not journal_path.is_file():
            status_counts["missing_final_journal"] += 1
            continue
        data = json.loads(journal_path.read_text(encoding="utf-8"))
        if data.get("status") == "not_selected_for_language_review":
            # Stage4A may enumerate thousands of legitimate alternatives,
            # but these were not reviewed as trainable complete Episodes.
            status_counts["screened_out"] += 1
            continue
        scene_id = item["question_id"]
        if scene_id not in scenes or scene_id not in clean:
            raise ValueError("stage4_scene_without_clean_pair:" + scene_id)
        case = _case_for_branch(
            scenes[scene_id], clean[scene_id], item, data,
            published=case_id in published,
        )
        if case is not None:
            cases.append(case)
            status_counts["accepted" if case_id in published
                          else "diagnostic"] += 1
        else:
            status_counts["no_episode"] += 1
        if progress and (number + 1) % 128 == 0:
            bar.set_postfix(accepted=status_counts["accepted"],
                            diagnostic=status_counts["diagnostic"],
                            skipped=status_counts["screened_out"] +
                                    status_counts["missing_final_journal"],
                            refresh=False)
    if progress:
        bar.set_postfix(accepted=status_counts["accepted"],
                        diagnostic=status_counts["diagnostic"],
                        skipped=status_counts["screened_out"] +
                                status_counts["missing_final_journal"],
                        refresh=True)
    if not set(published) <= {
        c["case_id"] for c in cases if c["recovery"]["globally_recovered"]
    }:
        raise ValueError("unmatched_stage4_published_episodes")
    return cases, status_counts


def _build_decisions(
    sources: list[tuple[str, list[dict], dict]],
    scenes: dict[str, dict],
    published_cases: set[str],
    private_states: dict, *, progress: bool = False,
) -> tuple[list[dict], list[dict]]:
    """Reconstruct exact O1 prompts; do not filter labels for any trainer."""
    decisions, teacher = [], []
    identities = set()
    case_rounds = set()
    for stage, turns, settings in sources:
        actions = Counter()
        bar = tqdm(turns, total=len(turns), desc=f"{stage} Main Decisions",
                   unit="decision", disable=not progress)
        for index, turn in enumerate(bar):
            scene_id = turn.get("task_id")
            if scene_id not in scenes:
                raise ValueError("decision_unknown_scene:" + str(scene_id))
            number = turn.get("round")
            if type(number) is not int or not 1 <= number <= settings["max_attempts"]:
                raise ValueError("decision_invalid_round")
            if stage == "stage3":
                sample_id = turn.get("sample_id")
                if sample_id != f"{scene_id}-round-{number:02d}":
                    raise ValueError("stage3_sample_id_round_mismatch")
                case_id = "clean:" + scene_id
                decision_id = "stage3:" + sample_id
                source = "clean"
            elif stage == "stage4":
                case_id = turn.get("round_sample_id")
                if not isinstance(case_id, str) or case_id not in published_cases:
                    raise ValueError("stage4_turn_without_published_case")
                decision_id = f"stage4:{case_id}:round:{number:02d}"
                source = "untrusted"
            else:
                raise ValueError("unknown_decision_source")
            if decision_id in identities or (case_id, number) in case_rounds:
                raise ValueError("duplicate_case_decision:" + decision_id)
            identities.add(decision_id)
            case_rounds.add((case_id, number))
            ref = deepcopy(turn.get("output"))
            if (not isinstance(ref, dict) or
                    set(ref) != {"action", "reasoning", "params"} or
                    ref["action"] not in {"delegate_task", "complete"} or
                    not isinstance(ref["reasoning"], str) or
                    not ref["reasoning"].strip() or
                    not isinstance(ref["params"], dict)):
                raise ValueError("invalid_reference_action:" + decision_id)
            if ref["action"] == "delegate_task":
                if not isinstance(ref["params"].get("tasks"), list) or not ref["params"]["tasks"]:
                    raise ValueError("invalid_reference_delegation:" + decision_id)
            elif not isinstance(ref["params"].get("answer"), str):
                raise ValueError("invalid_reference_completion:" + decision_id)
            state = turn.get("input")
            if not isinstance(state, dict) or set(state) != {"subtask_history"}:
                raise ValueError("invalid_main_visible_history:" + decision_id)
            if not isinstance(state["subtask_history"], str):
                raise ValueError("invalid_main_visible_history_text")
            scene = scenes[scene_id]
            prompt = o1_prompt_from_turn(
                scene, turn,
                max_attempts=settings["max_attempts"],
                sub_model=settings["sub_model"],
                model_alias=settings["model_alias"],
            )
            decisions.append({
                "schema": DECISION_SCHEMA,
                "decision_id": decision_id,
                "case_id": case_id,
                "scene_id": scene_id,
                "source": source,
                "decision_round": number,
                "generation_mode": turn.get("generation_mode", "unknown"),
                "question": scene["question"],
                "visible_state": {"subtask_history": state["subtask_history"]},
                "prompt": [{"role": "user", "content": prompt}],
                "reference_action": ref,
            })
            actions[ref["action"]] += 1
            if progress and (index + 1) % 128 == 0:
                bar.set_postfix(delegate=actions["delegate_task"],
                                complete=actions["complete"], refresh=False)
            # Explicitly private common teacher state, NOT trainer-specific
            # reward metadata and NEVER included in model-visible prompts.
            teacher.append({
                "schema": LABEL_SCHEMA,
                "decision_id": decision_id,
                "scene_id": scene_id,
                "case_id": case_id,
                "ground_truth": _answer_text(scene),
                "reference_action": deepcopy(ref),
                "reward_state": deepcopy(
                    private_states.get((case_id, number))
                ) if stage == "stage4" else None,
            })
        if progress:
            bar.set_postfix(delegate=actions["delegate_task"],
                            complete=actions["complete"], refresh=True)
    return decisions, teacher


def export_dataset(
    *, scenes_path: Path, clean_episodes_path: Path,
    clean_turns_path: Path, stage4_dir: Path, output_dir: Path,
    overwrite: bool = False, strict_conflicts: bool = False,
    progress: bool = False,
) -> dict:
    paths = [Path(p) for p in (
        scenes_path, clean_episodes_path, clean_turns_path,
    )]
    stage4_dir, output_dir = Path(stage4_dir), Path(output_dir)
    if any(not p.is_file() for p in paths):
        raise FileNotFoundError("missing_stage1_or_stage3_export_input")
    if not (stage4_dir / "manifest.json").is_file():
        raise FileNotFoundError("missing_stage4_generation_manifest")
    source_settings = {
        "stage3": settings_for(paths[2], paths[0], "stage3"),
        "stage4": settings_for(stage4_dir / "main_turns.jsonl",
                               paths[0], "stage4"),
    }
    scenes = _index(read_jsonl(paths[0], progress=progress,
                               desc="Read Scenes"), "task_id", "scene")
    clean = _index(read_jsonl(paths[1], progress=progress,
                              desc="Read Stage3 episodes"), "task_id",
                   "stage3_clean_episode")
    for scene_id, episode in clean.items():
        if scene_id not in scenes:
            raise ValueError("stage3_unknown_scene")
        _check_clean(episode, scene_id, scenes[scene_id])
    if not clean:
        raise ValueError("empty_clean_episode_collection")
    cases = [
        _case_for_clean(scenes[scene_id], episode)
        for scene_id, episode in tqdm(sorted(clean.items()), desc="Clean Cases",
                                      unit="case", disable=not progress,
                                      leave=False)
    ]
    branches, status_counts = _stage4_cases(
        stage4_dir, scenes, clean, progress=progress,
    )
    cases.extend(branches)
    accepted = {
        item["case_id"] for item in branches
        if item["recovery"]["globally_recovered"]
    }
    states = {}
    path = stage4_dir / "private" / "decision_reward_states.jsonl"
    if path.is_file():
        for state in read_jsonl(path, progress=progress,
                                desc="Read reward snapshots"):
            key = (state["round_sample_id"], state["decision_round"])
            if key in states:
                raise ValueError("duplicate_stage4_reward_snapshot")
            states[key] = state["state"]
    clean_turns = read_jsonl(paths[2], progress=progress,
                            desc="Read Stage3 turns")
    if any(item.get("task_id") not in clean for item in clean_turns):
        raise ValueError("stage3_turn_without_accepted_clean_case")
    decisions, labels = _build_decisions([
        ("stage3", clean_turns, source_settings["stage3"]),
        ("stage4", read_jsonl(stage4_dir / "main_turns.jsonl",
                              progress=progress, desc="Read Stage4 turns"),
         source_settings["stage4"]),
    ], scenes, accepted, states, progress=progress)
    if not decisions:
        raise ValueError("no_main_decisions")
    quality = audit_dataset(cases, decisions, labels, progress=progress)
    if strict_conflicts and quality["conflicting_prompt_groups"]:
        raise ValueError("conflicting_labels_for_identical_prompt")
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError("dataset_output_exists_use_new_dir_or_explicit_overwrite")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "cases.jsonl", cases, progress=progress)
    _write_jsonl(output_dir / "decisions.jsonl", decisions, progress=progress)
    _write_jsonl(output_dir / "private" / "decision_labels.jsonl",
                 labels, progress=progress)
    # Validate persisted base records; method adapters belong to baseline/*.
    on_disk = audit_output(output_dir, progress=progress)
    if on_disk != quality:
        raise RuntimeError("base_data_quality_round_trip_mismatch")
    _write_json(output_dir / "quality_report.json", quality)
    counts = Counter(row["source"] for row in decisions)
    result = {
        "schema": DATASET_SCHEMA,
        "source_sha256": {
            "scenes": digest(paths[0]),
            "clean_episodes": digest(paths[1]),
            "clean_turns": digest(paths[2]),
            "stage4_manifest": digest(stage4_dir / "manifest.json"),
            "stage4_published_episodes": digest(stage4_dir / "episodes.jsonl"),
            "stage4_main_turns": digest(stage4_dir / "main_turns.jsonl"),
        },
        "base_file_sha256": {
            "cases.jsonl": digest(output_dir / "cases.jsonl"),
            "decisions.jsonl": digest(output_dir / "decisions.jsonl"),
            "private/decision_labels.jsonl": digest(
                output_dir / "private/decision_labels.jsonl"
            ),
        },
        "scene_count": len(clean),
        "case_count": len(cases),
        "decision_count": len(decisions),
        "decision_sources": dict(sorted(counts.items())),
        "stage4_journal_outcomes": dict(sorted(status_counts.items())),
        "accepted_fault_cases": len(accepted),
        "missing_decision_reward_states": sum(
            decisions[i]["source"] == "untrusted" and
            labels[i]["reward_state"] is None
            for i in range(len(decisions))
        ),
        "quality_report": "quality_report.json",
        "label_policy": (
            "Cases and private teacher state are offline supervision only; "
            "Main policy input is decisions[*].prompt. Methods own all "
            "split/filter/Parquet/reward transformations."
        ),
    }
    _write_json(output_dir / "manifest.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("scenes", "clean-episodes", "clean-turns",
                 "stage4-dir", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--strict-conflicts", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-progress", action="store_true",
                        help="Disable tqdm progress bars for batch logs")
    args = parser.parse_args(argv)
    result = export_dataset(
        scenes_path=args.scenes, clean_episodes_path=args.clean_episodes,
        clean_turns_path=args.clean_turns, stage4_dir=args.stage4_dir,
        output_dir=args.output_dir, overwrite=args.overwrite,
        strict_conflicts=args.strict_conflicts,
        progress=not args.no_progress,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
