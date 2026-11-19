"""Unified MAESTRO paired Case and Main-decision dataset export.

Private case labels and gold/reward state NEVER enter the O1 policy prompt.
Stage3 clean and Stage4 fault variants share one Scene-level split.
No LLM calls are made by this exporter.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import shutil

from dataset.prepare_da_grpo import (
    digest, make_examples, read_jsonl, settings_for,
)
from .splits import load_splits, make_splits, verify_splits

CASE_SCHEMA = "maestro.paired_case.v1"
DECISION_SCHEMA = "maestro.main_decision.v1"
LABEL_SCHEMA = "maestro.decision_teacher.v1"
DATASET_SCHEMA = "maestro.dataset.v1"
SPLITS = ("train", "val", "test")


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


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for item in rows:
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


def _case_for_clean(scene: dict, episode: dict, split: str):
    return {
        "schema": CASE_SCHEMA, "case_id": "clean:" + scene["task_id"],
        "scene_id": scene["task_id"], "split": split,
        "question": scene["question"],
        "clean_reference_id": episode["clean_reference_id"],
        "kind": "clean",
        "clean": deepcopy(episode),
        "untrusted": None,
        "fault": {"atoms": [], "applied_events": []},
        "recovery": {
            "publication_status": "accepted",
            "execution_outcome": "recovered",
            "globally_recovered": True,
            "observable_issues": [],
            "main_decisions": [],
            "invalidations": [],
            "revalidations": [],
        },
    }


def _case_for_branch(
    scene: dict, clean: dict, entry: dict, journal: dict, *,
    split: str, published: bool,
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
        "split": split,
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


def _stage4_cases(
    stage4_dir: Path, scenes: dict, clean: dict, splits: dict,
):
    index_path = stage4_dir / "round_index.jsonl"
    if not index_path.is_file():
        raise FileNotFoundError("stage4_round_index_missing:" + str(index_path))
    published = _index(
        read_jsonl(stage4_dir / "episodes.jsonl"),
        "round_sample_id", "stage4_published_episode",
    )
    cases, seen = [], set()
    status_counts = Counter()
    for number, item in enumerate(read_jsonl(index_path)):
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
            split=splits[scene_id], published=case_id in published,
        )
        if case is not None:
            cases.append(case)
            status_counts["accepted" if case_id in published
                          else "diagnostic"] += 1
        else:
            status_counts["no_episode"] += 1
    if not set(published) <= {
        c["case_id"] for c in cases if c["recovery"]["globally_recovered"]
    }:
        raise ValueError("unmatched_stage4_published_episodes")
    return cases, status_counts


def _build_decisions(
    sources: list[tuple[str, list[dict], dict]],
    scenes: dict, splits: dict, published_cases: set,
    states: dict,
    *, require_llm: bool,
):
    rows, private_rows, grpo_rows, sft_rows = [], [], [], []
    seen: set[str] = set()
    for stage, turns, settings in sources:
        if stage == "stage3":
            examples, _, _ = make_examples(
                scenes, turns, source=stage, settings=settings,
                seen=seen, require_llm=require_llm,
            )
            index = _index(turns, "sample_id", "stage3_main_turn")
            prepared = [
                (example, index[example["extra_info"]["sample_id"]])
                for example in examples
            ]
        else:
            # The original Stage4 display sample_id is NOT unique across
            # multiple one-fault variants of the same fork/decision round.
            # round_sample_id is the explicit and immutable candidate key.
            from train_maestro_8b.data.export_da_grpo import convert
            index = {}
            for turn in turns:
                key = (turn.get("round_sample_id"), turn.get("round"))
                if (not isinstance(key[0], str) or type(key[1]) is not int
                        or key in index):
                    raise ValueError("duplicate_or_missing_stage4_decision_key")
                index[key] = turn
            if turns:
                examples, _, _ = convert(
                    list(scenes.values()), turns, settings,
                    reviewed_only=require_llm,
                )
            else:
                examples = []
            prepared = []
            for example in examples:
                key = (
                    example["extra_info"]["sample_id"].rsplit(
                        "-decision-round-", 1
                    )[0],
                    example["extra_info"]["step_index"],
                )
                if key not in index:
                    raise ValueError("stage4_candidate_round_join_mismatch")
                prepared.append((example, index[key]))
        for example, turn in prepared:
            ident = example["extra_info"]["sample_id"]
            scene_id = turn["task_id"]
            split = splits[scene_id]
            if stage == "stage4":
                case_id = turn.get("round_sample_id")
                if case_id not in published_cases:
                    raise ValueError("stage4_turn_without_published_case")
            else:
                case_id = "clean:" + scene_id
            ref = deepcopy(turn["output"])
            if (not isinstance(ref, dict)
                    or set(ref) != {"action", "reasoning", "params"}):
                raise ValueError("incomplete_reference_main_action:" + ident)
            decision = {
                "schema": DECISION_SCHEMA,
                "decision_id": stage + ":" + ident,
                "case_id": case_id,
                "scene_id": scene_id,
                "split": split,
                "source": "clean" if stage == "stage3" else "untrusted",
                "decision_round": turn["round"],
                "generation_mode": turn.get("generation_mode"),
                "prompt": deepcopy(example["prompt"]),
                "reference_action": ref,
            }
            rows.append(decision)
            reward = deepcopy(example)
            reward["extra_info"] = {
                **reward["extra_info"],
                "case_id": case_id, "decision_id": decision["decision_id"],
                "split": split,
            }
            grpo_rows.append((split, reward))
            sft_rows.append((split, {
                "prompt": deepcopy(example["prompt"]),
                "completion": [{
                    "role": "assistant",
                    "content": json.dumps(ref, ensure_ascii=False),
                }],
            }))
            private_rows.append({
                "schema": LABEL_SCHEMA,
                "decision_id": decision["decision_id"],
                "scene_id": scene_id, "case_id": case_id,
                "split": split,
                "reference_action": ref,
                "reward_state": deepcopy(
                    states.get((case_id, turn["round"]))
                ) if stage == "stage4" else None,
            })
    return rows, private_rows, grpo_rows, sft_rows


def export_dataset(
    *, scenes_path: Path, clean_episodes_path: Path,
    clean_turns_path: Path, stage4_dir: Path, output_dir: Path,
    seed: int = 17, ratios=(0.8, 0.1, 0.1),
    split_map_path: Path | None = None,
    require_llm: bool = True, parquet: bool = False,
    overwrite: bool = False,
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
    scenes_list = read_jsonl(paths[0])
    scenes = _index(scenes_list, "task_id", "scene")
    clean_list = read_jsonl(paths[1])
    clean = _index(clean_list, "task_id", "stage3_clean_episode")
    for scene_id, episode in clean.items():
        if scene_id not in scenes:
            raise ValueError("stage3_unknown_scene")
        _check_clean(episode, scene_id, scenes[scene_id])
    if not clean:
        raise ValueError("empty_clean_episode_collection")
    if split_map_path is None:
        split_data = make_splits(clean, seed=seed, ratios=ratios)
    else:
        split_data = load_splits(split_map_path, clean)
    splits = verify_splits(split_data, clean)
    cases = [
        _case_for_clean(scenes[scene_id], episode, splits[scene_id])
        for scene_id, episode in sorted(clean.items())
    ]
    branches, counts = _stage4_cases(
        stage4_dir, scenes, clean, splits,
    )
    cases.extend(branches)
    accepted_cases = {
        item["case_id"] for item in branches
        if item["recovery"]["globally_recovered"]
    }
    private_states = {}
    states_file = stage4_dir / "private" / "decision_reward_states.jsonl"
    if states_file.is_file():
        for state in read_jsonl(states_file):
            key = (state["round_sample_id"], state["decision_round"])
            if key in private_states:
                raise ValueError("duplicate_stage4_decision_reward_state")
            private_states[key] = state["state"]
    decisions, labels, grpo, sft = _build_decisions(
        [
            ("stage3", read_jsonl(paths[2]), source_settings["stage3"]),
            ("stage4", read_jsonl(stage4_dir / "main_turns.jsonl"),
             source_settings["stage4"]),
        ], scenes, splits, accepted_cases, private_states,
        require_llm=require_llm,
    )
    if not decisions:
        raise ValueError("no_valid_main_training_decisions")
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                "dataset_output_exists_use_new_dir_or_explicit_overwrite"
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "splits.json", split_data)
    for name in SPLITS:
        _write_jsonl(
            output_dir / "cases" / (name + ".jsonl"),
            (row for row in cases if row["split"] == name),
        )
        _write_jsonl(
            output_dir / "decisions" / (name + ".jsonl"),
            (row for row in decisions if row["split"] == name),
        )
        _write_jsonl(
            output_dir / "private" / "decision_labels" /
            (name + ".jsonl"),
            (row for row in labels if row["split"] == name),
        )
        _write_jsonl(
            output_dir / "adapters" / "da-grpo" / (name + ".jsonl"),
            (row for split, row in grpo if split == name),
        )
        _write_jsonl(
            output_dir / "adapters" / "da-sft" / (name + ".jsonl"),
            (row for split, row in sft if split == name),
        )
        if parquet:
            try:
                import pyarrow as pa
                import pyarrow.parquet as pq
            except ImportError as exc:
                raise RuntimeError(
                    "Parquet adapter requires pyarrow"
                ) from exc
            for adapter, payload in (("da-grpo", grpo), ("da-sft", sft)):
                selected = [row for split, row in payload if split == name]
                if selected:
                    path = (output_dir / "adapters" / adapter /
                            (("eval" if name == "val" and adapter == "da-sft"
                              else name) + ".parquet"))
                    pq.write_table(pa.Table.from_pylist(selected), path)
    distribution = Counter(row["split"] for row in decisions)
    case_distribution = Counter(row["split"] for row in cases)
    source_distribution = Counter(row["source"] for row in decisions)
    result = {
        "schema": DATASET_SCHEMA,
        "source_sha256": {
            "scenes": digest(paths[0]),
            "clean_episodes": digest(paths[1]),
            "clean_turns": digest(paths[2]),
            "stage4_manifest": digest(stage4_dir / "manifest.json"),
            "stage4_published_episodes": digest(
                stage4_dir / "episodes.jsonl"
            ),
            "stage4_main_turns": digest(
                stage4_dir / "main_turns.jsonl"
            ),
        },
        "splits_sha256": _sha(split_data),
        "scene_counts": split_data["counts"],
        "case_counts": dict(sorted(case_distribution.items())),
        "decision_counts": dict(sorted(distribution.items())),
        "decision_sources": dict(sorted(source_distribution.items())),
        "stage4_journal_outcomes": dict(sorted(counts.items())),
        "accepted_fault_cases": len(accepted_cases),
        "require_llm": require_llm,
        "parquet": parquet,
        "label_policy": (
            "Private fault and recovery state only in paired Cases and "
            "private/decision_labels; O1 prompts rebuilt exactly from "
            "public turn history. Never concatenate labels into prompts."
        ),
    }
    _write_json(output_dir / "manifest.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("scenes", "clean-episodes", "clean-turns",
                "stage4-dir", "output-dir"):
        parser.add_argument("--" + arg, type=Path, required=True)
    parser.add_argument("--split-map", type=Path)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--allow-template", action="store_true")
    parser.add_argument("--parquet", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    ratio = (args.train_ratio, args.val_ratio,
             1 - args.train_ratio - args.val_ratio)
    result = export_dataset(
        scenes_path=args.scenes,
        clean_episodes_path=args.clean_episodes,
        clean_turns_path=args.clean_turns,
        stage4_dir=args.stage4_dir,
        output_dir=args.output_dir, ratios=ratio,
        split_map_path=args.split_map,
        seed=args.seed, require_llm=not args.allow_template,
        parquet=args.parquet, overwrite=args.overwrite,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
