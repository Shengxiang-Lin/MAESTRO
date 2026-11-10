"""Bounded accepted-quota refill, with independently auditable generation rounds."""
from __future__ import annotations
import copy
import json
import math
from collections import Counter
from pathlib import Path

from envir.schema import normalize_label
from envir.streaming import stable_signature, file_digest
from .spec import DIFFICULTIES, difficulty_sequence
from .stages import read_jsonl


def quota_targets(n, mix, seed):
    counts = Counter(difficulty_sequence(target=n, mix=mix, run_seed=seed))
    return {k: counts[k] for k in DIFFICULTIES}


def deficits(targets, counts):
    return {k: max(0, targets[k] - counts[k]) for k in DIFFICULTIES}


def percentage_mix(missing):
    total = sum(missing.values())
    if not total:
        raise ValueError("quota_already_complete")
    numbers = [100 * missing[k] / total for k in DIFFICULTIES]
    base = [int(x) for x in numbers]
    for i in sorted(range(3), key=lambda j: (base[j] - numbers[j], j))[
        :100 - sum(base)
    ]:
        base[i] += 1
    return tuple(base)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    temp.replace(path)


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def generate_with_quotas(args, *, generate_batch, quality_report):
    """Select real accepted scenes, never fabricate answers or difficulty labels."""
    root = args.output_dir
    if (root / "summary.json").exists() and not args.overwrite:
        raise FileExistsError("completed_quota_run_choose_new_output_dir")
    targets = quota_targets(
        args.target, tuple(map(int, args.difficulty_mix.split(","))), args.seed
    )
    multiplier = getattr(args, "max_candidate_multiplier", 3)
    if type(multiplier) is not int or not 1 <= multiplier <= 10:
        raise ValueError("max_candidate_multiplier_must_be_between_1_and_10")
    budget = args.target * multiplier
    selected, seen, rounds = [], set(), []
    failures = {"scenario": [], "question": []}
    totals = {"scenario": Counter(), "question": Counter()}
    attempted = overflow = 0

    for round_index in range(1, 20):
        counts = Counter(x["problem_spec"]["difficulty_target"]
                         for x in selected)
        missing = deficits(targets, counts)
        need = sum(missing.values())
        if need == 0 or attempted >= budget:
            break
        count = min(args.target, budget - attempted,
                    args.target if round_index == 1
                    else max(need, math.ceil(need * 1.45)))
        mix = percentage_mix(missing)
        sub = copy.copy(args)
        sub.output_dir = root / "rounds" / f"{round_index:02d}"
        sub.target = count
        sub.seed = args.seed + 104729 * (round_index - 1)
        sub.difficulty_mix = ",".join(map(str, mix))
        sub.quota_refill = False
        manifest = sub.output_dir / "quota_request.json"
        summary_path = sub.output_dir / "summary.json"
        identity = stable_signature({
            "index": round_index, "count": count, "mix": mix, "seed": sub.seed,
            "models": [sub.scenario_model, sub.scenario_critic_model,
                       sub.question_model, sub.question_critic_model],
            "input": str(sub.mmqa), "input_sha": file_digest(sub.mmqa),
            "code": {name: file_digest(Path(__file__).with_name(name))
                     for name in ("quota_pipeline.py", "scenario.py",
                                  "scenario_fields.py", "question.py",
                                  "spec.py")},
        })
        if manifest.exists() and not args.overwrite:
            if json.loads(manifest.read_text())["signature"] != identity:
                raise ValueError("quota_round_resume_identity_mismatch")
        elif summary_path.exists() and not args.overwrite:
            raise ValueError("completed_quota_round_missing_identity")
        else:
            # Commit the identity *before* the long-running LLM batch, so
            # interruption after its completion cannot strand a checkpoint.
            write_json(manifest, {"signature": identity})
        outcome = (
            json.loads(summary_path.read_text())
            if summary_path.exists() and not args.overwrite
            else generate_batch(sub)
        )
        attempted += outcome["attempted"]
        totals["scenario"].update(outcome.get("scenario_authoring", {}))
        totals["question"].update(outcome.get("question_authoring", {}))
        for kind, path in (
            ("scenario", sub.output_dir / "constructed" / "failed.jsonl"),
            ("question", sub.output_dir / "questions" / "failed.jsonl"),
        ):
            if path.exists():
                failures[kind].extend(read_jsonl(path))
        scenes_path = sub.output_dir / "questions" / "scenes.jsonl"
        if not scenes_path.exists():
            scenes_path = sub.output_dir / "constructed" / "scenes.jsonl"
        rows = read_jsonl(scenes_path) if scenes_path.exists() else []
        before = len(selected)
        for scene in rows:
            level = scene["problem_spec"]["difficulty_target"]
            key = normalize_label(scene["question"])
            if (counts[level] < targets[level] and key and key not in seen
                    and scene.get("question_authoring", {}).get("status")
                    in {"accepted", "off", None}):
                selected.append(scene)
                seen.add(key)
                counts[level] += 1
            else:
                overflow += 1
        rounds.append({"round": round_index, "attempted": outcome["attempted"],
                       "accepted": len(selected) - before,
                       "mix": list(mix), "batch": str(sub.output_dir)})
    final_counts = Counter(x["problem_spec"]["difficulty_target"] for x in selected)
    remaining = deficits(targets, final_counts)
    private = root / "questions"
    constructed = root / "constructed"
    write_jsonl(private / "scenes.jsonl", selected)
    write_jsonl(private / "questions.jsonl", [x["public"] for x in selected])
    write_jsonl(private / "difficulty_audit.jsonl", [
        {"task_id": x["task_id"], "difficulty": x["difficulty_audit"],
         "question_authoring": x.get("question_authoring")} for x in selected
    ])
    write_jsonl(constructed / "scenes.jsonl", selected)
    write_jsonl(constructed / "questions.jsonl", [x["public"] for x in selected])
    write_jsonl(constructed / "failed.jsonl", failures["scenario"])
    write_jsonl(private / "failed.jsonl", failures["question"])
    report = quality_report(selected, no_progress=getattr(args, "no_progress", False))
    write_json(root / "quality_report.json", report)
    summary = {
        "status": ("done" if not sum(remaining.values())
                   and report["status"] == "valid"
                   else "partial" if selected else "failed"),
        "pipeline": "maestro_quota_generation",
        "source": str(args.mmqa), "requested": args.target,
        "attempted": attempted, "generated": len(selected),
        "failed": attempted - len(selected),
        "failed_scenario": len(failures["scenario"]),
        "failed_question": len(failures["question"]),
        "scenario_authoring": dict(totals["scenario"]),
        "question_authoring": dict(totals["question"]),
        "difficulty_distribution": dict(final_counts),
        "difficulty_quotas": {
            "requested": targets, "achieved": {k: final_counts[k] for k in DIFFICULTIES},
            "deficit": remaining, "met": not any(remaining.values()),
            "candidate_budget": budget, "attempted": attempted,
            "discarded_over_quota_or_duplicate": overflow, "rounds": rounds,
        },
        "constructed_scenes": str(constructed / "scenes.jsonl"),
        "final_scenes": str(private / "scenes.jsonl"),
        "final_public_questions": str(private / "questions.jsonl"),
        "failed_case_files": {
            "scenario": str(constructed / "failed.jsonl"),
            "question": str(private / "failed.jsonl"),
        },
        "quality_report": str(root / "quality_report.json"),
        "quality": {
            "checked": report["checked"], "replay_pass": report["replay_pass"],
            "replay_fail": report["replay_fail"],
            "macro_graph_diversity": report["macro_graph_diversity"],
            "distinct_program_signatures": report["reasoning"][
                "unique_structural_program_signatures"],
            "distinct_question_shapes": report["question_surface"][
                "unique_abstract_shapes"],
        },
        "limitations": report["interpretation_limits"],
    }
    write_json(root / "summary.json", summary)
    return summary
