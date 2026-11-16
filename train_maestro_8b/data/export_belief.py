"""Export Stage4 accepted O1 histories for trainable-LoRA evidence-belief pilot.

Reuses export_risk's exact O1 prompt/span/round alignment. Label supervision
is derived ONLY from Main-observable claim/source/dependency snapshots.
Never expose world facts, oracle correctness, attack types or future reports.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

from train_maestro_8b.data.contracts import iter_jsonl
from train_maestro_8b.data.export_risk import (
    branch_from_sample_id, index_states, render_samples,
)

SCHEMA = "maestro.evidence_belief.v1"
CLASSES = ("unverified", "corroborated", "disputed", "partial")
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASSES)}
# The exporter is conservative: a report with multiple incompatible
# statuses is unlabelled rather than silently assigned an arbitrary class.
STATUS_TO_CLASS = {
    "unverified": "unverified",
    "corroborated": "corroborated",
    "disputed": "disputed",
    "intra_report_conflict": "disputed",
    "partial": "partial",
}
UNKNOWN = -100


def validate_belief(row: dict) -> dict:
    if row.get("schema") != SCHEMA:
        raise ValueError("Wrong Stage1 belief schema")
    prompt, reports = row.get("prompt"), row.get("reports")
    if not isinstance(prompt, str) or not prompt or not isinstance(reports, list) or not reports:
        raise ValueError("Belief sample needs an O1 prompt and report spans")
    previous, supervised = 0, 0
    for report in reports:
        start, end = report.get("start"), report.get("end")
        if (type(start) is not int or type(end) is not int
                or not previous <= start < end <= len(prompt)
                or not prompt[start:end].strip()):
            raise ValueError("Invalid/overlapping report span")
        previous = end
        label, update = report.get("belief_label"), report.get("update_label")
        if type(label) is not int or label not in (UNKNOWN, *range(len(CLASSES))):
            raise ValueError("Invalid belief class")
        if type(update) is not int or update not in (UNKNOWN, 0, 1):
            raise ValueError("Invalid belief update")
        if label != UNKNOWN:
            supervised += 1
        if update != UNKNOWN and label == UNKNOWN:
            raise ValueError("Cannot supervise update for unknown belief")
    if not supervised:
        raise ValueError("No supervised belief reports")
    return row


def annotate_beliefs(risk_row: dict, state: dict) -> dict | None:
    """Align CURRENT claim statuses to report spans; old/superseded stay masked."""
    if risk_row["decision_round"] != state.get("main_can_act_from_round"):
        raise ValueError("Belief/claim decision-round mismatch")
    active = {c["claim_id"]: c for c in state.get("claim_nodes", [])}
    task_states: dict[str, set[str]] = {}
    task_keys: dict[str, set[str]] = {}
    for key, claim_state in state.get("claim_states", {}).items():
        label = STATUS_TO_CLASS.get(claim_state.get("status"))
        for claim_id in claim_state.get("supporting_claim_ids", []):
            claim = active.get(claim_id)
            if claim is None:
                raise ValueError("Belief state refers to missing active claim ID")
            task = claim["report_task_id"]
            if label:
                task_states.setdefault(task, set()).add(label)
                task_keys.setdefault(task, set()).add(key)
    transitions = {
        item["claim_key"]
        for item in state.get("transitions", [])
        if item.get("before") not in (None, "missing")
        and item.get("before") != item.get("after")
        and item.get("next_main_decision_round") == risk_row["decision_round"]
    }
    out = []
    for report in risk_row["reports"]:
        item = {k: report[k] for k in ("report_id", "start", "end")}
        task = report["report_id"].split(":", 2)[2]
        reason = report["label_reason"]
        classes = task_states.get(task, set())
        label = UNKNOWN
        if report["mask"] and reason == "incomplete_or_missing":
            label = CLASS_TO_ID["partial"]
        elif report["mask"] and len(classes) == 1:
            label = CLASS_TO_ID[next(iter(classes))]
        # If the report was superseded, unparseable, or context-dependent
        # with no safe label, never force it into the negative class.
        update = UNKNOWN
        round_id = int(report["report_id"].split(":", 2)[0])
        if (label != UNKNOWN and round_id == risk_row["decision_round"] - 1
                and classes and reason != "incomplete_or_missing"):
            update = int(bool(task_keys.get(task, set()) & transitions))
        item.update(belief_label=label, update_label=update)
        out.append(item)
    if not any(r["belief_label"] != UNKNOWN for r in out):
        return None
    return validate_belief({
        "schema": SCHEMA, "sample_id": risk_row["sample_id"],
        "scene_id": risk_row["scene_id"],
        "round_sample_id": risk_row["round_sample_id"],
        "decision_round": risk_row["decision_round"],
        "prompt": risk_row["prompt"], "reports": out,
    })


def export(scenes_path, episodes_path, states_path, output_dir,
           *, max_attempts: int = 16, val_fraction: float = 0.1,
           seed: int = 17) -> dict:
    if not (0 <= val_fraction < 1 and max_attempts >= 2):
        raise ValueError("Invalid Stage1 belief split/max attempts")
    scenes = {row["task_id"]: row for row in iter_jsonl(scenes_path)}
    states = index_states(states_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    stats = Counter()
    fingerprints: dict[str, tuple] = {}
    seen = set()
    with (output / "train.jsonl").open("w", encoding="utf-8") as train, (
            output / "val.jsonl").open("w", encoding="utf-8") as val:
        for episode in iter_jsonl(episodes_path):
            scene_id = episode["task_id"]
            if scene_id not in scenes:
                raise ValueError("Stage1 belief: Scene not found: " + scene_id)
            branch = episode.get("round_sample_id")
            if branch is None:
                candidates = [
                    cid for sid, cid in states
                    if sid == scene_id and
                    branch_from_sample_id(scene_id, cid) == episode["branch_round"]
                ]
                if len(candidates) != 1:
                    raise ValueError("Ambiguous Stage4 candidate ID")
                branch = candidates[0]
                episode["round_sample_id"] = branch
            if branch_from_sample_id(scene_id, branch) != episode["branch_round"]:
                raise ValueError("Stage4 belief branch mismatch")
            ident = (scene_id, branch)
            if ident in seen or ident not in states:
                raise ValueError("Duplicate or missing exact Stage4 candidate")
            seen.add(ident)
            # Split by SCENE, not by branch or decision. A fault variant of
            # the same question must NEVER appear in both train and val.
            fraction = int(sha256(f"{seed}:{scene_id}".encode()).hexdigest()[:12], 16) / 16**12
            destination = val if fraction < val_fraction else train
            split = "val" if destination is val else "train"
            for risk_row in render_samples(scenes[scene_id], episode,
                                           states[ident], max_attempts):
                current = states[ident].get(risk_row["decision_round"])
                if current is None:
                    raise ValueError("Missing exactly aligned belief snapshot")
                belief = annotate_beliefs(risk_row, current)
                if belief is None:
                    stats["all_masked_skipped"] += 1
                    continue
                fingerprint = sha256(belief["prompt"].encode("utf-8")).hexdigest()
                signature = tuple(
                    (r["start"], r["end"], r["belief_label"], r["update_label"])
                    for r in belief["reports"]
                )
                if fingerprint in fingerprints:
                    if fingerprints[fingerprint] != signature:
                        raise ValueError("Identical Main prompt with conflicting belief labels")
                    stats["duplicate_prompts_skipped"] += 1
                    continue
                fingerprints[fingerprint] = signature
                destination.write(json.dumps(belief, ensure_ascii=False) + "\n")
                stats[split] += 1
                for report in belief["reports"]:
                    if report["belief_label"] != UNKNOWN:
                        stats["status_" + CLASSES[report["belief_label"]]] += 1
                    else:
                        stats["unknown_masked"] += 1
                    if report["update_label"] != UNKNOWN:
                        stats["update_supervised"] += 1
                        stats["update_" + str(report["update_label"])] += 1
    stats["episodes"] = len(seen)
    stats["schema"] = SCHEMA
    stats["classes"] = list(CLASSES)
    result = dict(stats)
    (output / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if stats["train"] < 1:
        raise ValueError("No trainable evidence-belief samples after filtering")
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenes", default="environment/data/stage1/questions/scenes.jsonl")
    p.add_argument("--episodes", default="environment/data/stage4/episodes.jsonl")
    p.add_argument("--states", default="environment/data/stage4/private/claim_state_supervision.jsonl")
    p.add_argument("--output_dir", default="environment/data/stage4/private/evidence_belief")
    p.add_argument("--max_attempts", type=int, default=16)
    p.add_argument("--val_fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=17)
    args = p.parse_args(argv)
    print(json.dumps(export(
        args.scenes, args.episodes, args.states, args.output_dir,
        max_attempts=args.max_attempts, val_fraction=args.val_fraction,
        seed=args.seed,
    ), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
