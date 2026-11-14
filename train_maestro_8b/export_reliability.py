"""Export reviewed Stage-4 histories into weak, Main-visible risk supervision.

The learner sees only native O1 prompt text. Teacher-side claim IDs and source
lineage are used OFFLINE to derive labels; missing parses are masked. No hidden
fault type, Gold answer, or future Sub-Agent message enters the prompt.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import re

from .risk_schema import VERSION, iter_jsonl, validate

BAD_STATES = {"disputed", "intra_report_conflict", "partial"}
BATCH_RE = re.compile(r"(?m)^\[Attempt (\d+)\] 🔄 Parallel Batch \((\d+) subtasks\)")
SUB_RE = re.compile(r"(?m)^  \[(\d+)\] [^\n]*")


def branch_from_sample_id(task_id: str, sample_id: str) -> int:
    match = re.match(re.escape(task_id) + r"-fork-round-(\d+)", sample_id)
    if not match:
        raise ValueError("cannot align private branch sample ID to public episode")
    return int(match.group(1))


def index_states(path):
    """Key states by (scene, exact candidate ID, next Main decision round)."""
    out = defaultdict(dict)
    for row in iter_jsonl(path):
        scene = row["task_id"]
        scenario_id = row["round_sample_id"]
        branch_from_sample_id(scene, scenario_id)  # validates provenance
        rd = row["main_can_act_from_round"]
        key = (scene, scenario_id)
        if rd in out[key]:
            raise ValueError("ambiguous duplicate state round: " + repr((key, rd)))
        out[key][rd] = row
    return out


def history_spans(history: str, expected: list[tuple[int, int]]):
    """Exact report-block offsets in native BaseMainAgent._format_subtask_history.

    Do not parse claims or guess boundaries inside free-form Summary prose.
    """
    batches = list(BATCH_RE.finditer(history))
    if len(batches) != len(expected):
        raise ValueError("O1 history batch count mismatch")
    found = []
    for i, match in enumerate(batches):
        round_id, count = int(match.group(1)), int(match.group(2))
        if (round_id, count) != expected[i]:
            raise ValueError("O1 history batch ordering/size mismatch")
        batch_end = batches[i + 1].start() if i + 1 < len(batches) else history.rfind("\n---\nSummary:")
        if batch_end < match.end():
            raise ValueError("O1 history footer or batch boundary missing")
        subs = list(SUB_RE.finditer(history, match.end(), batch_end))
        if len(subs) != count:
            raise ValueError("O1 history subtask count mismatch")
        for slot, sub in enumerate(subs):
            if int(sub.group(1)) != slot + 1:
                raise ValueError("O1 history subtask ordering mismatch")
            end = subs[slot + 1].start() if slot + 1 < len(subs) else batch_end
            found.append((round_id, slot, sub.start(), end))
    return found


def label_reports(state: dict, snapshots: list[dict], slots: list[tuple[int, int, int, int]]):
    """Risk means an OBSERVABLE warning, not ground-truth incorrectness.

    Clean parsed claims with no visible warning -> 0.
    Visible disagreement, intra-report conflict, partial/missing -> 1.
    Unparseable, superseded or dependency-ambiguous -> MASKED.
    """
    latest = {}
    slot_to_task = {}
    for snap in snapshots:
        for slot, event in enumerate(snap["report_events"]):
            task = event["macro_task_id"]
            latest[task] = (snap["main_can_act_from_round"] - 1, event)
            slot_to_task[(snap["main_can_act_from_round"] - 1, slot)] = task
    active_claims = {c["claim_id"]: c for c in state.get("claim_nodes", [])}
    claim_status = {}
    for value in state.get("claim_states", {}).values():
        for claim_id in value.get("supporting_claim_ids", []):
            claim_status[claim_id] = value["status"]
    direct_risk = {
        c["report_task_id"] for cid, c in active_claims.items()
        if claim_status.get(cid) in BAD_STATES
    }
    # An observed same-subject upstream dependency is not an independent
    # witness. When its parent is disputed, avoid a false negative on child.
    ambiguous_children = set()
    for edge in state.get("claim_dependency_edges", []):
        parent = active_claims.get(edge["parent_claim_id"])
        child = active_claims.get(edge["child_claim_id"])
        if parent and child and parent["report_task_id"] in direct_risk:
            ambiguous_children.add(child["report_task_id"])
    result = []
    for round_id, slot, start, end in slots:
        key = (round_id, slot)
        if key not in slot_to_task:
            raise ValueError("private/public report task mapping unavailable")
        task = slot_to_task[key]
        latest_round, event = latest[task]
        risk, mask, reason = 0, 0, "unknown"
        if latest_round != round_id:
            reason = "superseded"
        elif event["status"] != "done":
            risk, mask, reason = 1, 1, "incomplete_or_missing"
        elif task in direct_risk:
            risk, mask, reason = 1, 1, "visible_conflict"
        elif task in ambiguous_children:
            reason = "dependency_requires_review"
        elif event.get("visible_claim_count", 0) > 0:
            risk, mask, reason = 0, 1, "parsed_no_visible_warning"
        else:
            reason = "unparsed"
        result.append(dict(
            report_id=f"{round_id}:{slot}:{task}", start=start, end=end,
            risk=risk, mask=mask, label_reason=reason,
        ))
    return result


def render_samples(scene: dict, episode: dict, states: dict, max_attempts: int):
    """Use full accumulated reports available BEFORE each Main decision."""
    from environment.base_orchestration.o1_compat import (
        o1_history, o1_prompt_from_history,
    )
    from environment.untrusted_orchestration.state_machine import public_history_item
    actions = [a for a in episode["trajectory"] if a.get("action") == "delegate_task"]
    actions.sort(key=lambda a: a["attempt"])
    history_items = []
    snapshots = []
    for action in actions:
        rd = action["attempt"]
        reports = action["result"]["subtask_results"]
        history_items.append({
            "attempt": rd, "is_parallel_batch": True,
            "subtask_results": [public_history_item(r, i) for i, r in enumerate(reports)],
            "total_steps": sum(r.get("steps_taken", 0) for r in reports),
            "total_cost": 0.0,
        })
        next_round = rd + 1
        state = states.get(next_round)
        if state is None:
            raise ValueError("missing private visible-state snapshot for round")
        snapshots.append(state)
        if next_round > max_attempts:
            continue
        history = o1_history(history_items)
        prompt = o1_prompt_from_history(
            scene, history, attempt=next_round, max_attempts=max_attempts,
            sub_model="qwen3-8b", model_alias="model_1",
        )
        base = prompt.find(history)
        if base < 0 or prompt.find(history, base + 1) != -1:
            raise ValueError("native subtask history not uniquely embedded in prompt")
        slots = history_spans(
            history, [(a["attempt"], len(a["result"]["subtask_results"]))
                      for a in actions if a["attempt"] <= rd],
        )
        reports_out = label_reports(state, snapshots, slots)
        for item in reports_out:
            item["start"] += base
            item["end"] += base
        if not any(r["mask"] for r in reports_out):
            continue
        yield validate(dict(
            schema=VERSION,
            sample_id=f"{episode['round_sample_id']}-decision-{next_round:02d}",
            round_sample_id=episode["round_sample_id"],
            scene_id=scene["task_id"], branch_round=episode["branch_round"],
            decision_round=next_round, prompt=prompt, reports=reports_out,
        ))


def export(scenes_path, episodes_path, states_path, output_dir,
           max_attempts=16, val_fraction=0.1, seed=17):
    if not 0 <= val_fraction < 1:
        raise ValueError("val_fraction must be in [0,1)")
    scene_map = {s["task_id"]: s for s in iter_jsonl(scenes_path)}
    state_map = index_states(states_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    seen = set()
    seen_prompts = {}
    with (out / "train.jsonl").open("w", encoding="utf-8") as train, \
         (out / "val.jsonl").open("w", encoding="utf-8") as val:
        for ep in iter_jsonl(episodes_path):
            scene_id, fork = ep["task_id"], ep["branch_round"]
            scenario_id = ep.get("round_sample_id")
            if scenario_id is None:
                # Legacy Stage4 output predates candidate IDs. Only permit
                # an unambiguous fallback; never mix distinct fault variants.
                candidates = [
                    sid for (s, sid) in state_map
                    if s == scene_id
                    and branch_from_sample_id(scene_id, sid) == fork
                ]
                if len(candidates) != 1:
                    raise ValueError(
                        "legacy Stage4 episode lacks round_sample_id and "
                        "has ambiguous candidate branches; republish Stage4 "
                        "from the existing journal with the updated writer"
                    )
                scenario_id = candidates[0]
                ep["round_sample_id"] = scenario_id
            if branch_from_sample_id(scene_id, scenario_id) != fork:
                raise ValueError("episode branch does not match candidate ID")
            key = (scene_id, scenario_id)
            if key in seen:
                raise ValueError("duplicate accepted scene/candidate in episodes")
            seen.add(key)
            if scene_id not in scene_map or key not in state_map:
                raise ValueError("missing scene or exact candidate supervision")
            bucket = int(sha256(f"{seed}:{scene_id}".encode()).hexdigest()[:12], 16) / 16**12
            writer = val if bucket < val_fraction else train
            for row in render_samples(scene_map[scene_id], ep, state_map[key], max_attempts):
                signature = sha256(row["prompt"].encode("utf-8")).hexdigest()
                supervision = [(r["start"], r["end"], r["risk"], r["mask"])
                               for r in row["reports"]]
                if signature in seen_prompts:
                    if seen_prompts[signature] != supervision:
                        raise ValueError("identical Main prompt has inconsistent risk labels")
                    counts["duplicate_prompts_skipped"] += 1
                    continue
                seen_prompts[signature] = supervision
                writer.write(json.dumps(row, ensure_ascii=False) + "\n")
                counts["total"] += 1
                counts["val" if writer is val else "train"] += 1
                for r in row["reports"]:
                    counts["report_" + r["label_reason"]] += 1
                    if r["mask"]:
                        counts["supervised_reports"] += 1
        counts["episodes"] = len(seen)
    summary = dict(counts)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenes", default="environment/data/stage1/questions/scenes.jsonl")
    p.add_argument("--episodes", default="environment/data/stage4/episodes.jsonl")
    p.add_argument("--states", default="environment/data/stage4/private/claim_state_supervision.jsonl")
    p.add_argument("--output_dir", default="environment/data/stage4/private/window_risk")
    p.add_argument("--max_attempts", type=int, default=16)
    p.add_argument("--val_fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=17)
    a = p.parse_args()
    print(json.dumps(export(a.scenes, a.episodes, a.states, a.output_dir,
                            a.max_attempts, a.val_fraction, a.seed), indent=2))


if __name__ == "__main__":
    main()
