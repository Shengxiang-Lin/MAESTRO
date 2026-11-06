"""Export a read-only Stage 2/Stage 3 paired question audit bundle.

No LLM calls or changes to existing training data. The output is teacher-
private audit evidence, NOT a proof of semantic equivalence.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path

def records(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]

def find_realized(explicit, ids, n):
    paths = ([Path(explicit)] if explicit else sorted(Path("environment/data").glob(
        "questions_*/realized_capability_worlds.jsonl")))
    matches = []
    for path in paths:
        if not path.is_file():
            continue
        sample = records(path)[:n]
        if len(sample) == n and {str(x.get("task_id")) for x in sample} == ids:
            matches.append((path, sample))
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one matching Stage 2 realized file; found {len(matches)}. "
            "Provide --realized explicitly or verify the v7 input files."
        )
    return matches[0]

def find_programs(world_file, ids, seed):
    from environment.task_construction.rendering.pipeline import render_task_record
    found = {}
    with Path(world_file).open(encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            if not line.strip():
                continue
            world = json.loads(line)
            rendered = render_task_record(world, seed=seed + index)
            tid = str(rendered.task_id)
            if tid in ids:
                if tid in found:
                    raise ValueError(f"Duplicate world task_id: {tid}")
                found[tid] = {
                    "programir": world["program"],
                    "world_row_index": index,
                    "renderer_question": rendered.question,
                }
            if len(found) == len(ids):
                break
    missing = ids - set(found)
    if missing:
        raise ValueError(
            f"Could not match world task IDs {sorted(missing)[:5]}. "
            "Check --worlds and --seed."
        )
    return found

def export(args):
    bank_file = Path(args.bank)
    quarantine_file = Path(args.quarantine or (str(bank_file) + ".quarantine.jsonl"))
    accepted = {}
    for entry in records(bank_file):
        bank = entry["bank"]
        if bank["task_id"] in accepted:
            raise ValueError("Duplicate accepted task")
        accepted[bank["task_id"]] = bank
    quarantined = {}
    for entry in records(quarantine_file):
        tid = entry["task_id"]
        if tid in quarantined:
            raise ValueError("Duplicate quarantined task")
        quarantined[tid] = entry
    ids = set(accepted) | set(quarantined)
    if set(accepted) & set(quarantined) or len(ids) != args.max_records:
        raise ValueError(
            "Bank and quarantine task IDs must be disjoint and cover "
            f"exactly {args.max_records} tasks"
        )
    realized_file, stage2_rows = find_realized(args.realized, ids, args.max_records)
    if args.worlds:
        worlds_file = Path(args.worlds)
    else:
        folder = realized_file.parent.name
        if not folder.startswith("questions_"):
            raise ValueError("Use --worlds for nonstandard Stage 2 folders")
        worlds_file = Path("environment/data") / (
            "typed_worlds_" + folder[len("questions_"):] + ".jsonl"
        )
    programs = find_programs(worlds_file, ids, args.seed)
    audit_file = realized_file.parent / "question_critic_audit.jsonl"
    critic_audits = {r["task_id"]: r for r in records(audit_file)}
    if not ids.issubset(critic_audits):
        raise ValueError("Stage 2 critic audit is missing task IDs")
    output = Path(args.output)
    protected = [bank_file, quarantine_file, realized_file, worlds_file, audit_file]
    if output.resolve() in {p.resolve() for p in protected}:
        raise ValueError("Never overwrite an input with the audit output")
    output.parent.mkdir(parents=True, exist_ok=True)
    tally = Counter()
    with output.open("w", encoding="utf-8") as handle:
        for row in stage2_rows:
            tid = row["task_id"]
            attempts = row.get("authoring", {}).get("audit", [])
            final = row["question"]
            eligibility = (
                accepted[tid]["eligibility"] if tid in accepted else
                quarantined[tid].get("audit", {}).get("eligibility", {})
            )
            result = {
                "task_id": tid,
                "draft_question": row.get("draft_question"),
                "realized_question": final,
                "same_as_draft": final == row.get("draft_question"),
                "renderer_question": programs[tid]["renderer_question"],
                "same_as_renderer": final == programs[tid]["renderer_question"],
                "programir": programs[tid]["programir"],
                "world_row_index": programs[tid]["world_row_index"],
                "stage2_author_critic_attempts": attempts,
                "stage2_ordered_critic_audit": critic_audits[tid],
                "stage3": {
                    "status": "accepted" if tid in accepted else "quarantined",
                    "failure_stage": (
                        quarantined[tid]["reason"] if tid in quarantined else None
                    ),
                    "failure_issues": (
                        quarantined[tid].get("issues", []) if tid in quarantined else []
                    ),
                    "eligibility": eligibility,
                },
            }
            tally[result["stage3"]["status"]] += 1
            tally["same_as_draft"] += result["same_as_draft"]
            tally["same_as_renderer"] += result["same_as_renderer"]
            tally["author_attempt_differs"] += any(
                x.get("author_response", {}).get("question") not in (None, final)
                for x in attempts if isinstance(x, dict)
            )
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
    return {
        "audited": len(stage2_rows), **tally,
        "realized": str(realized_file), "worlds": str(worlds_file),
        "output": str(output),
        "warning": "Teacher-private evidence; not a model prompt or semantic proof",
    }

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", required=True)
    parser.add_argument("--quarantine")
    parser.add_argument("--realized")
    parser.add_argument("--worlds")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-records", type=int, default=50)
    parser.add_argument("--output", default="work/stage2_paired_audit_v7_50.jsonl")
    print(json.dumps(export(parser.parse_args()), ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
