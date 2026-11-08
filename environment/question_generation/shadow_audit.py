"""Run an independent semantic audit over saved Stage 2B drafts."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

from environment.config import parse_with_config
from environment.delegation_blueprint import compile_delegation_outline
from environment.trajectory_simulation.compiler import compile_gold_solution
from environment.trajectory_simulation.eligibility import controlled_question_reference, digest
from environment.task_construction.rendering.pipeline import render_task_record

from .boundary import validate_frozen_inputs
from .goal_review import validate_goal_record
from .pipeline import _backends, _rows, build_parser as generation_parser
from .question_brief import compile_scoped_question_brief
from .realization import _review_attempt
from .scoped_surface import build_scoped_plan, render_scoped
from .semantic_equivalence import InvalidSemanticGraph, canonical_expression
from .shadow_verifier import SCHEMA_VERSION, verify_candidate

_LABELS = {"equivalent", "different", "ambiguous"}


def _legacy_status(feedback):
    if feedback and (feedback["verdict"] == "pass" or (
        feedback["verdict"] == "revise" and not feedback["differences"]
        and feedback["naturalness"] == "needs_polish"
    )):
        return "match"
    if feedback and feedback["verdict"] == "revise" and feedback["differences"]:
        return "mismatch"
    return "unresolved"


def collect_candidates(audits, cases):
    """Select first and final nonempty drafts; deduplicate by task and wording."""
    candidates = {}

    def entry(task_id, question):
        return candidates.setdefault((task_id, question), {
            "task_id": task_id, "question": question, "observations": [],
        })

    for path in audits:
        for row in _rows(path):
            drafts = [item for item in row["audit"]
                      if item.get("stage") == "candidate" and item.get("question")]
            if not drafts:
                continue
            reference = next(item for item in row["audit"] if item.get("stage") == "reference")
            for draft in (drafts[0], drafts[-1]) if len(drafts) > 1 else drafts:
                feedback = draft.get("critic_feedback")
                entry(row["task_id"], draft["question"])["observations"].append({
                    "audit": str(path),
                    "attempt": draft["attempt"],
                    "reference_sha256": reference["scoped_question_sha256"],
                    "accepted": bool(draft.get("accepted")),
                    "failure": draft.get("failure"),
                    "legacy_status": _legacy_status(feedback),
                    "legacy_naturalness": feedback.get("naturalness") if feedback else None,
                })
    for path in cases:
        for row in _rows(path):
            label = row["label"]
            if label not in _LABELS or not row.get("question") or not row.get("category"):
                raise ValueError(f"Invalid labeled candidate in {path}")
            candidate = entry(row["task_id"], row["question"])
            if "label" in candidate:
                raise ValueError(f"Duplicate label for {row['task_id']} in {path}")
            candidate.update(label=label, category=row["category"])
    if not candidates:
        raise ValueError("No candidates. Supply at least one --audit or --cases file.")
    return sorted(candidates.values(), key=lambda row: "label" not in row)


def _gold_inputs(settings, blueprints, selected):
    validate_frozen_inputs(
        settings.manifest, settings.authoring_inputs, settings.capabilities,
        worlds_path=settings.worlds,
    )
    payloads = {row["task_id"]: row for row in _rows(settings.authoring_inputs)
                if row["task_id"] in selected}
    caps = {row["task_id"]: row for row in _rows(settings.capabilities)
            if row["task_id"] in selected}
    records = {}
    for row in _rows(blueprints):
        task_id = row["task_id"]
        if task_id in selected:
            if task_id in records:
                raise ValueError(f"Duplicate certified blueprint for {task_id}")
            records[task_id] = row
    if selected - (payloads.keys() & caps.keys() & records.keys()):
        raise ValueError("Each candidate needs matching frozen inputs and a certified blueprint.")

    found = {}
    with Path(settings.worlds).open(encoding="utf-8") as handle:
        for i, line in enumerate(handle):
            world = json.loads(line)
            rendered = render_task_record(world, seed=settings.seed + i)
            key = rendered.task_id
            if key not in selected:
                continue
            canonical = rendered.question
            payload = payloads[key]
            if canonical != payload["draft_question"]:
                raise ValueError(f"Frozen renderer mismatch for {key}")
            gold = compile_gold_solution({
                "task_id": key, "question": canonical, "capability_world": caps[key],
            }, world, rendered, schedule="compact")
            reference = controlled_question_reference(gold, world, rendered)
            if not reference or reference.get("question") != canonical:
                raise ValueError(f"Unverified reference for {key}")
            plan = build_scoped_plan(world, canonical, gold, reference)
            outline = compile_delegation_outline(
                gold, world, canonical, plan, payload.get("resources", ()),
            )
            validate_goal_record(
                records[key], payload, world, canonical, outline, oracle_answer=rendered.answer,
            )
            blueprint = records[key]["delegation_blueprint"]
            brief = compile_scoped_question_brief(blueprint, plan)
            graph = {field: brief[field] for field in ("answer_scope", "scopes")}
            found[key] = (graph, render_scoped(plan, [0] * len(plan)), blueprint)
            if found.keys() == selected:
                break
    if found.keys() != selected:
        raise ValueError(f"Missing rendered worlds: {sorted(selected - found.keys())}")
    return found


def _paired_outcome(label, status):
    if label == "ambiguous":
        return "ambiguous_" + status
    if status == "unresolved":
        return "unresolved_" + label
    if label == "equivalent":
        return "correct_match" if status == "match" else "false_mismatch"
    return "false_match" if status == "match" else "correct_mismatch"


def run(args, settings):
    candidates = collect_candidates(args.audit, args.cases)
    gold_inputs = _gold_inputs(settings, args.blueprints, {row["task_id"] for row in candidates})
    for row in candidates:
        graph, reference, _ = gold_inputs[row["task_id"]]
        if any(item["reference_sha256"] != digest(reference) for item in row["observations"]):
            raise ValueError(f"Historical reference changed for {row['task_id']}")
    _, backend = _backends(settings)

    def review(row):
        graph, reference, blueprint = gold_inputs[row["task_id"]]
        try:
            canonical_expression(graph)
        except (InvalidSemanticGraph, ArithmeticError, ValueError) as exc:
            shadow = {"status": "unresolved", "reason": "unsupported_gold_graph",
                      "detail": str(exc)}
        else:
            shadow = verify_candidate(backend, row["question"], graph)
        known = {item["legacy_status"] for item in row["observations"]} - {"unresolved"}
        legacy = next(iter(known)) if len(known) == 1 else "unresolved"
        feedback = None
        if args.recheck_legacy and not known:
            feedback = _review_attempt(
                backend, reference, row["question"], {"critic_calls": 0}, blueprint=blueprint,
            )
            legacy = _legacy_status(feedback)
        return {**row, "shadow": shadow, "legacy_status": legacy, "legacy_recheck": feedback}

    by_status, by_label, by_category = Counter(), Counter(), {}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for row in pool.map(review, candidates):
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                by_status[row["shadow"]["status"]] += 1
                if "label" not in row:
                    continue
                for arm, status in (
                    ("legacy", row["legacy_status"]), ("shadow", row["shadow"]["status"]),
                ):
                    outcome = _paired_outcome(row["label"], status)
                    by_label[f"{arm}/{outcome}"] += 1
                    by_category.setdefault(row["category"], Counter())[f"{arm}/{outcome}"] += 1
    return {
        "schema": SCHEMA_VERSION,
        "mode": "offline_diagnostic",
        "task_count": len(gold_inputs),
        "candidate_count": len(candidates),
        "labeled_count": sum("label" in row for row in candidates),
        "shadow_statuses": dict(by_status),
        "labeled_outcomes": dict(by_label),
        "labeled_by_category": {key: dict(counts) for key, counts in by_category.items()},
        "model": settings._resolved_models["question_critic"],
        "output": str(args.output),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", action="append", type=Path, default=[],
                        help="Historical question_review_audit.jsonl; repeat for multiple arms.")
    parser.add_argument("--cases", action="append", type=Path, default=[],
                        help="JSONL with task_id, question, label and category.")
    parser.add_argument("--blueprints", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="New diagnostic JSONL.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--recheck-legacy", action="store_true",
                        help="Call the existing Critic for candidates without a recorded decision.")
    args = parse_with_config(parser, "stage2_shadow")
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.output.exists():
        parser.error("--output must name a new file")
    settings = parse_with_config(generation_parser(), "question_generation", ["--config", args.config])
    print(json.dumps(run(args, settings), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
