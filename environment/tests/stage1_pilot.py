"""Offline 100-question Stage-1 end-to-end pilot without an LLM endpoint.

The embedded RDF is a *tiny test vocabulary only*. The generator still
composes a new synthetic World. Use --knowledge-yago with the real YAGO
corpus for the full research benchmark; never conflate this CI smoke test
with local Qwen Author/Critic acceptance.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from environment.config import DEFAULT_CONFIG, args_from_config, load_config
from environment.generate_question.pipeline import quality_report
from environment.generate_question.stages import construct_scenes


_FIXTURE_RDF = (
    "@prefix p: <https://example.org/> .\n"
    "@prefix s: <http://schema.org/> .\n"
    "p:person s:birthPlace p:town .\n"
    "p:town s:containedInPlace p:region .\n"
    "p:region s:containedInPlace p:country .\n"
)


def run_pilot(output_dir: Path, *, target: int = 100,
              knowledge_yago: Path | None = None) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    if knowledge_yago is None:
        knowledge_yago = output_dir / "yago-test-motif.ttl"
        knowledge_yago.write_text(_FIXTURE_RDF, encoding="utf-8")
    opts = args_from_config(
        load_config(DEFAULT_CONFIG),
        output_dir=output_dir / "constructed",
        target=target, no_progress=True,
    )
    opts.knowledge_yago = knowledge_yago
    opts.max_records = max(target * 4, target)
    opts.scenario_model = None
    opts.scenario_api_base = None
    opts.question_model = None
    opts.question_api_base = None
    opts.question_critic_model = None
    opts.question_critic_api_base = None
    construction = construct_scenes(opts)
    with (output_dir / "constructed" / "scenes.jsonl").open(
            encoding="utf-8") as src:
        scenes = [json.loads(line) for line in src if line.strip()]
    report = quality_report(scenes, no_progress=True)
    summary = report["knowledge_world_grounding"]["single_task_plan"]
    result = {
        "pilot_kind": (
            "offline_synthetic_motif_ci" if knowledge_yago.name ==
            "yago-test-motif.ttl" else "offline_user_supplied_yago"
        ),
        "llm_author_critic": "not_run",
        "target": target,
        "constructed": construction["generated"],
        "construction_status": construction["status"],
        "strict_replay_pass": report["replay_pass"],
        "strict_replay_fail": report["replay_fail"],
        "mode_distribution": summary["mode_distribution"],
        "question_topic_distribution": summary["question_topic_distribution"],
        "macro_op_distribution": dict(Counter(
            t["operation"] for scene in scenes
            for t in scene["problem_spec"]["macro_plan"]["macro_tasks"]
        )),
        "grounded_evidence": summary["grounded_evidence"],
        "decision_opportunities": summary["macro_decision_opportunities"],
        "questions_leaking_relation_program": sum(
            "documented relationship sequence" in
            scene["question"].lower() for scene in scenes
        ),
        "mean_question_chars": round(
            sum(len(scene["question"]) for scene in scenes) /
            len(scenes), 1
        ) if scenes else 0,
        "answer_shortcut_warning_fields":
            summary["answer_shortcut_warning_fields"],
        "source": str(knowledge_yago),
    }
    (output_dir / "quality_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "pilot_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("STAGE1_100_PILOT_RESULTS=" + json.dumps(result, sort_keys=True))
    # Stage 1 intentionally rejects a minority of candidates that fail
    # strict quality gates; a fixed target is NOT a guarantee of 100%
    # construction. Require >=90% yield AND 100% replay/grounding of all
    # published Scenes. Never weaken per-Scene quality to fill the quota.
    min_accepted = (target * 9 + 9) // 10
    if (result["constructed"] < min_accepted or
            result["strict_replay_pass"] != result["constructed"] or
            result["strict_replay_fail"] != 0 or
            not result["grounded_evidence"]["all_grounding_checks_pass"] or
            result["questions_leaking_relation_program"] or
            result["answer_shortcut_warning_fields"]):
        raise RuntimeError("stage1_100_pilot_quality_gate_failed")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target", type=int, default=100)
    parser.add_argument("--knowledge-yago", type=Path)
    args = parser.parse_args()
    run_pilot(args.output_dir, target=args.target,
              knowledge_yago=args.knowledge_yago)


if __name__ == "__main__":
    main()
