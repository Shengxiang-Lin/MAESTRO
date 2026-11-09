"""Single canonical end-to-end orchestration and read-only quality reporting.

No second generator implementation: run the same tested source-scoped
builders and reviewers, changing only their destinations between stages.
"""
from __future__ import annotations

import copy
import json
import re
from difflib import SequenceMatcher
from collections import Counter
from pathlib import Path

from envir.schema import normalize_label

from .stages import construct_scenes, review_questions, read_jsonl
from .quality import validate_replay
from .macro_contract import macro_graph_profile, macro_question_contract
from .surface_quality import question_language_warnings
from envir.streaming import stable_signature


def _construction_identity(args) -> str:
    return stable_signature({
        "target": args.target,
        "max_records": args.max_records,
        "seed": args.seed,
        "difficulty_mix": args.difficulty_mix,
        "mmqa": str(args.mmqa),
        "scenario_model": args.scenario_model,
        "scenario_api_base": args.scenario_api_base,
        "scenario_critic_model": args.scenario_critic_model,
        "scenario_critic_api_base": args.scenario_critic_api_base,
        "scenario_max_revisions": args.scenario_max_revisions,
    })



def _dump(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                    encoding="utf-8")
    temp.replace(path)


def _abstract_question(question: str, spec: dict) -> str:
    """Conservative surface-family proxy, not a semantic equivalence check."""
    abstract = normalize_label(question)
    for value in (
        spec["scenario"]["topic"],
        spec["scenario"]["user_role"],
        spec["scenario"]["entity_kind"],
        spec["scenario"]["purpose"],
        spec["scenario"]["motivation"],
    ):
        label = normalize_label(value)
        if label:
            abstract = abstract.replace(label, "<slot>")
    # Entity/topic digit suffixes are paraphrase-insensitive placeholders.
    return re.sub(r"\b\d+\b", "#", abstract)


def quality_report(scenes: list[dict], *, no_progress: bool = False) -> dict:
    """Deterministic quality metrics over the same frozen source-scoped data."""
    from tqdm.auto import tqdm

    results, exact_questions, shapes, programs = [], Counter(), Counter(), Counter()
    difficulties, families, patterns = Counter(), Counter(), Counter()
    openers, reference_repetition = Counter(), Counter()
    operational_media_fields = Counter()
    graph_counts, graph_by_pattern, graph_profiles = Counter(), {}, []
    graph_contract_failures = Counter()
    semantic_language_warnings = Counter()
    semantic_warning_examples = []
    modalities, scenario_reviews, question_reviews = Counter(), Counter(), Counter()
    one_source = Counter()
    cf_sensitive = Counter()
    answer_signatures = Counter()
    answer_by_difficulty = {}
    question_failure_stages, question_guard_issues = Counter(), Counter()
    macro_tasks, operator_depth, macro_depth = [], [], []
    issue_counts, examples = Counter(), []
    replay_accepted = replay_rejected = 0
    with tqdm(scenes, desc="Quality & replay", unit="task",
              disable=no_progress) as bar:
        for scene in bar:
            result = validate_replay(scene)
            results.append(result)
            replay_accepted += result.get("status") == "valid"
            replay_rejected += result.get("status") != "valid"
            bar.set_postfix({
                "accepted": replay_accepted,
                "rejected": replay_rejected,
            }, refresh=False)
            spec = scene.get("problem_spec", {})
            text = normalize_label(scene.get("question", ""))
            if text:
                exact_questions[text] += 1
            if spec:
                difficulties[spec["difficulty_target"]] += 1
                families[spec["family"]] += 1
                patterns[spec.get("reasoning_pattern", "approval_legacy")] += 1
                shapes[_abstract_question(scene.get("question", ""), spec)] += 1
                question_text = scene.get("question", "").strip()
                warnings = question_language_warnings(
                    question_text, spec, scene.get("macro_graph")
                )
                semantic_language_warnings.update(warnings)
                if warnings and len(semantic_warning_examples) < 20:
                    semantic_warning_examples.append({
                        "task_id": scene.get("task_id"),
                        "warnings": warnings,
                    })
                head = question_text[:90].casefold()
                openers["starts_as_role"] += int(
                    head.startswith(("as a ", "as an "))
                )
                openers["starts_direct_question"] += int(
                    head.startswith(("which ", "what ", "how ", "could ",
                                     "can ", "please ", "for which "))
                )
                from .question import reference_question
                expected = reference_question(spec)
                proximity = SequenceMatcher(
                    None, question_text.casefold(), expected.casefold()
                ).ratio()
                reference_repetition["identical_reference"] += int(
                    question_text.casefold() == expected.casefold()
                )
                reference_repetition["near_reference_95pct"] += int(
                    proximity >= 0.95
                )
                for role in spec["sources"]:
                    modalities[role["modality"]] += 1
                scenario_reviews[spec.get("scenario_authoring", {}).get(
                    "status", "off")] += 1
            review = scene.get("question_authoring", {})
            question_reviews[review.get("status", "off")] += 1
            for failure in review.get("failures", []):
                question_failure_stages[failure.get("stage", "unknown")] += 1
                question_guard_issues.update(failure.get("errors", []))
            answer_key = json.dumps(scene.get("gold_answer", []),
                                    ensure_ascii=False, sort_keys=True)
            answer_signatures[answer_key] += 1
            level = spec.get("difficulty_target", "unknown")
            answer_by_difficulty.setdefault(level, Counter())[answer_key] += 1
            if "gold_program" in scene:
                nodes = scene["gold_program"]["nodes"]
                key = tuple((node["op"], tuple(node["inputs"]))
                            for node in nodes)
                programs[str(key)] += 1
                if spec.get("version") == "maestro.problem_spec.v4":
                    for node in nodes:
                        if node["op"] == "compare_fields":
                            observed = node["right_field"]
                            if any(x in observed for x in (
                                "clip_", "utterance_", "region_",
                                "visual_", "spoken_"
                            )):
                                operational_media_fields[spec["reasoning_pattern"]] += 1
            if "macro_graph" in scene:
                profile = macro_graph_profile(scene["macro_graph"])
                graph_counts[profile["signature"]] += 1
                graph_by_pattern.setdefault(
                    spec.get("reasoning_pattern", "approval_legacy"),
                    Counter()
                )[profile["signature"]] += 1
                graph_profiles.append(profile)
                try:
                    macro_question_contract(spec, scene["macro_graph"])
                except ValueError as exc:
                    graph_contract_failures[str(exc)] += 1
            audit = scene.get("difficulty_audit", {})
            if audit:
                one_source[spec["difficulty_target"]] += int(
                    audit["one_source_baseline"]["exact_match"]
                )
                cf_sensitive[spec["difficulty_target"]] += int(
                    all(entry["changing_source_changes_answer"]
                        for entry in audit["counterfactual_source_tests"].values())
                )
                macro_tasks.append(scene["macro_graph"]["macro_task_count"])
                macro_depth.append(scene["macro_graph"]["critical_path_depth"])
                operator_depth.append(
                    audit["reasoning"]["necessary_structural_operator_depth"])
            for issue in result["issues"]:
                issue_counts[issue] += 1
            if result["issues"] and len(examples) < 20:
                examples.append({
                    "task_id": scene.get("task_id"),
                    "issues": result["issues"][:10],
                })
    n = len(scenes)
    valid = sum(x["status"] == "valid" for x in results)
    fallback_count = question_reviews.get("fallback", 0)
    publishable = sum(
        result["status"] == "valid"
        and item.get("question_authoring", {}).get("status", "off")
        in {"off", "accepted"}
        for item, result in zip(scenes, results)
    )
    data = {
        "status": (
            "invalid" if not n or valid != n or graph_contract_failures
            else "partial" if fallback_count else "valid"
        ),
        "question_publishable": publishable,
        "question_failed_fallback": fallback_count,
        "pipeline": "maestro_reasoning_first_quality",
        "checked": n,
        "replay_pass": valid,
        "replay_fail": n - valid,
        "issue_counts": dict(issue_counts),
        "first_failures": examples,
        "difficulty_distribution": dict(difficulties),
        "goal_family_distribution": dict(families),
        "semantic_reasoning_pattern_distribution": dict(patterns),
        "source_modality_distribution": dict(modalities),
        "scenario_authoring": dict(scenario_reviews),
        "question_review": dict(question_reviews),
        "question_rejections": {
            "by_stage": dict(question_failure_stages),
            "guard_reasons": dict(question_guard_issues),
            "audit_note": "Rejected candidate previews are stored only "
                          "in private question_authoring records.",
        },
        "answer_diversity": {
            "distinct_exact_answer_sets": len(answer_signatures),
            "largest_identical_answer_set": max(
                answer_signatures.values(), default=0),
            "by_difficulty": {
                level: {
                    "total": sum(counts.values()),
                    "distinct_answer_sets": len(counts),
                    "largest_repeated_answer_set": max(
                        counts.values(), default=0),
                } for level, counts in answer_by_difficulty.items()
            },
            "scope": "exact synthetic answer values; not a semantic diversity proof",
        },
        "question_surface": {
            "semantic_language_warnings": {
                "counts": dict(semantic_language_warnings),
                "first_examples": semantic_warning_examples,
                "scope": (
                    "Conservative wording checks only. These do not prove "
                    "semantic equality or inequality with the Gold Program."
                ),
            },
            "opening_patterns": dict(openers),
            "reference_copying": dict(reference_repetition),
            "unique_normalized_questions": len(exact_questions),
            "duplicate_normalized_questions": sum(
                count - 1 for count in exact_questions.values()
            ),
            "unique_abstract_shapes": len(shapes),
            "top_repeated_abstract_shapes": [{
                "occurrences": count, "shape": shape[:220]
            } for shape, count in shapes.most_common(5)],
            "scope": "heuristic_surface_signatures_not_paraphrase_equivalence",
        },
        "macro_graph_diversity": {
            "distinct_executable_graph_signatures": len(graph_counts),
            "most_common_graphs": [
                {"signature": signature, "count": count}
                for signature, count in graph_counts.most_common(8)
            ],
            "distinct_by_semantic_pattern": {
                name: len(counts) for name, counts in graph_by_pattern.items()
            },
            "maximum_repeated_graph_count": max(
                graph_counts.values(), default=0),
            "task_count_histogram": dict(sorted(
                Counter(x["macro_task_count"] for x in graph_profiles).items()
            )),
            "mean_macro_task_count": (
                round(sum(x["macro_task_count"] for x in graph_profiles)
                      / len(graph_profiles), 3)
                if graph_profiles else None
            ),
            "mean_dependency_edges": (
                round(sum(x["edge_count"] for x in graph_profiles)
                      / len(graph_profiles), 3)
                if graph_profiles else None
            ),
            "graph_to_question_contract_failures": dict(
                graph_contract_failures
            ),
            "scope": (
                "Naming-invariant fingerprint of executable Macro DAGs, "
                "operator chains and source roles; not a semantic uniqueness proof."
            ),
        },
        "reasoning": {
            "media_specific_observation_predicates_used": dict(
                operational_media_fields
            ),
            "semantic_pattern_count": len(patterns),
            "unique_structural_program_signatures": len(programs),
            "top_repeated_structural_programs": [
                {"occurrences": count, "signature": shape[:200]}
                for shape, count in programs.most_common(5)
            ],
            "mean_operator_depth": (
                round(sum(operator_depth) / len(operator_depth), 3)
                if operator_depth else None
            ),
            "mean_macro_task_count": (
                round(sum(macro_tasks) / len(macro_tasks), 3)
                if macro_tasks else None
            ),
            "mean_macro_dependency_depth": (
                round(sum(macro_depth) / len(macro_depth), 3)
                if macro_depth else None
            ),
            "macro_tasks_above_nine": sum(x > 9 for x in macro_tasks),
            "one_source_shortcut_success_by_difficulty": dict(one_source),
            "source_counterfactual_full_pass_by_difficulty": dict(cf_sensitive),
            "one_source_shortcut_policy": (
                "weak register-only baseline, not exhaustive minimal program search"
            ),
        },
        "interpretation_limits": [
            "Replay checks internal synthetic coherence, not real Agent accuracy.",
            "Counterfactuals sample interventions, not every possible shortcut.",
            "One-source baseline is intentionally weak; exact count by difficulty matters.",
            "Question acceptance is independent model judgment, not formal equivalence.",
            "Media roles represent simulated observations, never real perception.",
            "Structural and text signatures diagnose diversity, not sufficient novelty.",
        ],
    }
    return data


def audit_questions(args) -> dict:
    scenes = read_jsonl(args.scenes, limit=getattr(args, "max_records", None))
    result = quality_report(scenes, no_progress=getattr(args, "no_progress", False))
    if getattr(args, "output", None) is not None:
        _dump(args.output, result)
        result["report_path"] = str(args.output)
    return result


def generate_questions(args) -> dict:
    """One command from MMQA/topic seeds to validated Question quality."""
    if bool(args.question_model) != bool(args.question_api_base):
        raise ValueError(
            "For optional Question Author/Critic, supply both "
            "--question-model and --question-api-base or neither"
        )
    if args.question_model and args.question_workers < 1:
        raise ValueError("--question-workers must be positive")
    root = args.output_dir
    report_path = root / "quality_report.json"
    summary_path = root / "summary.json"
    if not args.overwrite and summary_path.exists():
        raise FileExistsError(
            "Unified generation already completed; choose a new directory "
            "or enable overwrite"
        )
    staged = copy.copy(args)
    staged.output_dir = root / "constructed"
    constructed_done = staged.output_dir / "summary.json"
    if constructed_done.is_file() and not args.overwrite:
        constructed_summary = json.loads(
            constructed_done.read_text(encoding="utf-8")
        )
        constructed_bank = staged.output_dir / "scenes.jsonl"
        if not constructed_bank.is_file():
            raise ValueError("constructed_summary_without_scene_bank")
        # Never silently reuse an unrelated frozen bank when a run was
        # interrupted AFTER construction but BEFORE final quality.
        previous_identity = constructed_summary.get("run_identity")
        current_identity = _construction_identity(staged)
        if previous_identity != current_identity:
            raise ValueError(
                "constructed bank belongs to different generation settings; "
                "choose another output directory or enable overwrite"
            )
    else:
        constructed_summary = construct_scenes(staged)
    scene_path = staged.output_dir / "scenes.jsonl"
    question_summary = None
    if args.question_model and constructed_summary["generated"] > 0:
        authored = copy.copy(args)
        authored.scenes = scene_path
        authored.output_dir = root / "questions"
        # Constructed bank is exactly the scene set; do not truncate.
        authored.max_records = None
        question_done = authored.output_dir / "summary.json"
        previous = (
            json.loads(question_done.read_text(encoding="utf-8"))
            if question_done.is_file() and not args.overwrite else {}
        )
        if (previous
                and previous.get("status") in {"done", "partial", "failed"}):
            if previous.get("source_scenes") != str(scene_path):
                raise ValueError("question_summary_source_mismatch")
            if not (authored.output_dir / "scenes.jsonl").exists():
                raise ValueError("question_summary_missing_scene_bank")
            question_summary = previous
        else:
            question_summary = review_questions(authored)
        scene_path = authored.output_dir / "scenes.jsonl"
    # The quality report also performs strict independent replay from the
    # frozen Program/World (not simply accepting stored difficulty flags).
    audit_args = copy.copy(args)
    audit_args.scenes = scene_path
    audit_args.max_records = None
    audit_args.output = report_path
    report = audit_questions(audit_args)
    final_count = report["question_publishable"]
    failed_question_count = (
        question_summary.get("failed", 0) if question_summary else 0
    )
    summary = {
        "status": (
            "done" if (
                final_count == args.target and report["status"] == "valid"
            ) else "failed" if final_count == 0 else "partial"
        ),
        "pipeline": "maestro_generate_question_v1",
        "source": constructed_summary["seed_source"],
        "requested": args.target,
        "attempted": constructed_summary.get("attempted", args.target),
        "generated": final_count,
        "failed": args.target - final_count,
        "failed_scenario": constructed_summary.get("failed", 0),
        "failed_question": failed_question_count,
        "failed_case_files": {
            "scenario": constructed_summary["failed_file"],
            "question": (
                str(root / "questions" / "failed.jsonl")
                if question_summary else None
            ),
        },
        "constructed_scenes": str(root / "constructed" / "scenes.jsonl"),
        "final_scenes": str(scene_path),
        "final_public_questions": str(scene_path.parent / "questions.jsonl"),
        "scenario_authoring": constructed_summary["scenario_authoring"],
        "question_authoring": (
            question_summary["question_results"] if question_summary
            else {"off": constructed_summary["generated"]}
        ),
        "difficulty_distribution": report["difficulty_distribution"],
        "quality": {
            "checked": report["checked"],
            "macro_graph_diversity": report["macro_graph_diversity"],
            "replay_pass": report["replay_pass"],
            "replay_fail": report["replay_fail"],
            "distinct_program_signatures": report["reasoning"][
                "unique_structural_program_signatures"],
            "distinct_question_shapes": report["question_surface"][
                "unique_abstract_shapes"],
        },
        "quality_report": str(report_path),
        "limitations": report["interpretation_limits"],
    }
    _dump(summary_path, summary)
    return summary
