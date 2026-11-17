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

from environment.schema import normalize_label

from environment.generate_question.stages import (
    construct_scenes, review_questions, read_jsonl,
    _construction_identity, _question_signature,
)
from environment.generate_question.auditing.quality import validate_replay
from environment.generate_question.auditing.macro_contract import macro_graph_profile, macro_question_contract
from environment.generate_question.authoring.surface_quality import question_language_warnings



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
    semantic_evidence_checked = 0
    semantic_evidence_enforced = 0
    semantic_evidence_flagged = 0
    semantic_evidence_flags = Counter()
    semantic_evidence_examples = []
    modalities, scenario_reviews, question_reviews = Counter(), Counter(), Counter()
    one_source = Counter()
    cf_sensitive = Counter()
    acquisition_scenes = 0
    graph_modes = Counter()
    macro_first_modes = Counter()
    macro_v2_modes = Counter()
    macro_v2_resolution_modes = Counter()
    macro_v2_goals = Counter()
    numeric_goal_answer_counts = {
        "comparison": Counter(), "calculation": Counter(),
    }
    numeric_goal_versioned = 0
    question_semantic_lock_versioned = 0
    macro_v2_evidence_sizes = []
    macro_v2_local_source_counts = []
    macro_v2_media_pairs = Counter()
    macro_v2_information_audits = []
    macro_v2_output_contracts = 0
    macro_v2_difficulty_profiles = {"medium": [], "hard": []}
    macro_v2_realism_scenes = 0
    typed_grammar_scenes = 0
    typed_grammar_variants = Counter()
    typed_grammar_local_sequences = Counter()
    typed_grammar_equivalence_checked = 0
    typed_grammar_equivalence_passed = 0
    macro_first_handoffs = 0
    macro_first_handoff_passes = 0
    macro_first_local_ops = []
    graph_lookup_total = 0
    graph_counterfactual_pass = 0
    graph_parallel_widths = []
    keyed_lookup_total = 0
    # Task Plan stores counterfactuals in macro_first_audit, not in the
    # older evidence_acquisition block. Count its actual Gold operations
    # directly, including the source access contract for each lookup.
    task_plan_keyed_lookup_total = 0
    task_plan_keyed_lookup_scenes = 0
    task_plan_clue_keyed_lookup_total = 0
    task_plan_clue_keyed_lookup_scenes = 0
    task_plan_keyed_source_mismatches = 0
    v2_keyed_lookup_total = 0
    v2_keyed_lookup_scenes = 0
    acquisition_depths = []
    clue_counterfactual_pass = 0
    full_read_bypass_count = 0
    full_read_attempt_count = 0
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
                if spec.get("reasoning_pattern") == "macro_first_v2":
                    from environment.generate_question.authoring.semantic_evidence import (
                        compile_semantic_contract, semantic_text_issues,
                        semantic_scenario_issues, has_semantic_contract,
                    )
                    semantic_evidence_checked += 1
                    semantic_evidence_enforced += int(has_semantic_contract(spec))
                    contract = compile_semantic_contract(
                        spec, retrospective=True,
                    )
                    flags = sorted(set(
                        semantic_text_issues(
                            scene.get("question", ""), spec, contract,
                        ) + semantic_scenario_issues(spec, contract)
                    ))
                    semantic_evidence_flagged += int(bool(flags))
                    semantic_evidence_flags.update(flags)
                    if flags and len(semantic_evidence_examples) < 20:
                        semantic_evidence_examples.append({
                            "task_id": scene.get("task_id"),
                            "flags": flags,
                            "enforcement_versioned": has_semantic_contract(spec),
                        })
                    plan = spec["macro_plan"]
                    question_semantic_lock_versioned += int(
                        bool(spec.get("question_semantic_lock_version"))
                    )
                    if spec.get("numeric_goal_world_version"):
                        numeric_goal_versioned += 1
                        kind = plan["goal_resolution"]["kind"]
                        field = plan["goal"]["answer_field"]
                        if (kind in numeric_goal_answer_counts
                                and len(scene.get("gold_answer", [])) == 1):
                            value = scene["gold_answer"][0][field]
                            numeric_goal_answer_counts[kind][str(value)] += 1
                    if spec.get("operator_grammar_version"):
                        typed_grammar_scenes += 1
                        evidence_audit = scene.get(
                            "difficulty_audit", {}
                        ).get("macro_first_audit", {}).get(
                            "operator_grammar_equivalence", {}
                        )
                        typed_grammar_equivalence_checked += int(
                            bool(evidence_audit)
                        )
                        typed_grammar_equivalence_passed += int(
                            evidence_audit.get(
                                "all_macro_outputs_equivalent", False
                            ) and evidence_audit.get(
                                "final_answer_and_provenance_equivalent", False
                            )
                        )
                        for role, group in zip(
                            plan["macro_tasks"],
                            scene["gold_program"]["macro_solver_groups"],
                        ):
                            variant = role["solver_variant"]
                            typed_grammar_variants[
                                role["role"] + "::" + variant
                            ] += 1
                            nodes_by_id = {
                                node["id"]: node
                                for node in scene["gold_program"]["nodes"]
                            }
                            sequence = tuple(
                                nodes_by_id[node_id]["op"]
                                for node_id in group["node_ids"]
                            )
                            typed_grammar_local_sequences[
                                role["role"] + "::" + ">".join(sequence)
                            ] += 1
                    if spec.get("scene_realism_version"):
                        macro_v2_realism_scenes += 1
                        proxy = scene.get("difficulty_audit", {}).get(
                            "macro_first_audit", {}
                        ).get("structural_difficulty_profile")
                        if proxy is not None:
                            macro_v2_difficulty_profiles[
                                spec["difficulty_target"]
                            ].append(proxy)
                    if spec.get("macro_information_necessity_version"):
                        macro_v2_output_contracts += 1
                        necessity = scene.get("difficulty_audit", {}).get(
                            "macro_first_audit", {}
                        ).get("candidate_information_necessity")
                        if necessity:
                            macro_v2_information_audits.append(necessity)
                    macro_v2_modes[plan["pattern"]] += 1
                    macro_v2_resolution_modes[
                        plan.get("goal_resolution", {}).get(
                            "kind", "direct_lookup"
                        )
                    ] += 1
                    n_lookups = sum(
                        node["op"] == "lookup_by_key"
                        for node in scene["gold_program"]["nodes"]
                    )
                    v2_keyed_lookup_total += n_lookups
                    v2_keyed_lookup_scenes += int(n_lookups > 0)
                    macro_v2_goals[plan["goal"]["answer_field"]] += 1
                    for macro in plan["macro_tasks"]:
                        macro_v2_local_source_counts.append(len(macro["source_scope"]))
                        macro_v2_evidence_sizes.extend(
                            macro["evidence_plan"]["source_observation_targets"].values()
                        )
                    if plan["pattern"] == "multimodal_event":
                        macro_v2_media_pairs["video_audio_in_one_macro"] += 1
                if spec.get("reasoning_pattern") == "macro_first":
                    macro_first_modes[spec["macro_plan"]["pattern"]] += 1
                if spec.get("reasoning_pattern") == "evidence_graph":
                    graph_modes[spec["evidence_graph"]["archetype"]] += 1
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
                from environment.generate_question.authoring.question import reference_question
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
                if spec.get("reasoning_pattern") == "task_plan":
                    keyed = [
                        node for node in nodes
                        if node["op"] == "lookup_by_key"
                    ]
                    task_plan_keyed_lookup_total += len(keyed)
                    task_plan_keyed_lookup_scenes += int(bool(keyed))
                    clue_sources = {
                        sid
                        for task in spec["macro_plan"]["macro_tasks"]
                        if task["operation"] == "match_clues"
                        for sid in task["source_scope"]
                    }
                    clue_keyed = sum(
                        node["source_id"] in clue_sources for node in keyed
                    )
                    task_plan_clue_keyed_lookup_total += clue_keyed
                    task_plan_clue_keyed_lookup_scenes += int(clue_keyed > 0)
                    declared = {
                        source["source_id"]: source
                        for source in spec["sources"]
                    }
                    task_plan_keyed_source_mismatches += sum(
                        declared.get(node["source_id"], {}).get(
                            "access_mode"
                        ) != "keyed_lookup"
                        or declared[node["source_id"]].get("key_field")
                           != node.get("source_key_field")
                        for node in keyed
                    )
                if spec.get("reasoning_pattern") == "task_plan":
                    # Count *executed-source* annotation predicates rather
                    # than only legacy compare_fields primitive names.
                    # This measures synthetic evidence affordances, not
                    # real binary image/audio/video perception.
                    scopes = {
                        sid for macro in spec["macro_plan"]["macro_tasks"]
                        for sid in macro["source_scope"]
                    }
                    for sid, role in declared.items():
                        if sid not in scopes or role["modality"] not in {
                            "image", "audio", "video"
                        }:
                            continue
                        for observation in scene["world"]["sources"][sid]["observations"]:
                            semantic = observation.get("observation_semantics", {})
                            predicate = semantic.get("predicate")
                            if predicate:
                                operational_media_fields[predicate] += 1
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
                macro_audit = audit.get("macro_first_audit", {})
                if macro_audit:
                    handoffs = macro_audit["handoff_counterfactuals"]
                    macro_first_handoffs += handoffs["edge_count"]
                    macro_first_handoff_passes += handoffs["sensitive_edges"]
                    macro_first_local_ops.extend(
                        macro_audit["primitive_ops_inside_macros"])
                graph_audit = audit.get("evidence_graph_audit", {})
                if graph_audit:
                    graph_lookup_total += graph_audit["keyed_lookup_count"]
                    graph_counterfactual_pass += int(
                        graph_audit["source_counterfactuals_pass"])
                    graph_parallel_widths.append(
                        graph_audit["effective_parallel_wave_width"])
                acquisition = audit.get("evidence_acquisition", {})
                if acquisition:
                    acquisition_scenes += 1
                    keyed_lookup_total += acquisition["keyed_lookup_count"]
                    acquisition_depths.append(
                        acquisition["evidence_acquisition_depth"]
                    )
                    clue_counterfactual_pass += int(
                        acquisition["clue_counterfactual_pass"]
                    )
                    denied = acquisition["full_read_blocked"]
                    full_read_attempt_count += len(denied)
                    full_read_bypass_count += sum(
                        not status for status in denied.values()
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
        "semantic_evidence_diagnostics": {
            "v2_scenes_checked": semantic_evidence_checked,
            "versioned_contract_enforced_scenes": semantic_evidence_enforced,
            "scenes_with_flags": semantic_evidence_flagged,
            "flag_counts": dict(semantic_evidence_flags),
            "first_examples": semantic_evidence_examples,
            "scope": (
                "High-confidence lexical and schema/predicate diagnostics. "
                "Older unversioned V2 scenes are assessed read-only and "
                "remain replay compatible; flags are not an NL equivalence proof."
            ),
        },
        "macro_first_v2": {
            "scenes": sum(macro_v2_modes.values()),
            "typed_operator_grammar": {
                "versioned_scenes": typed_grammar_scenes,
                "witness_equivalence_checked": typed_grammar_equivalence_checked,
                "witness_equivalence_passed": typed_grammar_equivalence_passed,
                "variant_histogram": dict(sorted(typed_grammar_variants.items())),
                "distinct_role_operator_sequences": len(
                    typed_grammar_local_sequences
                ),
                "role_operator_sequence_histogram": dict(sorted(
                    typed_grammar_local_sequences.items()
                )),
                "interpretation": (
                    "Type-safe equivalent relational execution plans; "
                    "structural variation is not by itself new evidence "
                    "or increased reasoning hardness."
                ),
            },
            "numeric_goal_diversity": {
                "versioned_numeric_scenes": numeric_goal_versioned,
                "versioned_question_semantic_lock_scenes":
                    question_semantic_lock_versioned,
                "histogram_by_goal": {
                    kind: dict(sorted(values.items()))
                    for kind, values in numeric_goal_answer_counts.items()
                },
                "distinct_answers_by_goal": {
                    kind: len(values)
                    for kind, values in numeric_goal_answer_counts.items()
                },
                "max_answer_frequency_share_by_goal": {
                    kind: (
                        round(max(values.values()) / sum(values.values()), 4)
                        if values else None
                    ) for kind, values in numeric_goal_answer_counts.items()
                },
                "interpretation": (
                    "Distribution over successfully published scenes; "
                    "a seeded sampler does not guarantee a perfectly "
                    "balanced accepted subset."
                ),
            },
            "pattern_distribution": dict(macro_v2_modes),
            "goal_resolution_distribution": dict(macro_v2_resolution_modes),
            "goal_field_distribution": dict(macro_v2_goals),
            "source_observation_count_histogram": dict(
                sorted(Counter(macro_v2_evidence_sizes).items())
            ),
            "source_count_per_macro_histogram": dict(
                sorted(Counter(macro_v2_local_source_counts).items())
            ),
            "video_audio_joint_macros": macro_v2_media_pairs[
                "video_audio_in_one_macro"
            ],
            "versioned_output_contract_scenes": macro_v2_output_contracts,
            "schema_locked_scene_count": macro_v2_realism_scenes,
            "keyed_lookup_count": v2_keyed_lookup_total,
            "scenes_with_keyed_lookup": v2_keyed_lookup_scenes,
            "structural_difficulty_by_target": {
                label: {
                    "scenes": len(items),
                    "mean_macro_tasks": (
                        round(sum(p["macro_task_count"] for p in items)
                              / len(items), 3) if items else None
                    ),
                    "mean_source_observations": (
                        round(sum(p["source_observation_total"] for p in items)
                              / len(items), 3) if items else None
                    ),
                    "mean_keyed_hops": (
                        round(sum(p["keyed_index_hops"] for p in items)
                              / len(items), 3) if items else None
                    ),
                    "mean_primitive_ops": (
                        round(sum(p["primitive_ops_total"] for p in items)
                              / len(items), 3) if items else None
                    ),
                    "mean_multimodal_alternatives": (
                        round(sum(p.get("alternative_candidates", 0)
                                  for p in items
                                  if "alternative_candidates" in p)
                              / sum("alternative_candidates" in p for p in items),
                              3)
                        if any("alternative_candidates" in p for p in items)
                        else None
                    ),
                } for label, items in macro_v2_difficulty_profiles.items()
            },
            "information_necessity": {
                "audited_multimodal_scenes": len(macro_v2_information_audits),
                "candidate_discrimination_passed": sum(
                    r["passed"] for r in macro_v2_information_audits
                ),
                "mean_left_candidates": (
                    round(sum(r["left_candidate_count"]
                              for r in macro_v2_information_audits)
                          / len(macro_v2_information_audits), 3)
                    if macro_v2_information_audits else None
                ),
                "mean_right_candidates": (
                    round(sum(r["right_candidate_count"]
                              for r in macro_v2_information_audits)
                          / len(macro_v2_information_audits), 3)
                    if macro_v2_information_audits else None
                ),
                "scope": "Actual candidate-set discrimination, not an optimal search lower bound",
            },
            "scope": "Synthetic heterogeneous observations, no real media decoding or open-web tools.",
        },
        "macro_first": {
            "scenes": sum(macro_first_modes.values()),
            "pattern_distribution": dict(macro_first_modes),
            "cross_macro_handoff_count": macro_first_handoffs,
            "sensitive_cross_macro_handoff_count": macro_first_handoff_passes,
            "handoff_dependency_pass_rate": (
                round(macro_first_handoff_passes / macro_first_handoffs, 4)
                if macro_first_handoffs else None
            ),
            "mean_primitive_ops_per_macro": (
                round(sum(macro_first_local_ops) / len(macro_first_local_ops), 3)
                if macro_first_local_ops else None
            ),
            "scope": "Frozen Macro DAGs with deterministic local Gold Solvers, not autonomous subagent runs.",
        },
        "evidence_graph": {
            "scenes": sum(graph_modes.values()),
            "archetype_distribution": dict(graph_modes),
            "keyed_lookup_count": graph_lookup_total,
            "source_counterfactual_full_pass_rate": (
                round(graph_counterfactual_pass / sum(graph_modes.values()), 4)
                if graph_modes else None
            ),
            "mean_max_parallel_wave_width": (
                round(sum(graph_parallel_widths) / len(graph_parallel_widths), 3)
                if graph_parallel_widths else None
            ),
            "scope": "Frozen synthetic typed graphs, not independently solved agent tasks.",
        },
        "evidence_acquisition": {
            "scenes_with_keyed_clue_chain": (
                acquisition_scenes + task_plan_clue_keyed_lookup_scenes
            ),
            "scenes_with_keyed_acquisition": (
                acquisition_scenes + v2_keyed_lookup_scenes
                + task_plan_keyed_lookup_scenes
            ),
            "keyed_lookup_count": (
                keyed_lookup_total + v2_keyed_lookup_total
                + task_plan_keyed_lookup_total
            ),
            "clue_chain_keyed_lookup_count": (
                keyed_lookup_total + task_plan_clue_keyed_lookup_total
            ),
            "task_plan_keyed_lookup_count": task_plan_keyed_lookup_total,
            "task_plan_scenes_with_keyed_lookup": task_plan_keyed_lookup_scenes,
            "task_plan_clue_keyed_lookup_count": (
                task_plan_clue_keyed_lookup_total
            ),
            "task_plan_scenes_with_keyed_clue_chain": (
                task_plan_clue_keyed_lookup_scenes
            ),
            "task_plan_keyed_source_contract_mismatches": (
                task_plan_keyed_source_mismatches
            ),
            "v2_keyed_lookup_count": v2_keyed_lookup_total,
            "v2_scenes_with_keyed_lookup": v2_keyed_lookup_scenes,
            "mean_evidence_acquisition_depth": (
                round(sum(acquisition_depths) / len(acquisition_depths), 3)
                if acquisition_depths else None
            ),
            "clue_counterfactual_pass_rate": (
                round(clue_counterfactual_pass / acquisition_scenes, 4)
                if acquisition_scenes else None
            ),
            "unauthorized_full_read_success_rate": (
                round(full_read_bypass_count / full_read_attempt_count, 4)
                if full_read_attempt_count else None
            ),
            "scope": (
                "Verified synthetic key-dependent information access; "
                "not open-web navigation or real-image perception."
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
    # Evidence audit is private teacher data; these are actual source-pair
    # relations and source-ablation state changes, not Graph signatures.
    ledgers = [item["evidence_ledger"] for item in scenes
               if "evidence_ledger" in item]
    evidence_counts = Counter()
    for ledger in ledgers:
        evidence_counts.update(ledger["relation_counts"])
    data["evidence_relations"] = {
        "versioned_scenes": len(ledgers),
        "scenes_with_comparable_pairs": sum(
            x["independent_comparable_pairs"] > 0 for x in ledgers
        ),
        "independent_comparable_pairs": sum(
            x["independent_comparable_pairs"] for x in ledgers
        ),
        "verification_sensitive_pairs": sum(
            x["verification_sensitive_pairs"] for x in ledgers
        ),
        "relation_counts": dict(evidence_counts),
        "reference_wave_trust": {
            "versioned_scenes": sum(
                "reference_wave_trust" in x for x in ledgers
            ),
            "scheduling_modes": dict(Counter(
                x["reference_wave_trust"]["mode"] for x in ledgers
                if "reference_wave_trust" in x
            )),
            "cross_wave_claim_pairs": sum(
                x["reference_wave_trust"]["cross_wave_claim_pairs"]
                for x in ledgers if "reference_wave_trust" in x
            ),
            "answer_relevant_claim_pairs": sum(
                x["reference_wave_trust"]["answer_relevant_claim_pairs"]
                for x in ledgers if "reference_wave_trust" in x
            ),
            "answer_relevant_cross_wave_pairs": sum(
                x["reference_wave_trust"]["answer_relevant_cross_wave_pairs"]
                for x in ledgers if "reference_wave_trust" in x
            ),
            "trust_state_transitions": sum(
                len(x["reference_wave_trust"]["transitions"])
                for x in ledgers if "reference_wave_trust" in x
            ),
            "scope": (
                "Potential visibility from completed reference Macro waves;"
                " distinct from actual freeform Main-visible report text."
            ),
        },
        "scope": (
            "Exact source-grounded comparable Claim pairs. Warrant changes "
            "are source-ablation counterfactuals, not chronological Main actions; "
            "version-supersession cases require actual versioned records."
        ),
    }
    knowledge_scenes = [
        scene for scene in scenes
        if scene.get("problem_spec", {}).get("reasoning_pattern")
        in {"typed_knowledge_v1", "typed_knowledge_v2", "typed_knowledge_v3", "typed_knowledge_v4", "task_graph_first_v1", "task_graph_first_v2", "task_plan"}
    ]
    if knowledge_scenes:
        role_counts = Counter(
            task["role"]
            for scene in knowledge_scenes
            for task in scene["problem_spec"]["macro_plan"]["macro_tasks"]
        )
        hop_counts = Counter(
            scene["problem_spec"]["macro_plan"].get(
                "hops", len(scene["problem_spec"]["macro_plan"].get("real_facts", []))
            )
            for scene in knowledge_scenes
        )
        macro_counts = Counter(
            len(scene["problem_spec"]["macro_plan"]["macro_tasks"])
            for scene in knowledge_scenes
        )
        sources_pass = sum(
            all(row["changing_source_changes_answer"]
                for row in scene.get("difficulty_audit", {}).get(
                    "counterfactual_source_tests", {}).values())
            for scene in knowledge_scenes
        )
        v2_scenes = [
            scene for scene in knowledge_scenes
            if scene["problem_spec"]["reasoning_pattern"] == "typed_knowledge_v2"
        ]
        necessity = [
            scene.get("difficulty_audit", {}).get("macro_first_audit", {}).get(
                "candidate_information_necessity", {}
            )
            for scene in v2_scenes
        ]
        v3_scenes = [
            scene for scene in knowledge_scenes
            if scene["problem_spec"]["reasoning_pattern"] == "typed_knowledge_v3"
        ]
        v3_audits = [
            scene.get("difficulty_audit", {}).get("macro_first_audit", {}).get(
                "candidate_information_necessity", {}
            ) for scene in v3_scenes
        ]
        v4_scenes = [
            scene for scene in knowledge_scenes
            if scene["problem_spec"]["reasoning_pattern"] == "typed_knowledge_v4"
        ]
        v4_audits = [
            scene.get("difficulty_audit", {}).get(
                "macro_first_audit", {}).get(
                "candidate_information_necessity", {}
            ) for scene in v4_scenes
        ]
        graph_first_scenes = [
            scene for scene in knowledge_scenes
            if scene["problem_spec"]["reasoning_pattern"] == "task_graph_first_v1"
        ]
        graph_first_audits = [
            scene.get("difficulty_audit", {}).get("macro_first_audit", {}).get(
                "candidate_information_necessity", {}
            ) for scene in graph_first_scenes
        ]
        graph_v2_scenes = [
            scene for scene in knowledge_scenes
            if scene["problem_spec"]["reasoning_pattern"] == "task_graph_first_v2"
        ]
        graph_v2_audits = [
            scene.get("difficulty_audit", {}).get("macro_first_audit", {}).get(
                "candidate_information_necessity", {}
            ) for scene in graph_v2_scenes
        ]
        current_scenes = [
            scene for scene in knowledge_scenes
            if scene["problem_spec"]["reasoning_pattern"] == "task_plan"
        ]
        current_audits = [
            scene.get("difficulty_audit", {}).get("macro_first_audit", {}).get(
                "candidate_information_necessity", {}
            ) for scene in current_scenes
        ]
        sizes = [
            len(scene["problem_spec"]["macro_plan"]["macro_tasks"])
            for scene in current_scenes
        ]
        answer_value_counts = {}
        for scene in current_scenes:
            goal_spec = scene["problem_spec"]["macro_plan"].get(
                "goal_spec", {}
            )
            output = goal_spec.get("output_spec", {})
            if not output:
                continue
            answer = scene["gold_answer"][0]["answer"]
            for name, semantic in output["bindings"].items():
                value = answer[name] if isinstance(answer, dict) else answer
                encoded = json.dumps(value, sort_keys=True, ensure_ascii=False)
                answer_value_counts.setdefault(semantic, Counter())[encoded] += 1
        from environment.generate_question.auditing.quality_metrics import summarize_decision_opportunities
        from environment.generate_question.auditing.quality_metrics import summarize_grounded_evidence
        decision_opportunities = summarize_decision_opportunities(current_scenes)
        grounded_evidence = summarize_grounded_evidence(current_scenes)
        answer_diversity = {
            semantic: {
                "count": sum(freq.values()),
                "unique_values": len(freq),
                "most_common_count": freq.most_common(1)[0][1],
                "most_common_share": round(
                    freq.most_common(1)[0][1]/sum(freq.values()), 4
                ),
                "potential_answer_shortcut": (
                    semantic not in {"conclusion", "higher_source"}
                    and sum(freq.values()) >= 5 and (
                        len(freq) < 4 or
                        freq.most_common(1)[0][1]/sum(freq.values()) > 0.45
                    )
                ),
            }
            for semantic, freq in sorted(answer_value_counts.items())
        }
        data["knowledge_world_grounding"] = {
            "single_task_plan": {
                "scenes": len(current_scenes),
                "necessity_passed": sum(
                    row.get("passed", False) for row in current_audits
                ),
                "mode_distribution": dict(Counter(
                    s["problem_spec"]["macro_plan"]["mode"]
                    for s in current_scenes
                )),
                "goal_profile_distribution": dict(Counter(
                    x["problem_spec"]["macro_plan"].get(
                        "goal_spec", {}).get("source_profile", "unknown")
                    for x in current_scenes
                )),
                "goal_output_distribution": dict(Counter(
                    x["problem_spec"]["macro_plan"]["goal_spec"][
                        "output_spec"]["kind"]
                    for x in current_scenes
                    if "output_spec" in x["problem_spec"]["macro_plan"].get(
                        "goal_spec", {}
                    )
                )),
                "goal_operation_distribution": dict(Counter(
                    node["op"]
                    for x in current_scenes
                    for node in x["problem_spec"]["macro_plan"].get(
                        "goal_spec", {}).get("operations", [])
                )),
                "goal_operation_output_pairs": dict(Counter(
                    ",".join(
                        node["op"]
                        for node in x["problem_spec"]["macro_plan"][
                            "goal_spec"]["operations"]
                        if node["op"] != "lookup"
                    ) + " -> " + x["problem_spec"]["macro_plan"][
                        "goal_spec"]["output_spec"]["kind"]
                    for x in current_scenes
                    if "output_spec" in x["problem_spec"]["macro_plan"].get(
                        "goal_spec", {}
                    )
                )),
                "semantic_output_fields": dict(Counter(
                    semantic
                    for x in current_scenes
                    for semantic in x["problem_spec"]["macro_plan"].get(
                        "goal_spec", {}).get("output_spec", {}).get(
                        "bindings", {}).values()
                )),
                "capability_combinations": dict(Counter(
                    "+".join(x["problem_spec"]["macro_plan"].get(
                        "goal_spec", {}).get("capabilities", [])) or "entity_only"
                    for x in current_scenes
                )),
                "answer_value_diversity": answer_diversity,
                "macro_decision_opportunities": decision_opportunities,
                "grounded_evidence": grounded_evidence,
                "answer_shortcut_warning_fields": sorted(
                    field for field, stats in answer_diversity.items()
                    if stats["potential_answer_shortcut"]
                ),
                "goal_answer_type_distribution": dict(Counter(
                    x["problem_spec"]["macro_plan"]["goal"]["answer_type"]
                    for x in current_scenes
                )),
                "macro_task_count_range": [
                    min(sizes, default=0), max(sizes, default=0)
                ],
                "mean_macro_tasks": (
                    round(sum(sizes)/len(sizes), 3) if sizes else 0
                ),
                "event_fields_causal": sum(
                    row.get("event_field_necessary") is True
                    for row in current_audits
                ),
                "real_fork_causal": sum(
                    row.get("real_fork_necessary") is True
                    for row in current_audits
                ),
                "non_target_field_substitution_passed": sum(
                    row.get("non_target_field_substitution", {}).get(
                        "passed", False
                    ) for row in current_audits
                ),
                "eligibility_cases": sum(
                    s["problem_spec"]["macro_plan"].get("condition")
                    is not None for s in current_scenes
                ),
                "calibrated_difficulty_profiles": {
                    level: {
                        "scenes": len(items),
                        "mean_macro_tasks": round(sum(
                            i["macro_graph"]["macro_task_count"]
                            for i in items
                        ) / len(items), 3) if items else 0,
                        "mean_dependency_depth": round(sum(
                            i["macro_graph"]["critical_path_depth"]
                            for i in items
                        ) / len(items), 3) if items else 0,
                        "mean_real_yago_hops": round(sum(
                            i["problem_spec"]["macro_plan"]["hops"]
                            for i in items
                        ) / len(items), 3) if items else 0,
                    }
                    for level in ("medium", "hard")
                    for items in [[
                        s for s in current_scenes
                        if s["problem_spec"]["difficulty_target"] == level
                    ]]
                },
                "question_topic_distribution": dict(Counter(
                    s["problem_spec"]["scenario"]["topic"]
                    for s in current_scenes
                )),
                "semantic_grounding": {
                    "scenes": sum(
                        s["problem_spec"]["macro_plan"].get(
                            "semantic_grounding_policy"
                        ) == "goal_constrained_yago_grounding_v1"
                        for s in current_scenes
                    ),
                    "goal_profile_distribution": dict(Counter(
                        s["problem_spec"]["macro_plan"]["goal"][
                            "semantic"]["profile"]
                        for s in current_scenes
                        if s["problem_spec"]["macro_plan"].get(
                            "semantic_grounding_policy"
                        ) == "goal_constrained_yago_grounding_v1"
                    )),
                    "domain_distribution": dict(Counter(
                        s["problem_spec"]["macro_plan"]["goal"][
                            "semantic"]["domain"]
                        for s in current_scenes
                        if s["problem_spec"]["macro_plan"].get(
                            "semantic_grounding_policy"
                        ) == "goal_constrained_yago_grounding_v1"
                    )),
                    "fictional_worlds": sum(
                        s["problem_spec"]["macro_plan"]["world_binding"][
                            "candidate_paths"
                        ][s["problem_spec"]["macro_plan"]["world_binding"][
                            "target_case"]][0]["origin"] == "synthetic"
                        for s in current_scenes
                    ),
                    "composed_motif_worlds": sum(
                        s["problem_spec"]["macro_plan"].get(
                            "knowledge_world_mode"
                        ) == "compose" for s in current_scenes
                    ),
                    "shared_context_worlds": sum(
                        "context_version" in s["problem_spec"]["macro_plan"][
                            "world_binding"
                        ].get("world_model", {})
                        for s in current_scenes
                    ),
                    "required_shared_relation_worlds": sum(
                        bool(s["problem_spec"]["macro_plan"][
                            "world_binding"
                        ].get("world_model", {}).get(
                            "required_context_relation"
                        )) for s in current_scenes
                    ),
                    "macro_goal_questions": sum(
                        s["problem_spec"]["macro_plan"].get(
                            "question_surface_policy"
                        ) == "macro_goal_reference_card_v2"
                        for s in current_scenes
                    ),
                    "reference_cards_in_public_view": sum(
                        s["public"].get("task_input") ==
                        s["problem_spec"].get("task_input") and
                        "task_input" in s["public"]
                        for s in current_scenes
                    ),
                    "natural_reference_cards": sum(
                        s["public"].get("task_input", {}).get("version")
                        == "macro_goal_reference_card_v2"
                        for s in current_scenes
                    ),
                    "mean_question_characters": (
                        round(sum(len(s["question"]) for s in current_scenes)
                              / len(current_scenes), 1)
                        if current_scenes else None
                    ),
                    "required_shared_relation_necessity_passed": sum(
                        s["difficulty_audit"]["macro_first_audit"][
                            "candidate_information_necessity"
                        ].get("required_shared_relation_necessary") is True
                        for s in current_scenes
                    ),
                    "keyed_event_worlds": sum(
                        bool(s["problem_spec"]["macro_plan"][
                            "world_binding"
                        ].get("world_model", {}).get(
                            "keyed_event_relation_version"
                        )) for s in current_scenes
                    ),
                    "keyed_event_wrong_key_audit_passed": sum(
                        s["difficulty_audit"]["macro_first_audit"][
                            "candidate_information_necessity"
                        ].get("keyed_event_dependency", {}).get(
                            "passed", False
                        ) for s in current_scenes
                    ),
                    "shared_context_entity_groups": sum(
                        s["problem_spec"]["macro_plan"]["world_binding"].get(
                            "world_model", {}
                        ).get("profile", {}).get("shared_context_group_count", 0)
                        for s in current_scenes
                    ),
                    "context_relations": sum(
                        len(s["problem_spec"]["macro_plan"][
                            "world_binding"
                        ].get("world_model", {}).get("context_relations", []))
                        for s in current_scenes
                    ),
                    "source_records_with_context": sum(
                        "document_context" in obs
                        for s in current_scenes
                        for source in s["world"]["sources"].values()
                        for obs in source["observations"]
                    ),
                    "excluded_yago_nonacademic_field_labels": max((
                        s["problem_spec"]["macro_plan"]["world_binding"].get(
                            "world_model", {}
                        ).get("profile", {}).get(
                            "rejected_nonacademic_field_labels", 0
                        )
                        for s in current_scenes
                    ), default=0),
                    "observed_motif_predicate_coverage": dict(Counter(
                        key for scene in current_scenes
                        for key, count in scene["problem_spec"]["macro_plan"][
                            "world_binding"].get("world_model", {}).get(
                                "profile", {}).get(
                                    "observed_predicate_counts", {}
                                ).items()
                        if count
                    )),
                    "loaded_graph_uniqueness_verified": sum(
                        s["problem_spec"]["macro_plan"][
                            "world_binding"].get(
                                "semantic_audit", {}
                            ).get("loaded_yago_singleton_witness", False)
                        for s in current_scenes
                    ),
                    "sampling_attempts": sum(
                        s["problem_spec"]["macro_plan"][
                            "world_binding"].get(
                                "semantic_audit", {}
                            ).get("attempts", 0)
                        for s in current_scenes
                    ),
                    "scope": (
                        "In compose mode, observed YAGO relation types and "
                        "type-filtered vocabulary are recombined into novel "
                        "Worlds. New scenes use one existing discovery source "
                        "to expose a required shared relation; event-mode scenes "
                        "add a canonical project-event-subject relation and "
                        "upstream-keyed event lookup, without full-ledger scans. "
                        "Other context edges remain non-required. "
                        "In witnessed mode, YAGO uniqueness is checked only "
                        "against the bounded loaded fact sample."
                    ),
                },
                "nested_legacy_specs": sum(
                    any("backbone" in k for k in s["problem_spec"])
                    for s in current_scenes
                ),
                "note": (
                    "One Task Plan grounds either a witnessed YAGO path or a "
                    "new typed World composed from YAGO relation motifs. "
                    "No historical generator Specs are nested."
                ),
            },
            "task_graph_first_v2": {
                "scenes_checked": len(graph_v2_scenes),
                "information_necessity_passed": sum(
                    a.get("passed", False) for a in graph_v2_audits
                ),
                "local_event_relation_distribution": dict(Counter(
                    s["problem_spec"]["task_first_v2_relation_sketch"][
                        "event_goal"] for s in graph_v2_scenes
                )),
                "abstract_macro_shape_distribution": dict(Counter(
                    s["problem_spec"]["task_first_macro_graph"]["shape"]
                    for s in graph_v2_scenes
                )),
                "unique_local_relation_signatures": len({
                    tuple((
                        s["problem_spec"]["task_first_macro_graph"]["shape"],
                        s["problem_spec"]["task_first_v2_relation_sketch"][
                            "event_relation"]["operator"],
                        s["problem_spec"]["task_first_v2_relation_sketch"][
                            "event_relation"]["field"],
                    )) for s in graph_v2_scenes
                }),
                "status_or_year_field_necessary": sum(
                    bool(a.get("local_relation_field_necessary"))
                    and all(a["local_relation_field_necessary"].values())
                    for a in graph_v2_audits
                ),
                "event_keyed_cases": sum(
                    a.get("event_handoff_alternative_plan", {}).get(
                        "required", False)
                    for a in graph_v2_audits
                ),
                "event_keyed_bypass_blocked": sum(
                    a.get("event_handoff_alternative_plan", {}).get(
                        "required", False)
                    and a.get("event_handoff_alternative_plan", {}).get(
                        "bypass_introduces_real_alternatives", False
                    ) and a.get("event_handoff_alternative_plan", {}).get(
                        "source_candidate_sets_differ_by_event", False
                    )
                    for a in graph_v2_audits
                ),
                "limitation": (
                    "A bounded replacement-plan re-execution checks that "
                    "project_id cannot replace event_id without admitting "
                    "other real YAGO answers. Status and year selection "
                    "are two actually compiled local relation families. "
                    "This is not proof against all possible strategies."
                ),
            },
            "task_graph_first_v1": {
                "scenes": len(graph_first_scenes),
                "information_necessity_passed": sum(
                    r.get("passed", False) for r in graph_first_audits
                ),
                "identification_unique": sum(
                    r.get("identification_unique", False)
                    for r in graph_first_audits
                ),
                "completed_event_status_causal": sum(
                    r.get("required_field_necessity", {}).get(
                        "event_status", False)
                    for r in graph_first_audits
                ),
                "real_yago_fork_causal": sum(
                    r.get("required_field_necessity", {}).get(
                        "real_fork_motif_object", False)
                    for r in graph_first_audits
                ),
                "both_reference_sources_causal": sum(
                    bool(r.get("simulated_directory_necessity"))
                    and all(r["simulated_directory_necessity"].values())
                    for r in graph_first_audits
                ),
                "abstract_graph_shape_distribution": dict(Counter(
                    scene["task_first_macro_graph"]["shape"]
                    if "task_first_macro_graph" in scene else
                    scene["problem_spec"]["task_first_macro_graph"]["shape"]
                    for scene in graph_first_scenes
                )),
                "distinct_graph_role_topologies": len({
                    tuple(
                        (node["role"], tuple(node["depends_on"]))
                        for node in s["problem_spec"]["task_first_macro_graph"]["nodes"]
                    ) for s in graph_first_scenes
                }),
                "synthetic_scheduled_event_count": sum(
                    len(s["problem_spec"]["task_first_evidence"][
                        "scheduled_event_records"])
                    for s in graph_first_scenes
                ),
                "causal_vs_context_scope": (
                    "Event status and the real fork are checked by executable "
                    "field-level counterfactuals; organizer and venue remain "
                    "explicitly contextual, not credited as causal work."
                ),
            },
            "causal_hybrid_v4": {
                "scenes_checked": len(v4_scenes),
                "necessity_passed": sum(
                    a.get("passed", False) for a in v4_audits
                ),
                "identification_unique": sum(
                    a.get("identification_unique", False) for a in v4_audits
                ),
                "all_single_clues_necessary": sum(
                    bool(a.get("single_clue_necessary"))
                    and all(a["single_clue_necessary"].values())
                    for a in v4_audits
                ),
                "all_discovery_macros_necessary": sum(
                    bool(a.get("discovery_macro_necessary"))
                    and all(a["discovery_macro_necessary"].values())
                    for a in v4_audits
                ),
                "both_simulated_directories_necessary": sum(
                    bool(a.get("link_branch_necessary"))
                    and all(a["link_branch_necessary"].values())
                    for a in v4_audits
                ),
                "synthetic_event_necessary": sum(
                    a.get("synthetic_event_necessary", False)
                    for a in v4_audits
                ),
                "genuine_yago_fork_necessary": sum(
                    a.get("grounded_real_fork_necessary", False)
                    for a in v4_audits
                ),
                "macro_shape_distribution": dict(Counter(
                    s["problem_spec"]["macro_plan"]["hybrid_shape"]
                    for s in v4_scenes
                )),
                "required_real_motif_distribution": dict(Counter(
                    s["problem_spec"]["macro_plan"]["real_fork"]["motif_kind"]
                    for s in v4_scenes
                )),
                "fictional_project_count": sum(
                    len(s["problem_spec"]["macro_plan"]["synthetic_graph"]["projects"])
                    for s in v4_scenes
                ),
                "fictional_event_count": sum(
                    len(s["problem_spec"]["macro_plan"]["synthetic_graph"]["events"])
                    for s in v4_scenes
                ),
                "fictional_relation_count": sum(
                    len(s["problem_spec"]["macro_plan"]["synthetic_truth"])
                    for s in v4_scenes
                ),
                "source_lineage_note": (
                    "Real YAGO Fork assertions are actually queried and "
                    "joined with synthetic event constraints. All fictional "
                    "projects/events/organizers/venues and their relations "
                    "remain simulated, not historical RDF facts. Fork "
                    "necessity is measured with controlled interventions."
                ),
            },
            "hybrid_world_v3": {
                "scenes_checked": len(v3_scenes),
                "necessity_passed": sum(
                    a.get("passed", False) for a in v3_audits
                ),
                "identification_unique": sum(
                    a.get("identification_unique", False) for a in v3_audits
                ),
                "all_clue_relaxations_ambiguous": sum(
                    bool(a.get("clue_relaxation_necessary"))
                    and all(a["clue_relaxation_necessary"].values())
                    for a in v3_audits
                ),
                "all_discovery_relaxations_ambiguous": sum(
                    bool(a.get("discovery_relaxation_necessary"))
                    and all(a["discovery_relaxation_necessary"].values())
                    for a in v3_audits
                ),
                "both_synthetic_branches_necessary": sum(
                    bool(a.get("synthetic_branch_necessity"))
                    and all(a["synthetic_branch_necessity"].values())
                    for a in v3_audits
                ),
                "all_discovery_bypass_ambiguous": sum(
                    a.get("all_discovery_filters_removed", {}).get(
                        "ambiguous", False
                    ) for a in v3_audits
                ),
                "real_yago_motif_distribution": dict(Counter(
                    s["problem_spec"]["macro_plan"]["real_motif"]["kind"]
                    for s in v3_scenes
                )),
                "macro_shape_distribution": dict(Counter(
                    s["problem_spec"]["macro_plan"]["hybrid_shape"]
                    for s in v3_scenes
                )),
                "synthetic_entity_count": sum(
                    len(s["problem_spec"]["macro_plan"]["synthetic_entities"])
                    for s in v3_scenes
                ),
                "synthetic_true_relation_count": sum(
                    len(s["problem_spec"]["macro_plan"]["synthetic_truth"])
                    for s in v3_scenes
                ),
                "source_lineage_note": (
                    "YAGO facts are real. Fictional projects and their "
                    "associations are not YAGO facts. Extra mined YAGO motif "
                    "edges are context and not yet consumed as a required "
                    "Macro source. Simulated candidate claims may be false."
                ),
            },
            "necessity_v2": {
                "scenes_checked": len(v2_scenes),
                "passed": sum(a.get("passed", False) for a in necessity),
                "identification_unique": sum(
                    a.get("identification_unique", False) for a in necessity
                ),
                "all_clue_relaxations_ambiguous": sum(
                    bool(a.get("source_counterfactuals"))
                    and all(
                        row.get("has_real_competing_answer", False)
                        for row in a["source_counterfactuals"].values()
                    ) for a in necessity
                ),
                "all_discovery_relaxations_ambiguous": sum(
                    bool(a.get("discovery_counterfactuals"))
                    and all(
                        row.get("has_real_competing_answer", False)
                        for row in a["discovery_counterfactuals"].values()
                    ) for a in necessity
                ),
                "no_discovery_bypass_blocked": sum(
                    a.get("all_discovery_filters_removed", {}).get(
                        "ambiguous", False
                    ) for a in necessity
                ),
                "scope": (
                    "Executable filter-relaxation checks with complete "
                    "real-YAGO alternative paths, not a proof of globally "
                    "optimal agent information acquisition."
                ),
            },
            "version": "typed_knowledge_v1_v2_v3_v4_and_task_graph_first_v1_v2_and_task_plan",
            "pattern_distribution": dict(Counter(
                scene["problem_spec"]["reasoning_pattern"]
                for scene in knowledge_scenes
            )),
            "scenes": len(knowledge_scenes),
            "yago_relation_hops_histogram": dict(sorted(hop_counts.items())),
            "macro_count_histogram": dict(sorted(macro_counts.items())),
            "macro_roles": dict(sorted(role_counts.items())),
            "all_source_counterfactuals_passed": sources_pass,
            "all_handoffs_necessary": sum(
                scene.get("difficulty_audit", {}).get(
                    "macro_first_audit", {}).get(
                    "handoff_counterfactuals", {}).get(
                    "all_handoffs_necessary", False)
                for scene in knowledge_scenes
            ),
            "provenance_note": (
                "YAGO triples are sampled from a local RDF file and frozen "
                "as provenance-bearing witnesses; synthetic dossier "
                "observations are fictional information carriers. Strict "
                "Replay is self-contained, not independent online YAGO verification."
            ),
            "scope": (
                "Canonical, versioned knowledge and hybrid-world subgrammars; "
                "not arbitrary six-role Macro DAGs or externally verified "
                "fictional media."
            ),
        }
    # Current Task Plan scene banks should not publish walls of zero-valued
    # statistics for retired generation experiments. Keep the general
    # Replay/graph/reasoning metrics and exact Task Plan necessity audit.
    # Older scene banks retain their original report shape for reproducible
    # historical tests and existing consumers.
    if scenes and all(
        scene.get("problem_spec", {}).get("reasoning_pattern") == "task_plan"
        and scene["problem_spec"]["macro_plan"].get("knowledge_world_mode")
        == "compose"
        for scene in scenes
    ):
        data.pop("macro_first_v2", None)
        data.pop("semantic_evidence_diagnostics", None)
        data.pop("evidence_graph", None)
        data.pop("evidence_relations", None)
        knowledge = data.get("knowledge_world_grounding")
        if knowledge is not None:
            keep = (
                "single_task_plan", "scenes", "pattern_distribution",
                "yago_relation_hops_histogram", "macro_count_histogram",
                "macro_roles", "all_source_counterfactuals_passed",
                "all_handoffs_necessary", "provenance_note", "scope",
            )
            data["knowledge_world_grounding"] = {
                "version": "task_plan_compose_v2",
                **{key: knowledge[key] for key in keep if key in knowledge},
            }
    return data


def audit_questions(args) -> dict:
    scenes = read_jsonl(args.scenes, limit=getattr(args, "max_records", None))
    result = quality_report(scenes, no_progress=getattr(args, "no_progress", False))
    if getattr(args, "output", None) is not None:
        _dump(args.output, result)
        result["report_path"] = str(args.output)
    return result


def _generate_batch(args) -> dict:
    """One independently reproducible Scenario/Question generation round."""
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
    # Keep the meaningful item-level Scenario and compilation bars visible.
    staged.no_progress = getattr(args, "no_progress", False)
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
        authored.no_progress = getattr(args, "no_progress", False)
        # Fast preflight validation has its own bar only in verbose mode.
        authored.verbose_progress = getattr(args, "verbose_progress", False)
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
            prior_state = authored.output_dir / "question_authoring_progress.json"
            if not prior_state.is_file():
                raise ValueError(
                    "completed_legacy_question_bank_has_no_resume_checkpoint"
                )
            previous_signature = json.loads(
                prior_state.read_text(encoding="utf-8")
            ).get("signature")
            current_signature = _question_signature(
                authored, read_jsonl(scene_path)
            )
            if previous_signature != current_signature:
                raise ValueError(
                    "Question model, frozen input, or prompt code changed; "
                    "choose a new output directory or enable overwrite"
                )
            question_summary = previous
        else:
            question_summary = review_questions(authored)
        scene_path = authored.output_dir / "scenes.jsonl"
        # The quality report also performs strict independent replay from the
    # frozen Program/World (not simply accepting stored difficulty flags).
    audit_args = copy.copy(args)
    # Strict Replay still runs, but its extra per-item bar is opt-in.
    audit_args.no_progress = (getattr(args, "no_progress", False)
                              or not getattr(args, "verbose_progress", False))
    audit_args.scenes = scene_path
    audit_args.max_records = None
    audit_args.output = report_path
    report = audit_questions(audit_args)
    final_count = report["question_publishable"]
    failed_question_count = (
        question_summary.get("failed", 0) if question_summary else 0
    )
    summary = {
        # A completed fixed candidate batch is DONE even when legitimate
        # validation rejects some candidates. Acceptance is an outcome,
        # never a quota to be refilled or a reason to discard valid Scenes.
        "status": (
            "done" if final_count > 0 and report["status"] == "valid"
            else "failed" if final_count == 0 else "partial"
        ),
        "pipeline": "maestro_generate_question_v1",
        "source": constructed_summary["seed_source"],
        "requested": args.target,
        "candidate_count": args.target,
        "attempted": constructed_summary.get("attempted", args.target),
        "generated": final_count,
        "accepted": final_count,
        "rejected": args.target - final_count,
        "failed": args.target - final_count,
        "acceptance_rate": round(final_count / args.target, 4),
        "selection_policy": "fixed_initial_candidates_no_refill_no_quota_filter",
        "difficulty_mix_policy": "initial_candidate_sampling_only",
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


def generate_questions(args) -> dict:
    """Review exactly one fixed candidate batch; retain every valid Scene.

    Candidate count and difficulty mix describe the *initial* proposed
    pool. No additional candidates are sampled to refill rejection losses,
    and no accepted Scene is withheld to satisfy a difficulty quota.
    """
    return _generate_batch(args)
