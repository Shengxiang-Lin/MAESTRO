"""Read-only diagnostic audit for V4 task-preserving Question naturalization.

Heuristic violations are *warnings*, not a proof of semantic drift. This is
useful for comparing the original 100-question local pilot against a fresh
v4.2 Author/Critic run, without rebuilding or modifying any Macro TaskGraph.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from .question_review import (
    polished_reference_question, reference_question, surface_issues,
)
from .validation import validate_clean_scene


FIDELITY_FLAGS = {
    "unsupported_concrete_detail",
    "invented_current_state",
    "internal_workflow_order_exposed",
    "ranking_target_changed",
    "invented_numeric_constraint",
    "invented_media_type",
    "unverified_attachment_or_observation",
    "private_orchestration_plan_leak",
    "changed_setting",
}
STYLE_FLAGS = {
    "formulaic_evidence_phrasing",
    "changed_or_missing_setting",
    "missing_case_subject",
}


def audit_scenes(scenes: list[dict], *, max_examples: int = 12) -> dict:
    if max_examples < 0:
        raise ValueError("max_examples must be nonnegative")
    counts = Counter()
    flagged = Counter()
    issues_by_status = Counter()
    examples = []
    unique = Counter()
    for index, scene in enumerate(scenes):
        problems = validate_clean_scene(scene)
        if problems:
            raise ValueError(
                f"Invalid source scene #{index + 1}: {problems}"
            )
        meta = scene.get("question_authoring", {})
        status = meta.get("status", "deterministic")
        policy = meta.get("review_policy_version", "v4.1_or_unreviewed")
        question = scene["question"]
        ref = reference_question(scene)
        checklist = surface_issues(scene, question, baseline=ref)
        if status == "fallback":
            counts["fallback"] += 1
            counts["fallback_reason:" + meta.get("reason", "unknown")] += 1
        elif status == "accepted":
            counts["accepted"] += 1
            if meta.get("revision_rounds", 0):
                counts["accepted_after_revision"] += 1
        else:
            counts["deterministic"] += 1
        counts["policy:" + policy] += 1
        counts["total"] += 1
        # Descriptive metrics for evaluating naturalness in a paired pilot.
        # Neither a question mark nor short length is mandatory.
        counts["total_word_count"] += len(question.split())
        if question.strip().casefold().startswith("please "):
            counts["starts_with_please"] += 1
        if "?" in question:
            counts["has_question_mark"] += 1
        if re.search(r"\b(?:then|next|finally|once)\b", question, re.I):
            counts["procedural_connector_cues"] += 1
        # Count ALL critic attempts, not just the ultimately accepted one.
        for attempt in meta.get("audit", []):
            review = attempt.get("critic_review") or {}
            if not review:
                continue
            counts["critic_review_rounds"] += 1
            if review.get("user_framing") == "workflow_instructions":
                counts["critic_flagged_workflow_rounds"] += 1
            if review.get("user_framing") == "robotic":
                counts["critic_flagged_robotic_rounds"] += 1
            if review.get("goal_alignment") == "changed":
                counts["critic_flagged_goal_shift_rounds"] += 1
            if review.get("task_sufficiency") == "insufficient":
                counts["critic_flagged_insufficiency_rounds"] += 1
            if review.get("unsupported_additions"):
                counts["critic_flagged_unsupported_additions_rounds"] += 1
        if "documented findings" in question.casefold():
            counts["stock_documented_findings"] += 1
        if question == polished_reference_question(ref) and question != ref:
            counts["safe_fallback_polished"] += 1
        key = re.sub(r"\W+", " ", question.casefold()).strip()
        unique[key] += 1
        for issue in checklist:
            flagged[issue] += 1
            issues_by_status[(status, issue)] += 1
        fidelity = [i for i in checklist if i in FIDELITY_FLAGS]
        style = [i for i in checklist if i in STYLE_FLAGS]
        if fidelity:
            counts["questions_with_fidelity_warnings"] += 1
        if style:
            counts["questions_with_style_warnings"] += 1
        if (fidelity or style) and len(examples) < max_examples:
            examples.append({
                "index": index + 1,
                "task_id": scene["task_id"],
                "status": status,
                "review_policy": policy,
                "fidelity_warnings": fidelity,
                "style_warnings": style,
                "reference_question": ref,
                "candidate_question": question,
            })
    return {
        "mode": "heuristic_posthoc_question_quality_audit",
        "checked": counts["total"],
        "question_status": {
            key: counts[key] for key in (
                "accepted", "accepted_after_revision", "fallback",
                "deterministic",
            )
        },
        "quality_metrics": {
            "unique_questions": len(unique),
            "duplicate_question_instances": sum(
                n - 1 for n in unique.values() if n > 1
            ),
            "stock_documented_findings": counts["stock_documented_findings"],
            "questions_with_fidelity_warnings":
                counts["questions_with_fidelity_warnings"],
            "questions_with_style_warnings":
                counts["questions_with_style_warnings"],
            "safe_fallback_polished": counts["safe_fallback_polished"],
            "mean_question_words": round(
                counts["total_word_count"] / counts["total"], 2
            ),
            "starts_with_please": counts["starts_with_please"],
            "has_question_mark": counts["has_question_mark"],
            "procedural_connector_cues": counts["procedural_connector_cues"],
        },
        "critic_diagnostics": {
            "reviewed_rounds": counts["critic_review_rounds"],
            "workflow_instruction_rounds":
                counts["critic_flagged_workflow_rounds"],
            "robotic_wording_rounds": counts["critic_flagged_robotic_rounds"],
            "goal_shift_rounds": counts["critic_flagged_goal_shift_rounds"],
            "task_insufficiency_rounds":
                counts["critic_flagged_insufficiency_rounds"],
            "unsupported_additions_rounds":
                counts["critic_flagged_unsupported_additions_rounds"],
            "notes": (
                "Counts are model judgments per review attempt, not "
                "independent semantic ground truth."
            ),
        },
        "warning_counts": dict(flagged),
        "status_warning_counts": {
            f"{status}:{issue}": n
            for (status, issue), n in sorted(issues_by_status.items())
        },
        "review_policy_counts": {
            name.removeprefix("policy:"): value
            for name, value in counts.items()
            if name.startswith("policy:")
        },
        "fallback_reasons": {
            name.removeprefix("fallback_reason:"): value
            for name, value in counts.items()
            if name.startswith("fallback_reason:")
        },
        "flagged_examples": examples,
        "limitations": [
            "lexical_and_pattern_warnings_not_formal_semantic_proof",
            "new_policy_warnings_do_not_retroactively_certify_legacy_questions",
            "manual_review_still_needed_for_unflagged_semantic_shifts",
        ],
    }


def audit_file(path: Path, *, max_records: int | None = None,
               max_examples: int = 12) -> dict:
    if max_records is not None and max_records < 1:
        raise ValueError("max_records must be >= 1")
    scenes = []
    with Path(path).open(encoding="utf-8") as reader:
        for line in reader:
            if line.strip():
                scenes.append(json.loads(line))
                if max_records is not None and len(scenes) >= max_records:
                    break
    if not scenes:
        raise ValueError("no V4 records available")
    return audit_scenes(scenes, max_examples=max_examples)
