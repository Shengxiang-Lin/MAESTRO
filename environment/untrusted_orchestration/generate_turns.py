"""Config-driven batch generation of round-local untrusted O1 Main decisions.

Each branch begins from an accepted clean Base Episode. At t>=2 only the
Sub-Agent batch returned in t-1 is eligible for injection; the entire
earlier visible history remains trusted. Deterministic WorkProduct replay
then drives every newly scheduled Sub-Agent action.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import json
import os
import random
import shutil
from threading import Lock

from tqdm.auto import tqdm

from .faults import (
    Fault, FaultUnsupported, choose_targets, probe_eligible_fault_modes,
)
from .trajectory_builder import build_branch
from .offline_pipeline import author_branch_reports, author_branch_main


def _jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(str(path))
    return [json.loads(row) for row in path.read_text(encoding="utf-8").splitlines()
            if row.strip()]


def _atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)
                         + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def _hash_file(path: Path) -> str:
    hasher = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _signature(inputs: tuple[Path, Path, Path], settings: dict) -> dict:
    directory = Path(__file__).resolve().parent
    code_files = (
        "faults.py", "state_machine.py", "report_author.py",
        "trajectory_builder.py", "generate_turns.py",
        "evidence_planning.py", "offline_pipeline.py",
        "../generate_question/compiler.py",
        "../generate_question/program_ir.py",
        "../base_orchestration/o1_compat.py",
        "../base_orchestration/action_author.py",
        "../implement_subtask/reporting.py",
        "config.py",
        "../../orchestra/base_main_agent.py",
    )
    return {
        "version": "maestro.untrusted_orchestration.every_feedback_round.v3",
        "input_sha256": [_hash_file(path) for path in inputs],
        "source_code_sha256": {
            name: _hash_file(directory / name) for name in code_files
        },
        "settings": settings,
    }


def _plan_branches(
    scenes: list[dict], base: dict[str, dict],
    *, branches_per_scene: int | None = None, seed: int,
    fault_modes: list[str], persistence: str,
    multi_target_probability: float,
    max_rounds: int,
) -> tuple[list[dict], int, int]:
    """Expand EVERY accepted Main feedback round into one fault scenario.

    The unit is (question, Main decision round), *not* a randomly sampled
    number of branches per Question. A decision round t receives the latest
    Sub-Agent batch returned from delegate round t-1. Every such available
    round is retained in the plan, even when no admissible fault exists;
    those rounds produce explicit skipped_unsupported journals.

    `branches_per_scene` is an optional *debugging cap* for compatibility
    with prior callers. None means no cap; the production YAML uses None.
    The max_rounds budget also limits the global round positions.
    """
    if branches_per_scene is not None and (
        type(branches_per_scene) is not int or branches_per_scene < 1
    ):
        raise ValueError("invalid_debug_round_cap")
    planned, no_base, no_eligible = [], 0, 0
    for scene_index, scene in enumerate(scenes):
        episode = base.get(scene["task_id"])
        if episode is None:
            no_base += 1
            continue
        delegates = [
            row for row in episode["trajectory"]
            if row["action"] == "delegate_task"
        ]
        waves = scene["macro_graph"]["reference_waves"]
        # Check every trusted returned batch against the actual global DAG
        # rather than silently producing incorrect unit/position labels.
        if len(delegates) > len(waves):
            raise ValueError("trusted_delegate_exceeds_reference_waves")
        rounds = []
        for index, delegate in enumerate(delegates):
            expected = index + 1
            if delegate.get("attempt") != expected:
                raise ValueError("trusted_delegate_round_alignment_invalid")
            results = (delegate.get("result") or {}).get("subtask_results")
            if not isinstance(results, list) or len(results) != len(waves[index]):
                raise ValueError("trusted_returned_batch_alignment_invalid")
            decision_round = expected + 1
            if decision_round <= max_rounds:
                rounds.append((decision_round, delegate))
        if branches_per_scene is not None:
            # A diagnostic cap uses *chronological rounds*, not random
            # selection. Production never caps coverage at two branches.
            rounds = rounds[:branches_per_scene]
        if not rounds:
            no_eligible += 1
            continue

        eligible = probe_eligible_fault_modes(scene, fault_modes)
        tasks = {task["task_id"]: task
                 for task in scene["macro_graph"]["tasks"]}
        eligible_count = 0
        for decision_round, previous_delegate in rounds:
            wave = waves[decision_round - 2]
            available_modes = [
                mode for mode in fault_modes
                if any(mode in eligible.get(tid, []) for tid in wave)
            ]
            # Independent deterministic randomness per (question, round)
            # means adding/removing another round cannot perturb this one.
            round_seed = seed + scene_index * 1009 + decision_round * 92821
            rng = random.Random(round_seed)
            faults = []
            skip_reason = None
            chosen_mode = None
            if available_modes:
                chosen_mode = rng.choice(available_modes)
                supported = [
                    tid for tid in wave if chosen_mode in eligible.get(tid, [])
                ]
                multiple = (
                    rng.random() < multi_target_probability
                    and len(supported) > 1
                )
                chosen_targets = choose_targets(
                    supported, tasks, seed=round_seed, multiple=multiple,
                )
                faults = [
                    Fault(task_id=tid, mode=chosen_mode,
                          seed=round_seed, persistence=persistence)
                    for tid in chosen_targets
                ]
                eligible_count += 1
            else:
                skip_reason = "no_executable_fault_on_returned_batch"

            planned.append({
                "index": len(planned),
                "unit": "main_feedback_round",
                "sample_id": (
                    scene["task_id"] + f"-fork-round-{decision_round:02d}"
                ),
                "scene_index": scene_index,
                "task_id": scene["task_id"],
                "branch_round": decision_round,
                "received_from_delegate_round": decision_round - 1,
                "incoming_task_ids": list(wave),
                "incoming_subtask_count": len(
                    previous_delegate["result"]["subtask_results"]
                ),
                "fault_mode": chosen_mode,
                "faults": faults,
                "skip_reason": skip_reason,
            })
        if not eligible_count:
            no_eligible += 1
    return planned, no_base, no_eligible

def _publish(output: Path, planned: list[dict]) -> dict:
    journals = output / "journal"
    episodes, positive_turns, audits, failures = [], [], [], []
    counters = Counter()
    mode_outcomes: dict[str, Counter] = defaultdict(Counter)
    report_modes = Counter()
    main_modes = Counter()
    report_statuses = Counter()
    main_statuses = Counter()
    execution_outcomes = Counter()
    report_guard_failures = Counter()
    report_review_failures = Counter()
    executor_normalized_results = 0
    for plan in planned:
        path = journals / f"{plan['index']:06d}.json"
        if not path.is_file():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if data["task_id"] != plan["task_id"]:
            raise RuntimeError("untrusted_journal_scene_mismatch")
        counters[data["status"]] += 1
        for mode in {fault.mode for fault in plan["faults"]}:
            mode_outcomes[mode][data["status"]] += 1
        if data["status"] == "recovered" and (
            not isinstance(data.get("private_audit"), dict)
            or data["private_audit"].get("parallel_batches_verified") is not True
        ):
            raise RuntimeError(
                "untrusted_unsafe_parallel_batch_cannot_publish_positive_sft"
            )
        if data.get("episode") is not None:
            public_episode = deepcopy(data["episode"])
            # Truth labels belong in private audit, never public SFT.
            public_episode.pop("outcome", None)
            public_episode.pop("recovered", None)
            episodes.append(public_episode)
        if data["status"] == "recovered":
            positive_turns.extend(data["turns"])
        else:
            failures.append({
                "task_id": data["task_id"],
                "branch_index": plan["index"],
                "branch_round": plan["branch_round"],
                "status": data["status"],
                "error": data.get("error"),
            })
        if data.get("private_audit") is not None:
            audit = data["private_audit"]
            audits.append(audit)
            execution_outcomes[audit.get("execution_outcome", audit["outcome"])] += 1
            for review in audit.get("subagent_report_reviews", []):
                result = review["authoring"]
                report_modes[result["mode"]] += 1
                report_statuses[result.get("status", "unknown")] += 1
                executor_normalized_results += bool(
                    result.get("executor_result_normalized")
                )
                for failure in result.get("failures", []):
                    if failure.get("stage") == "guard":
                        report_guard_failures.update(
                            failure.get("errors", [])
                        )
                    else:
                        report_review_failures.update(
                            [failure.get("stage", "unknown")]
                        )
            for review in audit.get("main_action_reviews", []):
                result = review["authoring"]
                main_modes[result["mode"]] += 1
                main_statuses[result.get("status", "unknown")] += 1
    _atomic_jsonl(output / "episodes.jsonl", episodes)
    _atomic_jsonl(output / "main_turns.jsonl", positive_turns)
    _atomic_jsonl(output / "failed.jsonl", failures)
    _atomic_jsonl(output / "private" / "fault_alignment.jsonl", audits)
    return {
        "completed_branches": sum(counters.values()),
        "recovered": counters["recovered"],
        "incorrect": counters["incorrect_result_after_actual_propagation"],
        "unresolved": sum(count for status, count in counters.items()
                          if status not in {
                              "recovered", "skipped_unsupported",
                              "rejected_validation",
                              "incorrect_result_after_actual_propagation",
                          }),
        "skipped_or_rejected": (
            counters["skipped_unsupported"] +
            counters["rejected_validation"]
        ),
        "main_training_turns": len(positive_turns),
        "execution_recovered_branches": execution_outcomes["recovered"],
        "execution_outcomes": dict(execution_outcomes),
        "language_review_rejected_after_recovery":
            sum(1 for audit in audits if audit.get("execution_recovered")
                and not audit.get("language_review_accepted", True)),
        "outcomes": dict(counters),
        "fault_mode_outcomes": {
            mode: dict(counts)
            for mode, counts in sorted(mode_outcomes.items())
        },
        "subagent_report_generation_modes": dict(report_modes),
        "subagent_report_review_statuses": dict(report_statuses),
        "subagent_executor_result_normalized": executor_normalized_results,
        "subagent_guard_failure_causes": dict(report_guard_failures),
        "subagent_review_failure_stages": dict(report_review_failures),
        "main_action_generation_modes": dict(main_modes),
        "main_action_review_statuses": dict(main_statuses),
    }


def generate_turns(
    *, scenes_path: Path, reports_path: Path, base_episodes_path: Path,
    output_dir: Path,
    max_scenes: int | None = 30, branches_per_scene: int | None = None,
    seed: int = 17, fault_modes: list[str] | None = None,
    persistence: str = "once", multi_target_probability: float = 0.0,
    max_rounds: int = 16, max_cost: int = 80, max_retries: int = 1,
    max_parallel_tasks: int = 4, workers: int = 4,
    max_revisions: int = 2,
    report_max_revisions: int | None = None,
    main_max_revisions: int | None = None,
    sub_model: str = "qwen3-8b",
    model_alias: str = "model_1", overwrite: bool = False,
    no_progress: bool = False,
    author=None, critic=None, report_author=None, report_critic=None,
    author_model: str | None = None, critic_model: str | None = None,
    report_author_model: str | None = None,
    report_critic_model: str | None = None,
    main_mode: str | None = None,
    report_mode: str | None = None,
) -> dict:
    paths = tuple(map(Path, (scenes_path, reports_path, base_episodes_path)))
    output = Path(output_dir)
    if not all(path.is_file() for path in paths):
        raise FileNotFoundError("untrusted_requires_frozen_scenes_reports_base")
    if type(workers) is not int or not 1 <= workers <= 64:
        raise ValueError("untrusted_workers_invalid")
    if branches_per_scene is not None and (
        type(branches_per_scene) is not int or branches_per_scene < 1
    ):
        raise ValueError("untrusted_invalid_round_debug_cap")
    if bool(author) != bool(critic) or bool(report_author) != bool(report_critic):
        raise ValueError("untrusted_author_and_critic_must_be_paired")
    if main_mode is None:
        main_mode = "llm" if author is not None else "template"
    if report_mode is None:
        report_mode = "llm" if report_author is not None else "template"
    if (main_mode not in {"template", "llm"}
            or report_mode not in {"template", "llm"}):
        raise ValueError("untrusted_invalid_stage_llm_mode")
    if ((main_mode == "llm") != (author is not None)
            or (report_mode == "llm") != (report_author is not None)):
        raise ValueError("untrusted_stage_mode_requires_matching_clients")
    if report_max_revisions is None:
        report_max_revisions = max_revisions
    if main_max_revisions is None:
        main_max_revisions = max_revisions
    if output.resolve() in {
        path.resolve() for path in paths
    } or any(output.resolve() == p.parent.resolve() for p in paths):
        raise ValueError("untrusted_output_overwrites_input")

    modes = list(fault_modes or ["missing", "partial", "wrong"])
    settings = {
        "max_scenes": max_scenes,
        "branches_per_scene": branches_per_scene, "seed": seed,
        "fault_modes": modes, "persistence": persistence,
        "multi_target_probability": multi_target_probability,
        "max_rounds": max_rounds, "max_cost": max_cost,
        "max_retries": max_retries, "max_parallel_tasks": max_parallel_tasks,
        "max_revisions": max_revisions,
        "report_max_revisions": report_max_revisions,
        "main_max_revisions": main_max_revisions,
        "sub_model": sub_model,
        "model_alias": model_alias, "author_model": author_model,
        "critic_model": critic_model,
        "report_author_model": report_author_model,
        "report_critic_model": report_critic_model,
        "main_mode": main_mode,
        "report_mode": report_mode,
    }
    signature = _signature(paths, settings)
    scenes = _jsonl(paths[0])[:max_scenes]
    base_list = _jsonl(paths[2])
    if len({x["task_id"] for x in base_list}) != len(base_list):
        raise ValueError("untrusted_duplicate_base_episode")
    base = {x["task_id"]: x for x in base_list}
    rows_by_scene = defaultdict(list)
    for row in _jsonl(paths[1]):
        rows_by_scene[row["scene_task_id"]].append(row)
    planned, no_base, no_eligible = _plan_branches(
        scenes, base, branches_per_scene=branches_per_scene,
        seed=seed, fault_modes=modes, persistence=persistence,
        multi_target_probability=multi_target_probability,
        max_rounds=max_rounds,
    )
    if overwrite and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "manifest.json"
    if manifest.is_file():
        if json.loads(manifest.read_text(encoding="utf-8")) != signature:
            raise ValueError("untrusted_resume_input_or_code_signature_mismatch")
    elif any(output.iterdir()):
        raise ValueError("untrusted_existing_output_without_manifest")
    else:
        _atomic_json(manifest, signature)
    # Explicit, inspectable round index; unsupported feedback rounds are
    # retained instead of disappearing silently from the total.
    _atomic_jsonl(output / "round_index.jsonl", [{
        "round_sample_id": plan["sample_id"],
        "question_id": plan["task_id"],
        "main_decision_round": plan["branch_round"],
        "received_from_delegate_round":
            plan["received_from_delegate_round"],
        "latest_subagent_task_ids": plan["incoming_task_ids"],
        "latest_subagent_count": plan["incoming_subtask_count"],
        "fault_eligible": not bool(plan["skip_reason"]),
        "skip_reason": plan["skip_reason"],
    } for plan in planned])

    # Three genuine phases with separate signed, resumable artifacts.
    # 4A: zero model calls; Global execution + Main-visible constrained
    #     evidence requests and ordered task attempts.
    # 4B: only Sub-Agent Author/Guard/Critic calls.
    # 4C: only Main Author/Guard/Critic calls, based on FINAL 4B reports.
    stage_a = output / "stage4a" / "journal"
    stage_b = output / "stage4b" / "journal"
    journal = output / "journal"
    for directory in (stage_a, stage_b, journal):
        directory.mkdir(parents=True, exist_ok=True)

    def saved(directory: Path, plan: dict) -> dict | None:
        path = directory / f"{plan['index']:06d}.json"
        return (json.loads(path.read_text(encoding="utf-8"))
                if path.is_file() else None)

    def write(directory: Path, plan: dict, data: dict) -> None:
        _atomic_json(directory / f"{plan['index']:06d}.json", data)

    def as_failure(scene: dict, exc: Exception) -> dict:
        if isinstance(exc, FaultUnsupported):
            status = "skipped_unsupported"
        else:
            status = "rejected_validation"
        return {
            "task_id": scene["task_id"], "status": status,
            "error": str(exc)[:300], "episode": None,
            "turns": [], "private_audit": None,
        }

    def run_parallel(phase: str, directory: Path, unit: str,
                     total: int, initial: int, callback_worker):
        pending = [p for p in planned if saved(directory, p) is None]
        with tqdm(
            total=max(total, initial), initial=initial,
            desc=phase, unit=unit, dynamic_ncols=True,
            disable=no_progress, leave=True,
        ) as bar:
            lock = Lock()
            def tick(amount: int = 1) -> None:
                with lock:
                    if bar.total is not None and bar.n + amount > bar.total:
                        bar.total = bar.n + amount
                    bar.update(amount)
            with ThreadPoolExecutor(
                max_workers=workers,
                thread_name_prefix="stage4-" + phase[:2],
            ) as pool:
                futures = [
                    pool.submit(callback_worker, plan, tick)
                    for plan in pending
                ]
                for future in as_completed(futures):
                    future.result()
            bar.total = bar.n
            bar.refresh()

    def simulate(plan: dict, tick) -> None:
        scene = scenes[plan["scene_index"]]
        if plan["skip_reason"]:
            write(stage_a, plan, {
                "task_id": scene["task_id"],
                "round_sample_id": plan["sample_id"],
                "fork_round": plan["branch_round"],
                "status": "skipped_unsupported",
                "error": plan["skip_reason"],
                "episode": None, "turns": [], "private_audit": None,
            })
            tick()
            return
        try:
            episode, _, audit = build_branch(
                scene, rows_by_scene[scene["task_id"]],
                base[scene["task_id"]],
                branch_round=plan["branch_round"], faults=plan["faults"],
                max_rounds=max_rounds, max_cost=max_cost,
                max_retries=max_retries,
                max_parallel_tasks=max_parallel_tasks,
                sub_model=sub_model, model_alias=model_alias,
                # Stage 4A must never call the language models.
                author=None, critic=None, report_author=None,
                report_critic=None,
            )
            data = {
                "task_id": scene["task_id"],
                "round_sample_id": plan["sample_id"],
                "fork_round": plan["branch_round"],
                "status": audit["execution_outcome"],
                "episode": episode, "turns": [],
                "private_audit": audit,
            }
        except (FaultUnsupported, ValueError, KeyError, TypeError) as exc:
            data = as_failure(scene, exc)
        write(stage_a, plan, data)
        tick()

    run_parallel(
        "Stage 4A evidence simulation", stage_a, "round",
        len(planned), sum(saved(stage_a, p) is not None for p in planned),
        simulate,
    )

    def report_count(data: dict) -> int:
        audit = (data or {}).get("private_audit") or {}
        return len(audit.get("subagent_report_reviews", []))

    total_reports = sum(report_count(saved(stage_a, p)) for p in planned)
    initial_reports = sum(report_count(saved(stage_b, p)) for p in planned)

    def make_reports(plan: dict, tick) -> None:
        source = saved(stage_a, plan)
        scene = scenes[plan["scene_index"]]
        if source["episode"] is None:
            write(stage_b, plan, source)
            return
        try:
            episode, audit = author_branch_reports(
                scene, rows_by_scene[scene["task_id"]],
                source["episode"], source["private_audit"],
                author=report_author, critic=report_critic,
                max_revisions=report_max_revisions,
                on_report=lambda _status: tick(),
            )
            data = {
                **source, "episode": episode, "private_audit": audit,
            }
        except (ValueError, KeyError, TypeError) as exc:
            data = as_failure(scene, exc)
        write(stage_b, plan, data)

    run_parallel(
        "Stage 4B Sub-Agent feedback", stage_b, "report",
        total_reports, initial_reports, make_reports,
    )

    def action_count(data: dict) -> int:
        if data is None or data.get("episode") is None:
            return 0
        audit = data["private_audit"]
        fork = audit["branch_round"]
        return sum(
            row["attempt"] >= fork
            and row["action"] in {"delegate_task", "complete"}
            for row in data["episode"]["trajectory"]
        )

    total_actions = sum(action_count(saved(stage_b, p)) for p in planned)
    initial_actions = sum(
        len((saved(journal, p) or {}).get("private_audit", {})
            .get("main_action_reviews", []))
        if (saved(journal, p) or {}).get("private_audit") else 0
        for p in planned
    )

    def make_main(plan: dict, tick) -> None:
        source = saved(stage_b, plan)
        scene = scenes[plan["scene_index"]]
        if source["episode"] is None:
            write(journal, plan, source)
            return
        try:
            episode, turns, audit = author_branch_main(
                scene, source["episode"], source["private_audit"],
                author=author, critic=critic,
                max_revisions=main_max_revisions,
                max_rounds=max_rounds, sub_model=sub_model,
                model_alias=model_alias,
                on_main=lambda _status: tick(),
            )
            data = {
                **source, "status": audit["outcome"],
                "episode": episode, "turns": turns,
                "private_audit": audit,
            }
        except (ValueError, KeyError, TypeError) as exc:
            data = as_failure(scene, exc)
        write(journal, plan, data)

    run_parallel(
        "Stage 4C Main orchestration", journal, "action",
        total_actions, initial_actions, make_main,
    )
    counts_summary = _publish(output, planned)
    summary = {
        "status": (
            "done" if counts_summary["recovered"] == len(planned)
            else "partial"
        ),
        "pipeline": "maestro_untrusted_orchestration_every_feedback_round_v3",
        "mode": ("llm" if main_mode == "llm" or report_mode == "llm"
                 else "template"),
        "subagent_report_mode": report_mode,
        "main_action_mode": main_mode,
        "input_scenes": len(scenes),
        "scenes_without_accepted_base": no_base,
        "round_sampling_unit": "question_x_main_feedback_round",
        "rounds_extracted_from_base": len(planned),
        "rounds_with_executable_fault": sum(
            not plan["skip_reason"] for plan in planned
        ),
        "rounds_without_executable_fault": sum(
            bool(plan["skip_reason"]) for plan in planned
        ),
        "questions_with_extracted_rounds": len({
            plan["task_id"] for plan in planned
        }),
        "rounds_per_question": dict(Counter(
            plan["task_id"] for plan in planned
        )),
        "round_index": str(output / "round_index.jsonl"),
        "scenes_without_eligible_fault": no_eligible,
        "requested_branches": len(planned),
        "requested_feedback_rounds": len(planned),
        **counts_summary,
        "stage4a_evidence_plans": str(output / "stage4a" / "journal"),
        "stage4b_reviewed_reports": str(output / "stage4b" / "journal"),
        "public_turns": str(output / "main_turns.jsonl"),
        "public_episodes": str(output / "episodes.jsonl"),
        "private_fault_alignment": str(
            output / "private" / "fault_alignment.jsonl"
        ),
        "limitations": [
            "Every accepted base Main decision round that receives a prior Sub-Agent batch is extracted, except an explicit debug cap or max-rounds cutoff.",
            "One fault injection window per round sample: the latest completed Sub-Agent batch.",
            "Only preflight-executable fault/macro pairs are sampled.",
            "Only publicly observable discrepancies trigger targeted repair labels.",
            "Clean World/Gold never change; uncorrected counterfactuals are private, not positive SFT.",
            "Stage 4A determines evidence requests and an executable global schedule using only Main-visible report justifications.", 
            "4B materializes every actual executed Sub-Agent Attempt before any Stage 4C Main LLM is called.",
            "4C naturalizes 4A's fixed legal scheduling contracts against the finalized Main-visible 4B history.",
            "Template actions are deterministic teacher demonstrations, not online Main-Agent rollouts.",
            "Sub-Agent Author/Critic write full prose for this attempt; status and actual WorkProduct remain code-owned.",
            "Main-Agent Author/Critic only naturalize the program-selected O1 tasks; private state does not enter their prompt.",
            "LLM-mode fallback reports are diagnostic only, never silently counted as reviewed positive SFT.",
        ],
    }
    _atomic_json(output / "summary.json", summary)
    if not no_progress:
        tqdm.write(
            "Stage 4 branches: "
            f"{summary['completed_branches']}/{summary['requested_feedback_rounds']} "
            f"| execution recovered={summary['execution_recovered_branches']} "
            f"| incorrect={summary['incorrect']} "
            f"| reviewed SFT branches={summary['recovered']} "
            f"| SFT turns={summary['main_training_turns']}"
        )
    return summary
