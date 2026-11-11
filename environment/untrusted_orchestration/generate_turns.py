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
        "../generate_question/compiler.py",
        "../generate_question/program_ir.py",
        "../base_orchestration/o1_compat.py",
        "../base_orchestration/action_author.py",
        "../implement_subtask/reporting.py",
        "config.py",
        "../../orchestra/base_main_agent.py",
    )
    return {
        "version": "maestro.untrusted_orchestration.round_local.v1",
        "input_sha256": [_hash_file(path) for path in inputs],
        "source_code_sha256": {
            name: _hash_file(directory / name) for name in code_files
        },
        "settings": settings,
    }


def _plan_branches(
    scenes: list[dict], base: dict[str, dict],
    *, branches_per_scene: int, seed: int,
    fault_modes: list[str], persistence: str,
    multi_target_probability: float,
    max_rounds: int,
) -> tuple[list[dict], int, int]:
    """Sample only executable last-batch faults, retaining mode diversity."""
    planned, no_base, no_eligible = [], 0, 0
    for i, scene in enumerate(scenes):
        episode = base.get(scene["task_id"])
        if episode is None:
            no_base += 1
            continue
        delegates = [
            row for row in episode["trajectory"]
            if row["action"] == "delegate_task"
        ]
        rounds = list(range(2, min(len(delegates) + 1, max_rounds) + 1))
        if not rounds:
            no_eligible += 1
            continue
        eligible = probe_eligible_fault_modes(scene, fault_modes)
        waves = scene["macro_graph"]["reference_waves"]
        tasks = {a["task_id"]: a for a in scene["macro_graph"]["tasks"]}
        rng = random.Random(seed + i * 1009)
        rng.shuffle(rounds)
        picked_rounds = set()
        added = 0
        for slot in range(min(branches_per_scene, len(rounds))):
            # Rotate the preferred mode, but search for an *actually
            # executable* mode rather than trying a doomed random target.
            priority = [
                fault_modes[(i + slot + offset) % len(fault_modes)]
                for offset in range(len(fault_modes))
            ]
            chosen = None
            for mode in priority:
                for decision_round in rounds:
                    if decision_round in picked_rounds:
                        continue
                    wave = waves[decision_round - 2]
                    supported = [tid for tid in wave
                                 if mode in eligible.get(tid, [])]
                    if not supported:
                        continue
                    chosen = (mode, decision_round, supported)
                    break
                if chosen is not None:
                    break
            if chosen is None:
                break
            mode, decision_round, supported = chosen
            multitarget = (rng.random() < multi_target_probability
                           and len(supported) > 1)
            targets = choose_targets(
                supported, tasks, seed=seed + i * 103 + slot,
                multiple=multitarget,
            )
            faults = [
                Fault(task_id=tid, mode=mode,
                      seed=seed + i * 103 + slot,
                      persistence=persistence)
                for tid in targets
            ]
            planned.append({
                "index": len(planned), "scene_index": i,
                "task_id": scene["task_id"],
                "branch_round": decision_round, "faults": faults,
            })
            picked_rounds.add(decision_round)
            added += 1
        if not added:
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
        "main_action_generation_modes": dict(main_modes),
        "main_action_review_statuses": dict(main_statuses),
    }


def generate_turns(
    *, scenes_path: Path, reports_path: Path, base_episodes_path: Path,
    output_dir: Path,
    max_scenes: int | None = 30, branches_per_scene: int = 2,
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
    journal = output / "journal"
    journal.mkdir(exist_ok=True)
    pending = [
        plan for plan in planned
        if not (journal / f"{plan['index']:06d}.json").is_file()
    ]

    def worker(plan: dict) -> tuple[int, dict]:
        scene = scenes[plan["scene_index"]]
        try:
            episode, turns, audit = build_branch(
                scene, rows_by_scene[scene["task_id"]],
                base[scene["task_id"]],
                branch_round=plan["branch_round"], faults=plan["faults"],
                author=author, critic=critic,
                report_author=report_author, report_critic=report_critic,
                max_rounds=max_rounds, max_cost=max_cost,
                max_retries=max_retries,
                max_parallel_tasks=max_parallel_tasks,
                max_revisions=max_revisions,
                report_max_revisions=report_max_revisions,
                main_max_revisions=main_max_revisions,
                sub_model=sub_model,
                model_alias=model_alias,
                on_report_review=track_report,
                on_main_review=track_main,
            )
            status = audit["outcome"]
            data = {
                "task_id": scene["task_id"], "status": status,
                "episode": episode, "turns": turns,
                "private_audit": audit,
            }
        except FaultUnsupported as exc:
            data = {
                "task_id": scene["task_id"],
                "status": "skipped_unsupported", "error": str(exc)[:300],
                "episode": None, "turns": [], "private_audit": None,
            }
        except (ValueError, KeyError, TypeError) as exc:
            data = {
                "task_id": scene["task_id"],
                "status": "rejected_validation", "error": str(exc)[:300],
                "episode": None, "turns": [], "private_audit": None,
            }
        _atomic_json(journal / f"{plan['index']:06d}.json", data)
        return plan["index"], data

    # Dynamic recovery interleaves Sub-Agent and Main calls within each
    # branch. Two independent event-driven bars reflect actual authoring
    # work, not a fictitious sequential Stage-2-then-Stage-3 batch. Totals
    # are forecasts from the clean schedule and grow as recovery adds tasks.
    estimated_reports = 0
    estimated_main = 0
    for plan in planned:
        waves = scenes[plan["scene_index"]]["macro_graph"]["reference_waves"]
        remaining = waves[plan["branch_round"] - 1:]
        estimated_reports += (
            len(plan["faults"]) + sum(len(wave) for wave in remaining) + 1
        )
        estimated_main += len(remaining) + 2

    counts = Counter()
    report_counts = Counter()
    main_counts = Counter()
    completed = 0
    for plan in planned:
        path = journal / f"{plan['index']:06d}.json"
        if not path.is_file():
            continue
        saved = json.loads(path.read_text(encoding="utf-8"))
        counts[saved["status"]] += 1
        completed += 1
        audit = saved.get("private_audit") or {}
        for review in audit.get("subagent_report_reviews", []):
            result = review["authoring"]
            report_counts[result.get("status", "unknown")] += 1
            report_counts["written"] += 1
        for review in audit.get("main_action_reviews", []):
            result = review["authoring"]
            main_counts[result.get("status", "unknown")] += 1
            main_counts["written"] += 1
    display_lock = Lock()

    with tqdm(
        total=max(estimated_reports, report_counts["written"]),
        initial=report_counts["written"],
        desc="Sub-Agent feedback", unit="report", position=0,
        dynamic_ncols=True, disable=no_progress, leave=True,
    ) as report_bar, tqdm(
        total=max(estimated_main, main_counts["written"]),
        initial=main_counts["written"],
        desc="Main orchestration", unit="action", position=1,
        dynamic_ncols=True, disable=no_progress, leave=True,
    ) as main_bar:

        def refresh_postfix() -> None:
            report_bar.set_postfix({
                "accepted": report_counts["accepted"],
                "fallback": report_counts["fallback"],
                "written": report_counts["written"],
            }, refresh=False)
            main_bar.set_postfix({
                "accepted": main_counts["accepted"],
                "failed": main_counts["failed"],
                "written": main_counts["written"],
            }, refresh=False)

        def record(bar, counters, status: dict) -> None:
            # Called synchronously by a worker when the corresponding
            # independent Author/Guard/Critic finishes.
            with display_lock:
                counters[status.get("status", "unknown")] += 1
                counters["written"] += 1
                if bar.total is not None and bar.n + 1 >= bar.total and (
                    completed < len(planned)
                ):
                    bar.total = bar.n + max(2, len(planned) - completed)
                refresh_postfix()
                bar.update(1)

        def track_report(status: dict) -> None:
            record(report_bar, report_counts, status)

        def track_main(status: dict) -> None:
            record(main_bar, main_counts, status)

        refresh_postfix()
        with ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="untrusted-branch",
        ) as pool:
            futures = [pool.submit(worker, plan) for plan in pending]
            for future in as_completed(futures):
                _, result = future.result()
                with display_lock:
                    counts[result["status"]] += 1
                    completed += 1
                    refresh_postfix()

        # Forecasts are not exact for adaptive DAGs. Only at the end can
        # both bars display the exact number of performed calls as 100%.
        with display_lock:
            report_bar.total = report_bar.n
            main_bar.total = main_bar.n
            refresh_postfix()
            report_bar.refresh()
            main_bar.refresh()
    counts_summary = _publish(output, planned)
    summary = {
        "status": (
            "done" if counts_summary["recovered"] == len(planned)
            else "partial"
        ),
        "pipeline": "maestro_untrusted_orchestration_round_local_v1",
        "mode": ("llm" if main_mode == "llm" or report_mode == "llm"
                 else "template"),
        "subagent_report_mode": report_mode,
        "main_action_mode": main_mode,
        "input_scenes": len(scenes),
        "scenes_without_accepted_base": no_base,
        "scenes_without_eligible_fault": no_eligible,
        "requested_branches": len(planned),
        **counts_summary,
        "public_turns": str(output / "main_turns.jsonl"),
        "public_episodes": str(output / "episodes.jsonl"),
        "private_fault_alignment": str(
            output / "private" / "fault_alignment.jsonl"
        ),
        "limitations": [
            "One fault injection window per branch: the latest completed Sub-Agent batch.",
            "Only preflight-executable fault/macro pairs are sampled.",
            "Only publicly observable discrepancies trigger targeted repair labels.",
            "Clean World/Gold never change; uncorrected counterfactuals are private, not positive SFT.",
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
            f"{summary['completed_branches']}/{summary['requested_branches']} "
            f"| execution recovered={summary['execution_recovered_branches']} "
            f"| incorrect={summary['incorrect']} "
            f"| reviewed SFT branches={summary['recovered']} "
            f"| SFT turns={summary['main_training_turns']}"
        )
    return summary
