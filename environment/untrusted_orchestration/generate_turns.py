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

from tqdm.auto import tqdm

from .faults import Fault, FaultUnsupported, choose_targets
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
) -> tuple[list[dict], int]:
    planned, no_base = [], 0
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
            continue
        rng = random.Random(seed + i * 1009)
        rng.shuffle(rounds)
        for slot, t in enumerate(rounds[:branches_per_scene]):
            target_wave = scene["macro_graph"]["reference_waves"][t-2]
            tasks = {a["task_id"]: a for a in scene["macro_graph"]["tasks"]}
            multitarget = (
                rng.random() < multi_target_probability and
                len(target_wave) > 1
            )
            target = choose_targets(
                target_wave, tasks, seed=seed + i*103 + slot,
                multiple=multitarget,
            )
            mode = fault_modes[(i + slot) % len(fault_modes)]
            faults = [
                Fault(task_id=tid, mode=mode,
                      seed=seed + i*103 + slot,
                      persistence=persistence)
                for tid in target
            ]
            planned.append({
                "index": len(planned), "scene_index": i,
                "task_id": scene["task_id"],
                "branch_round": t, "faults": faults,
            })
    return planned, no_base


def _publish(output: Path, planned: list[dict]) -> dict:
    journals = output / "journal"
    episodes, positive_turns, audits, failures = [], [], [], []
    counters = Counter()
    for plan in planned:
        path = journals / f"{plan['index']:06d}.json"
        if not path.is_file():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if data["task_id"] != plan["task_id"]:
            raise RuntimeError("untrusted_journal_scene_mismatch")
        counters[data["status"]] += 1
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
            audits.append(data["private_audit"])
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
        "outcomes": dict(counters),
    }


def generate_turns(
    *, scenes_path: Path, reports_path: Path, base_episodes_path: Path,
    output_dir: Path,
    max_scenes: int | None = 30, branches_per_scene: int = 2,
    seed: int = 17, fault_modes: list[str] | None = None,
    persistence: str = "once", multi_target_probability: float = 0.0,
    max_rounds: int = 16, max_cost: int = 80, max_retries: int = 1,
    max_parallel_tasks: int = 4, workers: int = 4,
    max_revisions: int = 2, sub_model: str = "qwen3-8b",
    model_alias: str = "model_1", overwrite: bool = False,
    no_progress: bool = False,
    author=None, critic=None, report_author=None, report_critic=None,
    author_model: str | None = None, critic_model: str | None = None,
) -> dict:
    paths = tuple(map(Path, (scenes_path, reports_path, base_episodes_path)))
    output = Path(output_dir)
    if not all(path.is_file() for path in paths):
        raise FileNotFoundError("untrusted_requires_frozen_scenes_reports_base")
    if type(workers) is not int or not 1 <= workers <= 64:
        raise ValueError("untrusted_workers_invalid")
    if bool(author) != bool(critic) or bool(report_author) != bool(report_critic):
        raise ValueError("untrusted_author_and_critic_must_be_paired")
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
        "max_revisions": max_revisions, "sub_model": sub_model,
        "model_alias": model_alias, "author_model": author_model,
        "critic_model": critic_model,
        "mode": "llm" if author else "template",
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
    planned, no_base = _plan_branches(
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
                max_revisions=max_revisions, sub_model=sub_model,
                model_alias=model_alias,
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

    counts = Counter()
    with tqdm(total=len(planned),
              initial=len(planned)-len(pending),
              desc="Untrusted round branches", unit="branch",
              dynamic_ncols=True, disable=no_progress) as bar:
        for plan in planned:
            file = journal / f"{plan['index']:06d}.json"
            if file.is_file():
                saved = json.loads(file.read_text(encoding="utf-8"))
                counts[saved["status"]] += 1
        def progress():
            bar.set_postfix({
                "recovered": counts["recovered"],
                "unresolved": sum(x for key, x in counts.items()
                                  if key not in {
                                      "recovered", "skipped_unsupported",
                                      "rejected_validation",
                                  }),
                "skipped": counts["skipped_unsupported"] +
                           counts["rejected_validation"],
                "written": sum(counts.values()),
            })
        progress()
        with ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="untrusted-branch",
        ) as pool:
            futures = [pool.submit(worker, plan) for plan in pending]
            for future in as_completed(futures):
                _, result = future.result()
                counts[result["status"]] += 1
                bar.update(1)
                progress()
    counts_summary = _publish(output, planned)
    summary = {
        "status": (
            "done" if counts_summary["recovered"] == len(planned)
            else "partial"
        ),
        "pipeline": "maestro_untrusted_orchestration_round_local_v1",
        "mode": settings["mode"],
        "input_scenes": len(scenes),
        "scenes_without_accepted_base": no_base,
        "requested_branches": len(planned),
        **counts_summary,
        "public_turns": str(output / "main_turns.jsonl"),
        "public_episodes": str(output / "episodes.jsonl"),
        "private_fault_alignment": str(
            output / "private" / "fault_alignment.jsonl"
        ),
        "limitations": [
            "One fault injection window per branch: the latest completed Sub-Agent batch.",
            "Only publicly observable discrepancies trigger targeted repair labels.",
            "Clean World/Gold never change; uncorrected counterfactuals are private, not positive SFT.",
            "Template actions are deterministic teacher demonstrations, not online Main-Agent rollouts.",
        ],
    }
    _atomic_json(output / "summary.json", summary)
    return summary
