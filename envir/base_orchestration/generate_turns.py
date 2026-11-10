"""Stream and resume reviewed Main-Agent turn datasets from Stage 1/2.

A per-scene atomic journal is the crash-safe source of truth. Public
turns/episodes and PRIVATE alignment files are reconstructed from journals.
No scenario failed by an LLM Critic is silently turned into a pass.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
from threading import current_thread

from tqdm import tqdm

from envir.generate_question.stages import read_jsonl
from .trajectory_builder import build_episode


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _dump_json(path: Path, value: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    os.replace(temp, path)


def _write_jsonl(path: Path, data: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as stream:
        for row in data:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temp, path)


def _signature(scenes: Path, reports: Path, *, max_scenes, sub_model,
               model_alias, max_attempts, max_revisions, mode,
               author_model, critic_model) -> dict:
    root = Path(__file__).parent
    tracked = (
        "trajectory_builder.py", "action_author.py",
        "o1_compat.py", "generate_turns.py",
        "../generate_question/evidence_tools.py",
        "../../orchestra/base_main_agent.py",
        "../../orchestra/prompts/omnigaia.py",
        "../../benchmark/omnigaia/tools/google_search.py",
        "../../benchmark/omnigaia/tools/extract_url_jina.py",
        "../../benchmark/omnigaia/tools/multimodal_toolkit.py",
        "../../benchmark/omnigaia/tools/audio_analysis.py",
        "../../benchmark/omnigaia/tools/video_analysis.py",
        "../../benchmark/omnigaia/tools/execute_code.py",
    )
    return {
        "scenes_sha256": _digest(scenes),
        "reports_sha256": _digest(reports),
        "code_sha256": {p: _digest(root / p) for p in tracked},
        "max_scenes": max_scenes,
        "sub_model": sub_model,
        "model_alias": model_alias,
        "max_attempts": max_attempts,
        "max_revisions": max_revisions,
        "mode": mode,
        "author_model": author_model,
        "critic_model": critic_model,
    }


def _round_directory(journal: Path, index: int) -> Path:
    """Incomplete rounds never enter the published training dataset."""
    return journal / f"{index:06d}.rounds"


def _load_rounds(directory: Path, *, task_id: str) -> list[dict]:
    if not directory.exists():
        return []
    if not directory.is_dir():
        raise RuntimeError("stage3_checkpoint_path_not_directory")
    paths = sorted(directory.glob("round-*.json"))
    saved = []
    for number, path in enumerate(paths, 1):
        if path.name != f"round-{number:04d}.json":
            raise RuntimeError("stage3_checkpoint_round_gap_or_bad_name")
        row = json.loads(path.read_text(encoding="utf-8"))
        if (row.get("round") != number
                or row.get("turn", {}).get("task_id") != task_id):
            raise RuntimeError("stage3_checkpoint_wrong_round_or_scene")
        saved.append(row)
    return saved


def _save_round(directory: Path, checkpoint: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"round-{checkpoint['round']:04d}.json"
    if path.exists():
        raise RuntimeError("stage3_checkpoint_would_overwrite_existing_round")
    _dump_json(path, checkpoint)


def _append_jsonl(path: Path, records: list[dict]) -> None:
    """Publish complete episodes incrementally; journals repair torn appends."""
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        for item in records:
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _rebuild_publication(
    scenes: list[dict], journal: Path, output_dir: Path,
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    """Scene journals are authoritative across interrupted multi-file appends."""
    episodes, turns, private, failures = [], [], [], []
    for index, scene in enumerate(scenes):
        file = journal / f"{index:06d}.json"
        if not file.is_file():
            continue
        saved = json.loads(file.read_text(encoding="utf-8"))
        if saved.get("task_id") != scene["task_id"]:
            raise RuntimeError("stage3_journal_scene_order_changed")
        if saved["status"]["status"] == "accepted":
            episodes.append(saved["episode"])
            turns.extend(saved["turns"])
            private.extend(saved["private_audit"])
        else:
            failures.append({
                "task_id": scene["task_id"], **saved["status"],
            })
    _write_jsonl(output_dir / "episodes.jsonl", episodes)
    _write_jsonl(output_dir / "main_turns.jsonl", turns)
    _write_jsonl(output_dir / "private" / "teacher_alignment.jsonl", private)
    _write_jsonl(output_dir / "failed.jsonl", failures)
    return episodes, turns, private, failures


def generate_turns(
    *, scenes_path: Path, reports_path: Path, output_dir: Path,
    max_scenes: int | None = None,
    author=None, critic=None, max_revisions: int = 2,
    max_attempts: int = 16, sub_model: str = "qwen3-8b",
    model_alias: str = "model_1",
    author_model: str | None = None,
    critic_model: str | None = None,
    overwrite: bool = False, no_progress: bool = False,
    verbose_progress: bool = False, workers: int = 4,
) -> dict:
    scenes_path, reports_path = Path(scenes_path), Path(reports_path)
    output_dir = Path(output_dir)
    if not scenes_path.is_file() or not reports_path.is_file():
        raise FileNotFoundError("stage3_requires_stage1_scenes_and_stage2_reports")
    if max_scenes is not None and (
        type(max_scenes) is not int or max_scenes < 1
    ):
        raise ValueError("stage3_max_scenes_must_be_positive")
    if type(max_revisions) is not int or not 0 <= max_revisions <= 8:
        raise ValueError("stage3_max_revisions_must_be_0_to_8")
    if bool(author) != bool(critic):
        raise ValueError("stage3_requires_both_author_and_critic")
    if type(workers) is not int or not 1 <= workers <= 64:
        raise ValueError("stage3_workers_must_be_between_1_and_64")
    mode = "llm_reviewed" if author is not None else "template"
    signature = _signature(
        scenes_path, reports_path, max_scenes=max_scenes,
        sub_model=sub_model, model_alias=model_alias,
        max_attempts=max_attempts, max_revisions=max_revisions, mode=mode,
        author_model=author_model, critic_model=critic_model,
    )
    if output_dir.exists() and overwrite:
        if output_dir.resolve() in {
            scenes_path.parent.resolve(), reports_path.parent.resolve(),
            scenes_path.resolve(), reports_path.resolve(),
        }:
            raise ValueError("stage3_output_overwrites_input")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = output_dir / "manifest.json"
    if manifest.exists():
        if json.loads(manifest.read_text(encoding="utf-8")) != signature:
            raise ValueError("stage3_resume_input_or_code_signature_mismatch")
    elif any(output_dir.iterdir()):
        raise ValueError("stage3_existing_output_without_manifest")
    else:
        _dump_json(manifest, signature)
    journal = output_dir / "journal"
    journal.mkdir(exist_ok=True)
    scene_rows = read_jsonl(scenes_path, limit=max_scenes)
    ids = [row["task_id"] for row in scene_rows]
    if len(set(ids)) != len(ids):
        raise ValueError("stage3_duplicate_scene_task_id")
    reports = defaultdict(list)
    for row in read_jsonl(reports_path):
        reports[row["scene_task_id"]].append(row)

    # Only the coordinator writes shared public JSONL files. Each worker
    # owns one Scene's round checkpoints and final atomic Scene journal.
    # This isolates concurrent generation from publication, and preserves
    # the prior restart contract for partially completed episodes.
    old_episodes, old_turns, _, old_failures = _rebuild_publication(
        scene_rows, journal, output_dir,
    )
    counts = Counter({"accepted": len(old_episodes),
                      "rejected": len(old_failures)})
    # Only turns from fully accepted Episodes are public SFT samples.
    published_turns = len(old_turns)
    pending = [
        (i, scene) for i, scene in enumerate(scene_rows)
        if not (journal / f"{i:06d}.json").is_file()
    ]

    def process_scene(i: int, scene: dict):
        """Execute a whole Scene serially inside one independent worker."""
        task_id = scene["task_id"]
        filename = journal / f"{i:06d}.json"
        rounds_dir = _round_directory(journal, i)
        saved_rounds = _load_rounds(rounds_dir, task_id=task_id)
        round_bar = None

        def on_episode_start(total: int, restored: int) -> None:
            nonlocal round_bar
            # Distinct ThreadPoolExecutor worker threads keep stable tqdm
            # positions; previous scenes can reuse the same line.
            slot = int(current_thread().name.rsplit("_", 1)[1])
            if verbose_progress and not no_progress:
                round_bar = tqdm(
                    total=total, initial=restored,
                    desc=f"  Worker {slot + 1}: scene {i + 1}/{len(scene_rows)}",
                    unit="round", position=slot + 1, leave=False,
                )

        def on_round(checkpoint: dict) -> None:
            # Persist before issuing any subsequent Author/Critic requests.
            _save_round(rounds_dir, checkpoint)
            if round_bar is not None:
                round_bar.update(1)

        try:
            try:
                episode, turns, alignments, status = build_episode(
                    scene, reports[task_id],
                    author=author, critic=critic,
                    max_revisions=max_revisions, max_attempts=max_attempts,
                    sub_model=sub_model, model_alias=model_alias,
                    resume_rounds=saved_rounds, on_round=on_round,
                    on_episode_start=on_episode_start,
                )
            except (ValueError, KeyError, TypeError) as exc:
                episode, turns, alignments = None, [], []
                status = {
                    "status": "rejected",
                    "cause": "validation_error",
                    "error": str(exc)[:250],
                }
        finally:
            if round_bar is not None:
                round_bar.close()

        data = {
            "task_id": task_id, "status": status,
            "episode": episode, "turns": turns,
            "private_audit": alignments,
        }
        # Crash-safe per-scene commit. Nothing incomplete is published.
        _dump_json(filename, data)
        return i, data

    def write_live_summary() -> None:
        # Persist compact run-level counters after *each* completed Scene,
        # including while other workers are still generating round decisions.
        current = len(scene_rows) - len(pending) + completed_now
        _dump_json(output_dir / "summary.json", {
            "status": "running" if current < len(scene_rows) else "finalizing",
            "pipeline": "maestro_stage3_main_turn_dataset_v1",
            "mode": mode,
            "workers": workers,
            "requested_scenes": len(scene_rows),
            "processed_scenes": current,
            "accepted_scenes": counts["accepted"],
            "rejected_scenes": counts["rejected"],
            "written_scenes": counts["accepted"] + counts["rejected"],
            "main_training_turns": published_turns,
            "public_turns": str(output_dir / "main_turns.jsonl"),
            "failed": str(output_dir / "failed.jsonl"),
        })

    completed_now = 0
    write_live_summary()
    with tqdm(total=len(scene_rows), initial=len(scene_rows) - len(pending),
              desc="Base orchestration scenes", unit="scene", position=0,
              dynamic_ncols=True, disable=no_progress) as outer:
        def update_progress() -> None:
            # Match Stage 2's live accepted/rejected/written counters.
            # "written" counts committed Scenes, while "turns" counts
            # publishable Main decisions from accepted complete Episodes.
            outer.set_postfix({
                "accepted": counts["accepted"],
                "rejected": counts["rejected"],
                "written": counts["accepted"] + counts["rejected"],
                "turns": published_turns,
            })

        update_progress()
        # LLM calls in different Scenes overlap; rounds within one Scene do not.
        # The standard OpenAICompatibleBackend keeps no per-request mutable
        # state and can be shared between workers.
        with ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="base-orchestration",
        ) as pool:
            futures = {
                pool.submit(process_scene, i, scene): i
                for i, scene in pending
            }
            for future in as_completed(futures):
                i, data = future.result()  # Surface corrupt journals/fatal errors.
                status = data["status"]
                if status["status"] == "accepted":
                    _append_jsonl(
                        output_dir / "episodes.jsonl", [data["episode"]],
                    )
                    _append_jsonl(
                        output_dir / "main_turns.jsonl", data["turns"],
                    )
                    _append_jsonl(
                        output_dir / "private" / "teacher_alignment.jsonl",
                        data["private_audit"],
                    )
                    published_turns += len(data["turns"])
                else:
                    _append_jsonl(output_dir / "failed.jsonl", [
                        {"task_id": data["task_id"], **status},
                    ])
                counts[status["status"]] += 1
                completed_now += 1
                outer.update(1)
                update_progress()
                write_live_summary()

    # Normalize all published outputs from atomic scene journals. A restart
    # does this on entry too, so torn appends cannot duplicate any samples.
    episodes, turns, private, failures = _rebuild_publication(
        scene_rows, journal, output_dir,
    )
    action_counts = Counter(row["output"]["action"] for row in turns)
    failure_causes = Counter(
        row.get("cause", "unknown") for row in failures
    )
    failure_stages = Counter()
    guard_issues = Counter()
    for failure in failures:
        for attempt in failure.get("authoring", {}).get("failures", []):
            failure_stages[attempt.get("stage", "unknown")] += 1
            guard_issues.update(attempt.get("errors") or [])
    summary = {
        "status": "done" if not failures else "partial",
        "pipeline": "maestro_stage3_main_turn_dataset_v1",
        "mode": mode,
        "workers": workers,
        "requested_scenes": len(scene_rows),
        "accepted_scenes": len(episodes),
        "rejected_scenes": len(failures),
        "rejection_causes": dict(failure_causes),
        "rejection_attempt_stages": dict(failure_stages),
        "guard_errors_across_attempts": dict(guard_issues),
        "main_training_turns": len(turns),
        "actions": dict(action_counts),
        "mean_turns_per_scene": round(
            len(turns) / max(1, len(episodes)), 3
        ),
        "parallel_delegation_rounds": sum(
            row["output"]["action"] == "delegate_task"
            and len(row["output"]["params"]["tasks"]) > 1
            for row in turns
        ),
        "public_turns": str(output_dir / "main_turns.jsonl"),
        "public_episodes": str(output_dir / "episodes.jsonl"),
        "private_teacher_alignment": str(
            output_dir / "private" / "teacher_alignment.jsonl"
        ),
        "failed": str(output_dir / "failed.jsonl"),
        "input_storage": "cumulative_o1_subtask_history_only",
        "prompt_reconstruction": "scene + round + manifest + baseline_o1_renderer",
        "checkpoint_granularity": "main_decision_round",
        "publication_granularity": "complete_episode",
        "limitations": [
            "Teacher-controlled, successful trusted trajectories are not Main-Agent online rollouts.",
            "Template mode is a deterministic format baseline, not independently reviewed natural language.",
            "The actual O1 prompt/history renderer is reused; simulated workspace tools are declarative, never executed.",
            "Gold DAG and Oracle results are excluded from the public per-turn input and output.",
        ],
    }
    _dump_json(output_dir / "summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    import argparse
    import os
    from envir.generate_question.backend import OpenAICompatibleBackend

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenes", type=Path, required=True)
    p.add_argument("--reports", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--max-scenes", type=int)
    p.add_argument("--workers", type=int, default=4,
                   help="Concurrent independent Scenes (default: 4)")
    p.add_argument("--mode", choices=("template", "llm"), default="template")
    p.add_argument("--author-api-base")
    p.add_argument("--author-model")
    p.add_argument("--critic-api-base")
    p.add_argument("--critic-model")
    p.add_argument("--api-key-env", default="OPENAI_API_KEY")
    p.add_argument("--temperature", type=float, default=0.4)
    p.add_argument("--max-revisions", type=int, default=2)
    p.add_argument("--max-attempts", type=int, default=16)
    p.add_argument("--sub-model", default="qwen3-8b")
    p.add_argument("--model-alias", default="model_1")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--no-progress", action="store_true",
                   help="Disable progress bars")
    p.add_argument("--verbose-progress", action="store_true",
                   help="Show per-worker Main-round progress bars (default: off)")
    args = p.parse_args(argv)
    author = critic = None
    if args.mode == "llm":
        if not args.author_api_base or not args.author_model:
            p.error("--mode llm requires --author-api-base and --author-model")
        key = os.getenv(args.api_key_env)
        author = OpenAICompatibleBackend(
            base_url=args.author_api_base, model=args.author_model,
            api_key=key, temperature=args.temperature, max_tokens=2800,
        )
        critic = OpenAICompatibleBackend(
            base_url=args.critic_api_base or args.author_api_base,
            model=args.critic_model or args.author_model,
            api_key=key, temperature=0.0, max_tokens=1200,
        )
    summary = generate_turns(
        scenes_path=args.scenes, reports_path=args.reports,
        output_dir=args.output_dir, max_scenes=args.max_scenes,
        author=author, critic=critic, max_revisions=args.max_revisions,
        max_attempts=args.max_attempts, sub_model=args.sub_model,
        model_alias=args.model_alias,
        author_model=args.author_model, critic_model=args.critic_model,
        overwrite=args.overwrite, no_progress=args.no_progress,
        verbose_progress=args.verbose_progress, workers=args.workers,
    )
    print(f"Base orchestration: {summary['status']} | details: "
          f"{args.output_dir}/summary.json")
    return 0 if summary["status"] == "done" else 1


if __name__ == "__main__":
    raise SystemExit(main())
