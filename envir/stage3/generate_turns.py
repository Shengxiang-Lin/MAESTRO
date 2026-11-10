"""Stream and resume reviewed Main-Agent turn datasets from Stage 1/2.

A per-scene atomic journal is the crash-safe source of truth. Public
turns/episodes and PRIVATE alignment files are reconstructed from journals.
No scenario failed by an LLM Critic is silently turned into a pass.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import shutil

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


def generate_turns(
    *, scenes_path: Path, reports_path: Path, output_dir: Path,
    max_scenes: int | None = None,
    author=None, critic=None, max_revisions: int = 2,
    max_attempts: int = 16, sub_model: str = "qwen3-8b",
    model_alias: str = "model_1",
    author_model: str | None = None,
    critic_model: str | None = None,
    overwrite: bool = False,
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
    reports = defaultdict(list)
    for row in read_jsonl(reports_path):
        reports[row["scene_task_id"]].append(row)
    seen = set()
    counts = Counter()
    for i, scene in enumerate(scene_rows):
        task_id = scene["task_id"]
        if task_id in seen:
            raise ValueError("stage3_duplicate_scene_task_id")
        seen.add(task_id)
        filename = journal / f"{i:06d}.json"
        if filename.exists():
            data = json.loads(filename.read_text(encoding="utf-8"))
            if data["task_id"] != task_id:
                raise ValueError("stage3_journal_scene_order_changed")
            counts[data["status"]["status"]] += 1
            continue
        try:
            episode, turns, alignments, status = build_episode(
                scene, reports[task_id],
                author=author, critic=critic,
                max_revisions=max_revisions, max_attempts=max_attempts,
                sub_model=sub_model, model_alias=model_alias,
            )
        except (ValueError, KeyError, TypeError) as exc:
            episode, turns, alignments = None, [], []
            status = {
                "status": "rejected",
                "cause": "validation_error",
                "error": str(exc)[:250],
            }
        data = {
            "task_id": task_id, "status": status,
            "episode": episode, "turns": turns, "private_audit": alignments,
        }
        _dump_json(filename, data)
        counts[status["status"]] += 1

    episodes, turns, private, failures = [], [], [], []
    for i, scene in enumerate(scene_rows):
        data = json.loads((journal / f"{i:06d}.json").read_text(
            encoding="utf-8"
        ))
        if data["task_id"] != scene["task_id"]:
            raise ValueError("stage3_journal_mismatch_at_merge")
        if data["status"]["status"] == "accepted":
            episodes.append(data["episode"])
            turns.extend(data["turns"])
            private.extend(data["private_audit"])
        else:
            failures.append({"task_id": scene["task_id"], **data["status"]})
    _write_jsonl(output_dir / "episodes.jsonl", episodes)
    _write_jsonl(output_dir / "main_turns.jsonl", turns)
    _write_jsonl(output_dir / "private" / "teacher_alignment.jsonl", private)
    _write_jsonl(output_dir / "failed.jsonl", failures)
    action_counts = Counter(row["output"]["action"] for row in turns)
    summary = {
        "status": "done" if not failures else "partial",
        "pipeline": "maestro_stage3_main_turn_dataset_v1",
        "mode": mode,
        "requested_scenes": len(scene_rows),
        "accepted_scenes": len(episodes),
        "rejected_scenes": len(failures),
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
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "done" else 1


if __name__ == "__main__":
    raise SystemExit(main())
