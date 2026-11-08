"""Independent MMQA-guided synthetic scenario pilot.

Usage: python -m envir.scenarios.cli build --mmqa ... --output-dir ...
Does not modify the existing envir SQLite bank or legacy environment.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .generator import generate_scenes
from .mmqa_motifs import load_motifs
from .runtime import SimulatedRuntime
from .validation import validate_scene


def _write_json(path: Path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def build(args) -> dict:
    if args.target < 1 or args.max_records < 1:
        raise ValueError("target and max-records must be positive")
    if not Path(args.mmqa).is_file():
        raise FileNotFoundError(args.mmqa)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = [out / name for name in (
        "mmqa_motifs.jsonl", "scenes_gold.jsonl",
        "public_tasks.jsonl", "scenario_index.json", "simulation_summary.json",
    )]
    if not args.overwrite and any(path.exists() for path in paths):
        raise FileExistsError(
            "scenario outputs already exist; use a new --output-dir "
            "or explicitly pass --overwrite"
        )
    motifs, motif_stats = load_motifs(args.mmqa, limit=args.max_records)
    scenes, scene_stats = generate_scenes(
        motifs, target=args.target, seed=args.seed,
        extensions=not args.no_media_extensions,
    )
    _jsonl(paths[0], motifs)
    _jsonl(paths[1], scenes)  # PRIVATE teacher-side simulation worlds.
    _jsonl(paths[2], [SimulatedRuntime(s).public_task() for s in scenes])
    _write_json(paths[3], {
        "scene_ids": [s["scene_id"] for s in scenes],
        "validation": {
            s["scene_id"]: "verified" if not validate_scene(s) else "failed"
            for s in scenes
        },
    })
    result = {
        "status": "done", "requested": args.target, "generated": len(scenes),
        "motif_statistics": motif_stats, "scene_statistics": scene_stats,
        "verification_policy": (
            "simulated observation and consequence replay, "
            "NOT real media perception or real-world fact verification"
        ),
        "files": {
            "motifs": str(paths[0]), "teacher_private": str(paths[1]),
            "public": str(paths[2]), "index": str(paths[3]),
            "summary": str(paths[4]),
        },
    }
    _write_json(paths[4], result)
    return result


def replay(args) -> dict:
    failures = []
    checked = 0
    with Path(args.gold).open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            checked += 1
            scene = json.loads(line)
            errors = validate_scene(scene)
            if errors:
                failures.append({"scene_id": scene.get("scene_id"),
                                 "errors": errors})
    return {"status": "valid" if not failures else "invalid",
            "checked": checked, "failures": failures[:20],
            "failure_count": len(failures)}


def tool_demo(args) -> dict:
    scene = None
    with Path(args.gold).open("r", encoding="utf-8") as fh:
        for line in fh:
            item = json.loads(line)
            if item["scene_id"] == args.scene_id:
                scene = item
                break
    if scene is None:
        raise KeyError(f"scene not found: {args.scene_id}")
    runtime = SimulatedRuntime(scene)
    result = runtime.call(
        args.action, source_id=args.source_id, query=args.query,
    )
    return {"tool_observation": result, "cost_units": runtime.total_cost}


def parser():
    p = argparse.ArgumentParser(
        description="MMQA-inspired simulated media scenarios for orchestration"
    )
    sub = p.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="Extract motifs and build simulated scenes")
    b.add_argument("--mmqa", type=Path, required=True)
    b.add_argument("--output-dir", type=Path,
                   default=Path("envir/data/sim_scenarios_v1"))
    b.add_argument("--target", type=int, default=100)
    b.add_argument("--max-records", type=int, default=3000)
    b.add_argument("--seed", type=int, default=17)
    b.add_argument("--no-media-extensions", action="store_true")
    b.add_argument("--overwrite", action="store_true",
                   help="Explicitly overwrite scenario outputs")
    v = sub.add_parser("replay", help="Verify every synthetic teacher-side fact")
    v.add_argument("--gold", type=Path, required=True)
    demo = sub.add_parser("tool", help="Reveal one simulated tool observation")
    demo.add_argument("--gold", type=Path, required=True)
    demo.add_argument("--scene-id", required=True)
    demo.add_argument("--action", required=True,
                      choices=("InspectSimulatedMedia","ReadSimulatedDocument",
                               "SimulatedSearch","OpenSimulatedPage"))
    demo.add_argument("--source-id")
    demo.add_argument("--query", default="")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    result = {"build": build, "replay": replay,
              "tool": tool_demo}[args.command](args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("status", "done") not in ("invalid",) else 2


if __name__ == "__main__":
    raise SystemExit(main())
