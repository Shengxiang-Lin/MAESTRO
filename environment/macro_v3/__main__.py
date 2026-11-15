"""CLI for Stage-1 v3 experimental Macro-first data generation."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from .composer import compose_scene, replay_scene
from .yago import load_yago


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="YAGO-grounded Macro-first v3 pilot")
    p.add_argument("--yago", type=Path, default=Path("data/world-source/YAGO-4.6"))
    p.add_argument("--output-dir", type=Path, default=Path("environment/data/stage1_v3_pilot"))
    p.add_argument("--target", type=int, default=100)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--max-triples", type=int, default=250_000)
    p.add_argument("--max-edges", type=int, default=70_000)
    p.add_argument("--min-hops", type=int, default=1)
    p.add_argument("--max-hops", type=int, default=3)
    p.add_argument("--max-branches", type=int, default=4)
    p.add_argument("--overwrite", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.target < 1:
        raise ValueError("target must be positive")
    out = args.output_dir
    scene_file = out / "drafts" / "scenes.jsonl"
    summary_file = out / "summary.json"
    if (scene_file.exists() or summary_file.exists()) and not args.overwrite:
        raise FileExistsError(f"{out} already has v3 output; use --overwrite")
    graph = load_yago(args.yago, max_triples=args.max_triples,
                      max_edges=args.max_edges)
    out.joinpath("drafts").mkdir(parents=True, exist_ok=True)
    failures: Counter[str] = Counter()
    stats: Counter[str] = Counter()
    accepted = 0
    # Bounded retry: malformed/too-short graph walks do not silently become output.
    with scene_file.open("w", encoding="utf-8") as stream:
        for attempt in range(args.target * 10):
            if accepted >= args.target:
                break
            try:
                scene = compose_scene(
                    graph, seed=args.seed, ordinal=attempt,
                    min_hops=args.min_hops, max_hops=args.max_hops,
                    max_branches=args.max_branches,
                )
                replay_scene(scene, graph)
            except (ValueError, KeyError, TypeError) as exc:
                failures[type(exc).__name__ + ":" + str(exc)[:110]] += 1
                continue
            stream.write(json.dumps(scene, ensure_ascii=False) + "\n")
            stream.flush()
            accepted += 1
            stats["macro_count:" + str(scene["macro_graph"]["macro_task_count"])] += 1
            stats["branch_count:" + str(scene["quality_report"]["branch_count"])] += 1
    summary = {
        "status": "pilot_done" if accepted == args.target else "pilot_partial",
        "contract": "maestro.graph_grounded_scene.v3",
        "stage2_compatible": False,
        "notes": "These are deterministic Question drafts, not LLM-reviewed Stage-1 scenes.",
        "requested": args.target, "written": accepted,
        "yago": {"path": str(args.yago), "sample_file": graph.input_file,
                 "scanned_triples": graph.scanned, "retained_edges": len(graph.edges)},
        "stats": dict(stats), "failures": dict(failures),
        "output": str(scene_file),
    }
    summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if accepted == args.target else 1


if __name__ == "__main__":
    raise SystemExit(main())
