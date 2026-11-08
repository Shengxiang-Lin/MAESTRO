"""CLI for Subtask-First prototype: prepare / compose / inspect / search."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .composition import TaskGraphComposer
from .sources.kqa import ingest_kqa
from .sources.mmqa import ingest_mmqa
from .sources.yago import ingest_yago
from .store import EvidenceStore
from .validation import corpus_statistics


def _dump(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                    encoding="utf-8")


def prepare(args) -> dict:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for value in (args.kqa, args.yago, args.mmqa):
        if not Path(value).is_file():
            raise FileNotFoundError(f"required dataset file not found: {value}")
    bank_path = out / "bank.sqlite"
    if bank_path.is_file():
        if not args.restart_bank:
            raise FileExistsError(
                "source bank already exists; use a new --output-dir or "
                "--restart-bank explicitly"
            )
        for suffix in ("", "-wal", "-shm"):
            (out / ("bank.sqlite" + suffix)).unlink(missing_ok=True)
    with EvidenceStore(bank_path) as store:
        yago = ingest_yago(
            store, args.yago, max_triples=args.max_yago_triples,
            max_subjects=args.max_yago_subjects,
        )
        kqa = ingest_kqa(store, args.kqa, max_records=args.max_kqa_records)
        mmqa = ingest_mmqa(store, args.mmqa, max_records=args.max_mmqa_records)
        templates = yago.pop("templates")
        _dump(out / "subtask_templates.json", templates)
        summary = {
            "status": "prepared",
            "bank": str(out / "bank.sqlite"),
            "kqa": kqa, "yago": yago, "mmqa": mmqa,
            "store": store.stats(),
            "verification_policy": {
                "YAGO": "observed_fact_and_public_search_replay",
                "MultiModalQA": "official_dataset_label_only",
                "MMQA_media": "media_file_or_caption_proxy_marked_explicitly",
            },
        }
        _dump(out / "prepare_summary.json", summary)
        return summary


def compose(args) -> dict:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    db = Path(args.bank)
    if not db.is_file():
        raise FileNotFoundError(f"prepare the source/subtask bank first: {db}")
    with EvidenceStore(db) as store:
        builder = TaskGraphComposer(store, seed=args.seed)
        graphs = builder.build(
            target=args.target, max_attempts=args.max_attempts,
            patterns=tuple(args.patterns.split(",")),
        )
        from .validation import check_graph
        rejected = []
        for graph in graphs:
            problems = check_graph(graph, store)
            if problems:
                rejected.append({"task_id": graph["task_id"], "issues": problems})
        if rejected:
            raise RuntimeError("postflight graph replay failed: " + repr(rejected[:3]))
        # This file contains private answers/traces. NEVER use it as Main input.
        with (out / "gold_graphs.jsonl").open("w", encoding="utf-8") as fh:
            for graph in graphs:
                fh.write(json.dumps(graph, ensure_ascii=False) + "\n")
        # Only task question and public material references; NO Gold answer,
        # future subtask identifiers, dependency edges or trace.
        with (out / "public_questions.jsonl").open("w", encoding="utf-8") as fh:
            for graph in graphs:
                public_sources = sorted({
                    sid for node in graph["nodes"]
                    if node.get("kind") == "mmqa_qa"
                    for sid in node["evidence_ids"]
                })
                fh.write(json.dumps({
                    "task_id": graph["task_id"],
                    "question": graph["question"],
                    "provided_source_ids": public_sources,
                    "available_tools": ["LocalSearch", "OpenPage", "ReadProvidedSource"],
                }, ensure_ascii=False) + "\n")
        store.db.commit()
        stats = corpus_statistics(graphs)
        summary = {
            "status": "done" if graphs else "no_graphs",
            "requested": args.target,
            "generated": len(graphs),
            "statistics": stats,
            "rejections": dict(builder.rejections),
            "store": store.stats(),
            "files": {
                "gold": str(out / "gold_graphs.jsonl"),
                "public": str(out / "public_questions.jsonl"),
                "summary": str(out / "compose_summary.json"),
            },
        }
        _dump(out / "compose_summary.json", summary)
        return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="MAESTRO envir: independent source-backed Subtask-First prototype"
    )
    sub = p.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="Build source index and labeled subtask pool")
    prep.add_argument("--kqa", type=Path,
                      default=Path("data/world-source/KQA-Pro/train.json"))
    prep.add_argument("--yago", type=Path,
                      default=Path("data/world-source/YAGO-4.6/yago-facts.ttl"))
    prep.add_argument("--mmqa", type=Path,
                      default=Path("data/world-source/MultiModalQA/MMQA_train.jsonl.gz"))
    prep.add_argument("--output-dir", type=Path, default=Path("envir/data/pilot"))
    prep.add_argument("--max-yago-triples", type=int, default=300000)
    prep.add_argument("--max-yago-subjects", type=int, default=100000)
    prep.add_argument("--max-kqa-records", type=int, default=10000)
    prep.add_argument("--max-mmqa-records", type=int, default=2000)
    prep.add_argument("--restart-bank", action="store_true",
                      help="Deliberately discard the previous envir bank")

    comp = sub.add_parser("compose", help="Ground and replay task graphs lazily")
    comp.add_argument("--bank", type=Path, default=Path("envir/data/pilot/bank.sqlite"))
    comp.add_argument("--output-dir", type=Path, default=Path("envir/data/pilot"))
    comp.add_argument("--target", type=int, default=20)
    comp.add_argument("--max-attempts", type=int, default=200)
    comp.add_argument("--seed", type=int, default=17)
    comp.add_argument("--patterns", type=str,
                      default="chain,parallel,media_to_search")

    inst = sub.add_parser("instantiate",
                          help="Lazily verify a specific real YAGO subtask")
    inst.add_argument("--bank", type=Path, required=True)
    inst.add_argument("--subject", required=True,
                      help="Exact YAGO subject URI/QName from the source facts")
    inst.add_argument("--predicate", required=True,
                      help="Exact YAGO predicate URI/QName")
    inspect = sub.add_parser("inspect", help="Inspect pool without generation")
    inspect.add_argument("--bank", type=Path, required=True)
    search = sub.add_parser("search", help="Inspect the public local-search behavior")
    search.add_argument("--bank", type=Path, required=True)
    search.add_argument("--query", required=True)
    search.add_argument("--limit", type=int, default=10)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        if min(args.max_yago_triples, args.max_yago_subjects,
               args.max_kqa_records, args.max_mmqa_records) < 1:
            raise SystemExit("all record limits must be positive")
        result = prepare(args)
    elif args.command == "compose":
        result = compose(args)
    elif args.command == "instantiate":
        from .subtasks import SubtaskFactory
        with EvidenceStore(args.bank) as store:
            subtask = SubtaskFactory(store).yago(args.subject, args.predicate)
            store.db.commit()
            result = {"status": "verified" if subtask else "unavailable",
                      "subtask": subtask}
    else:
        with EvidenceStore(args.bank) as store:
            result = store.stats() if args.command == "inspect" else store.search(
                args.query, limit=args.limit,
            )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 2 if isinstance(result, dict) and result.get("status") in {"no_graphs","unavailable"} else 0


if __name__ == "__main__":
    raise SystemExit(main())
