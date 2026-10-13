"""Summarize Orchestra-o1 local-Qwen OmniGAIA evaluation results."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path

TOTAL_SAMPLES = 360
DEFAULT_MAIN_AGENT = "qwen3-8b"
HERE = Path(__file__).resolve().parent
BASELINE_ROOT = HERE.parent
REPO_ROOT = BASELINE_ROOT.parents[1]


def find_latest_csv(output_dir: Path) -> Path | None:
    files = list(output_dir.glob("omnigaia_qwen_*.csv"))
    if not files:
        files = list(output_dir.glob("omnigaia_orchestra_o1_*.csv"))
    return max(files, key=lambda path: path.stat().st_mtime) if files else None


def parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.strip())
    except ValueError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OmniGAIA Orchestra-o1 Qwen evaluation summary"
    )
    parser.add_argument("--csv_path", type=str, default=None)
    parser.add_argument("--main_agent", type=str, default=DEFAULT_MAIN_AGENT)
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory used when --csv_path is omitted.",
    )
    args = parser.parse_args()

    if args.csv_path:
        csv_path = Path(args.csv_path)
    else:
        output_dir = (
            Path(args.output_dir)
            if args.output_dir
            else REPO_ROOT / "logs/omnigaia_qwen"
        )
        csv_path = find_latest_csv(output_dir)
        if csv_path is None:
            print(f"Error: no evaluation CSV found under {output_dir}")
            return

    rows = []
    with csv_path.open("r", encoding="utf-8") as handle:
        rows.extend(csv.DictReader(handle))
    if not rows:
        print(f"Error: CSV is empty: {csv_path}")
        return

    total_success = sum(row.get("success", "").strip() == "True" for row in rows)
    total_cost = sum(float(row.get("total_cost", 0) or 0) for row in rows)
    denominator = TOTAL_SAMPLES if len(rows) <= TOTAL_SAMPLES else len(rows)
    overall_accuracy = total_success / denominator if denominator else 0.0

    category_stats = defaultdict(lambda: {"success": 0, "total": 0, "cost": 0.0})
    level_stats = defaultdict(lambda: {"success": 0, "total": 0, "cost": 0.0})
    starts, ends = [], []
    for row in rows:
        success = row.get("success", "").strip() == "True"
        cost = float(row.get("total_cost", 0) or 0)
        category = row.get("category", "unknown").strip() or "unknown"
        level = row.get("level", "unknown").strip() or "unknown"
        for stats, key in ((category_stats, category), (level_stats, level)):
            stats[key]["total"] += 1
            stats[key]["cost"] += cost
            stats[key]["success"] += int(success)
        if start := parse_datetime(row.get("start_time")):
            starts.append(start)
        if end := parse_datetime(row.get("end_time")):
            ends.append(end)

    lines = [
        "=" * 60,
        f"OmniGAIA Orchestra-o1 Evaluation | Main Agent: {args.main_agent}",
        "=" * 60,
        "",
        "[Overall]",
        f"  Rows evaluated:  {len(rows)}",
        f"  Successful:      {total_success}",
        f"  Accuracy:        {overall_accuracy * 100:.1f}% ({total_success}/{denominator})",
        f"  Total cost:      ${total_cost:.2f}",
        "",
        "[By Category]",
    ]
    for category in sorted(category_stats):
        stats = category_stats[category]
        accuracy = stats["success"] / stats["total"] * 100
        lines.append(
            f"  {category:<30} {stats['success']:>3}/{stats['total']:<3} "
            f"({accuracy:>5.1f}%)  ${stats['cost']:.2f}"
        )

    lines.extend(["", "[By Level]"])
    preferred_order = ["Easy", "Medium", "Hard"]
    ordered_levels = preferred_order + [
        level for level in sorted(level_stats) if level not in preferred_order
    ]
    for level in ordered_levels:
        if level not in level_stats:
            continue
        stats = level_stats[level]
        accuracy = stats["success"] / stats["total"] * 100
        lines.append(
            f"  {level:<15} {stats['success']:>3}/{stats['total']:<3} "
            f"({accuracy:>5.1f}%)  ${stats['cost']:.2f}"
        )

    if starts and ends:
        elapsed = max(ends) - min(starts)
        lines.extend(
            [
                "",
                "[Elapsed]",
                f"  Earliest start: {min(starts).isoformat(sep=' ', timespec='seconds')}",
                f"  Latest end:     {max(ends).isoformat(sep=' ', timespec='seconds')}",
                f"  Total elapsed:  {elapsed}",
            ]
        )

    lines.append("=" * 60)
    report = "\n".join(lines)
    output_path = csv_path.parent / f"results_{args.main_agent}.txt"
    output_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nReport saved to: {output_path}")


if __name__ == "__main__":
    main()
