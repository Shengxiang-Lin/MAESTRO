"""Generate reproducible stratified benchmark subsets from source metadata.

Nothing is sampled from hard-coded question IDs. Same source records, seed and
sizes yield the same three TXT manifests. Raw/decrypted benchmark data stays local.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEED = 20261115
PRETEST = "OmniGAIA/splits/pretest_60_ids.txt"
OMNI_FINAL = "OmniGAIA/splits/final_100_ids.txt"
MM_FINAL = "MM-BrowseComp/splits/final_100_ids.txt"


def load_task_ids(path):
    """Read one task ID per line, reject duplicates and malformed lines."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    result, seen = [], set()
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        value = line.split("#", 1)[0].strip()
        if not value:
            continue
        if "," in value or any(ch.isspace() for ch in value):
            raise ValueError(f"{path}:{number}: expected one task ID")
        if value in seen:
            raise ValueError(f"{path}:{number}: duplicate task ID {value}")
        result.append(value)
        seen.add(value)
    if not result:
        raise ValueError(f"Empty task ID file: {path}")
    return result


def read_records(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing metadata: {path}")
    if path.suffix == ".json":
        rows = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(rows, dict):
            rows = rows.get("data", rows.get("tasks"))
    elif path.suffix == ".jsonl":
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        ]
    else:
        raise ValueError("Metadata must be JSON or JSONL")
    if not isinstance(rows, list) or not rows or not all(isinstance(r, dict) for r in rows):
        raise ValueError(f"Invalid metadata: {path}")
    ids = [str(r.get("id", r.get("task_id"))) for r in rows]
    if "None" in ids or len(set(ids)) != len(ids):
        raise ValueError(f"Missing or duplicate IDs in {path}")
    return sorted(rows, key=id_key)


def id_key(row):
    value = str(row.get("id", row.get("task_id")))
    return (0, int(value)) if value.isdecimal() else (1, value)


def even_quotas(n, populations):
    """Balanced stratum quotas; extra seats go to larger available populations."""
    labels = sorted(populations, key=lambda k: (-populations[k], str(k)))
    if not labels:
        raise ValueError("No valid strata")
    base, rem = divmod(n, len(labels))
    result = {key: base + int(i < rem) for i, key in enumerate(labels)}
    if any(result[k] > populations[k] for k in result):
        raise ValueError("Balanced quotas exceed available data")
    return result


def proportional_quotas(n, populations):
    """Largest-remainder apportionment for MM-BrowseComp's original difficulty."""
    total = sum(populations.values())
    weights = {key: n * count / total for key, count in populations.items()}
    quota = {key: math.floor(weight) for key, weight in weights.items()}
    ranks = sorted(weights, key=lambda k: (
        -(weights[k] - quota[k]), -populations[k], str(k)
    ))
    for key in ranks[:n - sum(quota.values())]:
        quota[key] += 1
    return quota


def balanced_matrix(row_quota, col_quota, capacities):
    """Min-cost flow: exact margins, bounded cells, near-even cross strata."""
    rows, cols = list(row_quota), list(col_quota)
    total = sum(row_quota.values())
    if total != sum(col_quota.values()):
        raise ValueError("Unequal marginal totals")
    source, sink = len(rows) + len(cols), len(rows) + len(cols) + 1
    graph = [[] for _ in range(sink + 1)]

    def add(u, v, cap, cost):
        fwd = [v, cap, cost, len(graph[v])]
        back = [u, 0, -cost, len(graph[u])]
        graph[u].append(fwd)
        graph[v].append(back)
        return fwd

    for i, key in enumerate(rows):
        add(source, i, row_quota[key], 0)
    for j, key in enumerate(cols):
        add(len(rows) + j, sink, col_quota[key], 0)
    arcs = {}
    for i, row in enumerate(rows):
        for j, col in enumerate(cols):
            key = (row, col)
            target = row_quota[row] * col_quota[col] / total
            arcs[key] = [
                add(i, len(rows) + j, 1,
                    round(100 * ((k - target)**2 - (k - 1 - target)**2)))
                for k in range(1, capacities.get(key, 0) + 1)
            ]
    for sent in range(total):
        dist = [math.inf] * len(graph)
        pred = [None] * len(graph)
        dist[source] = 0
        for _ in range(len(graph) - 1):
            updated = False
            for u, edges in enumerate(graph):
                if math.isinf(dist[u]):
                    continue
                for edge in edges:
                    if edge[1] > 0 and dist[u] + edge[2] < dist[edge[0]]:
                        dist[edge[0]] = dist[u] + edge[2]
                        pred[edge[0]] = (u, edge)
                        updated = True
            if not updated:
                break
        if pred[sink] is None:
            raise ValueError(f"Infeasible stratified sample after {sent}/{total} units")
        v = sink
        while v != source:
            u, edge = pred[v]
            edge[1] -= 1
            graph[v][edge[3]][1] += 1
            v = u
    return {
        key: sum(edge[1] == 0 for edge in unit_edges)
        for key, unit_edges in arcs.items()
    }


def media_ids(row):
    return {
        str(m.get("id") or m.get("path"))
        for m in row.get("omni_modal_input", [])
        if isinstance(m, dict) and (m.get("id") or m.get("path"))
    }


def select_omni(records, pre_size, final_size, seed, trials):
    valid = [
        r for r in records
        if r.get("Level", r.get("level")) in ("Easy", "Medium", "Hard")
        and r.get("category")
        and r.get("task_type") in ("video w/ audio", "image + audio")
        and isinstance(r.get("omni_modal_input"), list)
        and r["omni_modal_input"]
        and isinstance(r.get("total_steps"), (int, float))
    ]
    categories = Counter(r["category"] for r in valid)
    levels = Counter(r.get("Level", r.get("level")) for r in valid)
    modalities = Counter(r["task_type"] for r in valid)
    cat_pre = even_quotas(pre_size, categories)
    cat_final = even_quotas(final_size, categories)
    lev_pre = even_quotas(pre_size, levels)
    lev_final = even_quotas(final_size, levels)
    by_cell = {
        (cat, level): [
            r for r in valid
            if r["category"] == cat and r.get("Level", r.get("level")) == level
        ]
        for cat in categories for level in levels
    }
    both_c = {cat: cat_pre[cat] + cat_final[cat] for cat in categories}
    both_l = {lv: lev_pre[lv] + lev_final[lv] for lv in levels}
    both = balanced_matrix(
        both_c, both_l, {k: len(v) for k, v in by_cell.items()}
    )
    pre_cells = balanced_matrix(cat_pre, lev_pre, both)
    groups = sorted(both, key=lambda k: (
        len(by_cell[k]) / max(both[k], 1), str(k)
    ))
    rng = random.Random(seed)
    tool_rate = sum(
        "code_executor" in (r.get("required_external_tools") or [])
        for r in valid
    ) / len(valid)
    best, successes = None, 0
    for _ in range(trials):
        used, pre, final = set(), [], []
        for key in groups:
            if both[key] == 0:
                continue
            options = by_cell[key][:]
            rng.shuffle(options)
            selected = []
            for row in options:
                mid = media_ids(row)
                if used.isdisjoint(mid):
                    selected.append(row)
                    used.update(mid)
                    if len(selected) == both[key]:
                        break
            if len(selected) != both[key]:
                break
            rng.shuffle(selected)
            pre.extend(selected[:pre_cells[key]])
            final.extend(selected[pre_cells[key]:])
        else:
            successes += 1
            pv = sum(r["task_type"] == "video w/ audio" for r in pre)
            fv = sum(r["task_type"] == "video w/ audio" for r in final)
            pc = sum(
                "code_executor" in (r.get("required_external_tools") or [])
                for r in pre
            )
            fc = sum(
                "code_executor" in (r.get("required_external_tools") or [])
                for r in final
            )
            score = (
                30 * abs(pv - pre_size / 2)
                + 30 * abs(fv - final_size / 2)
                + 2 * abs(pc - pre_size * tool_rate)
                + 2 * abs(fc - final_size * tool_rate)
            )
            if best is None or score < best[0]:
                best = (score, pre[:], final[:])
    if best is None:
        raise ValueError("Could not create media-disjoint OmniGAIA splits")
    return sorted(best[1], key=id_key), sorted(best[2], key=id_key), {
        "eligible": len(valid),
        "source_categories": dict(categories),
        "source_levels": dict(levels),
        "source_modalities": dict(modalities),
        "category_pretest_quota": cat_pre,
        "category_final_quota": cat_final,
        "level_pretest_quota": lev_pre,
        "level_final_quota": lev_final,
        "successful_trials": successes,
        "best_balance_score": round(best[0], 3),
    }


def select_mm(records, size, seed, trials):
    valid = [
        r for r in records
        if r.get("category") and r.get("subtask")
        and r.get("level", r.get("Level")) is not None
        and isinstance(r.get("images", []), list)
    ]
    categories = Counter(r["category"] for r in valid)
    cat_quota = even_quotas(size, categories)
    sub_quota = {}
    for cat in cat_quota:
        sub = Counter(r["subtask"] for r in valid if r["category"] == cat)
        sub_quota.update({
            (cat, task): amount
            for task, amount in even_quotas(cat_quota[cat], sub).items()
        })
    levels = Counter(str(r.get("level", r.get("Level"))) for r in valid)
    level_quota = proportional_quotas(size, levels)
    by_cell = {
        (group, lv): [
            r for r in valid
            if (r["category"], r["subtask"]) == group
            and str(r.get("level", r.get("Level"))) == lv
        ]
        for group in sub_quota for lv in level_quota
    }
    cell_quota = balanced_matrix(
        sub_quota, level_quota, {k: len(v) for k, v in by_cell.items()}
    )
    groups = sorted(cell_quota, key=lambda k: (
        len(by_cell[k]) / max(1, cell_quota[k]), str(k)
    ))
    rng = random.Random(seed + 1)
    image_target = size * sum(bool(r.get("images")) for r in valid) / len(valid)
    source_target = size * sum(
        len(r.get("source") or []) for r in valid
    ) / len(valid)
    best, successful = None, 0
    for _ in range(trials):
        used, selected = set(), []
        for cell in groups:
            if cell_quota[cell] == 0:
                continue
            options = by_cell[cell][:]
            rng.shuffle(options)
            found = []
            for row in options:
                images = {str(v) for v in (row.get("images") or [])}
                if used.isdisjoint(images):
                    found.append(row)
                    used.update(images)
                    if len(found) == cell_quota[cell]:
                        break
            if len(found) < cell_quota[cell]:
                break
            selected.extend(found)
        else:
            successful += 1
            score = (
                3 * abs(
                    sum(bool(r.get("images")) for r in selected) - image_target
                )
                + 0.3 * abs(
                    sum(len(r.get("source") or []) for r in selected)
                    - source_target
                )
            )
            if best is None or score < best[0]:
                best = (score, selected[:])
    if best is None:
        raise ValueError("Could not sample MM-BrowseComp with distinct images")
    return sorted(best[1], key=id_key), {
        "eligible": len(valid),
        "source_categories": dict(categories),
        "source_levels": dict(levels),
        "category_quota": cat_quota,
        "level_quota": level_quota,
        "subtasks_covered": len(sub_quota),
        "successful_trials": successful,
        "best_balance_score": round(best[0], 3),
    }


def generate_splits(omni, mm, *, seed=DEFAULT_SEED, pretest_size=60,
                    omni_final_size=100, mm_final_size=100, trials=512):
    """Generate all splits using only the source records, requested sizes and seed."""
    if min(pretest_size, omni_final_size, mm_final_size, trials) <= 0:
        raise ValueError("Sizes and trials must be positive")
    pre, final, omni_stat = select_omni(
        omni, pretest_size, omni_final_size, seed, trials
    )
    mm_final, mm_stat = select_mm(mm, mm_final_size, seed, trials)
    manifests = {
        PRETEST: [str(r["id"]) for r in pre],
        OMNI_FINAL: [str(r["id"]) for r in final],
        MM_FINAL: [str(r["id"]) for r in mm_final],
    }
    sizes = [pretest_size, omni_final_size, mm_final_size]
    if any(len(set(ids)) != expected
           for ids, expected in zip(manifests.values(), sizes)):
        raise ValueError("Duplicate or missing sampled IDs")
    if set(manifests[PRETEST]) & set(manifests[OMNI_FINAL]):
        raise ValueError("Overlapping OmniGAIA task IDs")
    pre_media = set().union(*(media_ids(r) for r in pre))
    final_media = set().union(*(media_ids(r) for r in final))
    if pre_media & final_media:
        raise ValueError("Overlapping OmniGAIA media IDs")
    return manifests, {
        "algorithm": "seeded-balanced-strata-v1",
        "seed": seed,
        "trials": trials,
        "sizes": dict(zip(
            ("omni_pretest", "omni_final", "mm_final"), sizes
        )),
        "omnigaia": omni_stat,
        "mm_browsecomp": mm_stat,
        "selected": {
            "omni_pretest": {
                "levels": dict(Counter(r.get("Level", r.get("level")) for r in pre)),
                "categories": dict(Counter(r["category"] for r in pre)),
                "modalities": dict(Counter(r["task_type"] for r in pre)),
            },
            "omni_final": {
                "levels": dict(Counter(r.get("Level", r.get("level")) for r in final)),
                "categories": dict(Counter(r["category"] for r in final)),
                "modalities": dict(Counter(r["task_type"] for r in final)),
            },
            "mm_final": {
                "levels": dict(Counter(
                    str(r.get("level", r.get("Level"))) for r in mm_final
                )),
                "categories": dict(Counter(r["category"] for r in mm_final)),
                "subtask_coverage": len({r["subtask"] for r in mm_final}),
            },
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--pretest-size", type=int, default=60)
    parser.add_argument("--omni-final-size", type=int, default=100)
    parser.add_argument("--mm-final-size", type=int, default=100)
    parser.add_argument("--trials", type=int, default=512)
    parser.add_argument(
        "--omnigaia-meta", type=Path,
        default=ROOT / "data/OmniGAIA/test_metadata.jsonl",
    )
    parser.add_argument(
        "--mm-meta", type=Path,
        default=ROOT / "data/MM-BrowseComp/data/MMBrowseComp_400_decrypted.jsonl",
    )
    parser.add_argument("--output-root", type=Path, default=ROOT / "data")
    parser.add_argument(
        "--check", action="store_true",
        help="Compare regenerated IDs with saved files",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Report only; do not change saved IDs",
    )
    args = parser.parse_args()
    omni, mm = read_records(args.omnigaia_meta), read_records(args.mm_meta)
    manifests, report = generate_splits(
        omni, mm, seed=args.seed,
        pretest_size=args.pretest_size,
        omni_final_size=args.omni_final_size,
        mm_final_size=args.mm_final_size, trials=args.trials,
    )
    report["source_sha256"] = {
        "omnigaia": hashlib.sha256(args.omnigaia_meta.read_bytes()).hexdigest(),
        "mm_browsecomp": hashlib.sha256(args.mm_meta.read_bytes()).hexdigest(),
    }
    for name, ids in manifests.items():
        path = args.output_root / name
        expected = "".join(f"{task_id}\n" for task_id in ids)
        if args.check:
            if not path.is_file() or path.read_text(encoding="utf-8") != expected:
                raise ValueError(f"Out-of-date manifest: {path}")
        elif not args.dry_run:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(expected, encoding="utf-8")
        print(
            f'{"Checked" if args.check else "Planned" if args.dry_run else "Wrote"}'
            f" {path} ({len(ids)} IDs)"
        )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
