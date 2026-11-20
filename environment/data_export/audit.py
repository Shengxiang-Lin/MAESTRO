"""Deterministic quality audit for unified MAESTRO Main-decision adapters.

Checks Scene-level isolation and the *actual* SFT prompt/completion contract.
Reports hash-only duplicate/conflict examples; never logs original prompts,
gold labels or private fault/teacher state.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
from statistics import median

SPLITS = ("train", "val", "test")
SCHEMA = "maestro.data_quality.v1"
REVIEWED_MODES = {
    "llm_reviewed", "llm_reviewed_completion",
    "llm_reviewed_contract_frozen",
}


def _require(test, reason):
    if not test:
        raise ValueError("data_quality:" + reason)


def _digest(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def _action_key(action: dict) -> str:
    return _digest(json.dumps(action, sort_keys=True, ensure_ascii=False))


def _check_action(action, name):
    _require(isinstance(action, dict) and
             set(action) == {"action", "reasoning", "params"}, name + ":action_schema")
    _require(action["action"] in {"delegate_task", "complete"}, name + ":action_type")
    _require(isinstance(action["reasoning"], str) and
             bool(action["reasoning"].strip()), name + ":reasoning")
    params = action["params"]
    _require(isinstance(params, dict), name + ":params")
    if action["action"] == "delegate_task":
        _require(isinstance(params.get("tasks"), list) and
                 bool(params["tasks"]), name + ":delegated_tasks")
    else:
        _require(isinstance(params.get("answer"), str), name + ":answer")


def audit_dataset(cases: list[dict], decisions: list[dict],
                  sft_rows: list[tuple[str, dict]], scene_to_split: dict,
                  *, strict_conflicts: bool = False) -> dict:
    """Validate pairing and return reproducible, non-sensitive diagnostics.

    sft_rows must follow decisions in order and contain (split, adapter row).
    Diagnostics count duplicates; they do not reweight, drop or rewrite labels.
    """
    _require(isinstance(scene_to_split, dict) and bool(scene_to_split),
             "missing_scene_split_map")
    _require(set(scene_to_split.values()) <= set(SPLITS),
             "invalid_scene_split")
    _require(len(decisions) == len(sft_rows), "sft_decision_row_count_mismatch")

    case_index = {}
    clean_ids = set()
    fault_cases = Counter()
    for case in cases:
        cid, scene = case.get("case_id"), case.get("scene_id")
        _require(isinstance(cid, str) and cid and cid not in case_index,
                 "duplicate_or_missing_case_id")
        _require(scene in scene_to_split and
                 case.get("split") == scene_to_split[scene],
                 "case_scene_split_mismatch:" + cid)
        kind = case.get("kind")
        _require(kind in {"clean", "fault_injected"}, "invalid_case_kind:" + cid)
        if kind == "clean":
            _require(cid == "clean:" + scene and scene not in clean_ids,
                     "invalid_clean_case:" + cid)
            clean_ids.add(scene)
        else:
            atoms = (case.get("fault") or {}).get("atoms") or []
            fault_cases[str(atoms[0].get("kind", "unknown"))
                        if atoms and isinstance(atoms[0], dict)
                        else "unknown"] += 1
        case_index[cid] = case
    _require(clean_ids == set(scene_to_split), "missing_clean_scene_case")

    seen_ids = set()
    seen_rounds = set()
    seen_prompts = defaultdict(lambda: {
        "splits": set(), "labels": set(), "ids": [],
    })
    exact = Counter()
    source_counts, action_counts, mode_counts = Counter(), Counter(), Counter()
    source_split, split_counts = Counter(), Counter()
    scene_counts, fault_decisions = Counter(), Counter()
    for i, (decision, (adapter_split, adapter)) in enumerate(
            zip(decisions, sft_rows)):
        did = decision.get("decision_id")
        scene = decision.get("scene_id")
        cid = decision.get("case_id")
        split = decision.get("split")
        _require(isinstance(did, str) and did not in seen_ids,
                 "duplicate_decision_id:" + str(did))
        seen_ids.add(did)
        _require(scene in scene_to_split and
                 split == scene_to_split[scene] == adapter_split,
                 "decision_scene_split_mismatch:" + str(did))
        case = case_index.get(cid)
        _require(case is not None and case["scene_id"] == scene,
                 "missing_or_wrong_case:" + did)
        source = decision.get("source")
        _require(source in {"clean", "untrusted"}, "invalid_source:" + did)
        _require((source == "clean" and case["kind"] == "clean") or
                 (source == "untrusted" and
                  case["kind"] == "fault_injected" and
                  (case.get("recovery") or {}).get("globally_recovered") is True),
                 "unpublished_or_mismatched_case:" + did)
        turn = decision.get("decision_round")
        _require(type(turn) is int and turn >= 1,
                 "invalid_decision_round:" + did)
        rkey = (cid, turn)
        _require(rkey not in seen_rounds, "duplicate_case_round:" + did)
        seen_rounds.add(rkey)
        prompt = decision.get("prompt")
        _require(isinstance(prompt, list) and len(prompt) == 1 and
                 isinstance(prompt[0], dict) and
                 prompt[0].get("role") == "user" and
                 isinstance(prompt[0].get("content"), str) and
                 bool(prompt[0]["content"]), "invalid_o1_prompt:" + did)
        ref = decision.get("reference_action")
        _check_action(ref, did)
        _require(isinstance(adapter, dict) and
                 set(adapter) == {"prompt", "completion"} and
                 adapter["prompt"] == prompt, "adapter_prompt_drift:" + did)
        completion = adapter["completion"]
        _require(isinstance(completion, list) and len(completion) == 1 and
                 isinstance(completion[0], dict) and
                 completion[0].get("role") == "assistant" and
                 isinstance(completion[0].get("content"), str),
                 "invalid_sft_completion:" + did)
        try:
            actual = json.loads(completion[0]["content"])
        except (ValueError, TypeError) as exc:
            raise ValueError("data_quality:invalid_sft_json:" + did) from exc
        _require(actual == ref, "adapter_completion_drift:" + did)

        ph = _digest(prompt[0]["content"])
        ah = _action_key(ref)
        group = seen_prompts[ph]
        group["splits"].add(split)
        group["labels"].add(ah)
        group["ids"].append(did)
        exact[(ph, ah)] += 1
        source_counts[source] += 1
        source_split[(split, source)] += 1
        action_counts[ref["action"]] += 1
        mode_counts[str(decision.get("generation_mode", "unknown"))] += 1
        split_counts[split] += 1
        scene_counts[scene] += 1
        if source == "untrusted":
            atoms = (case.get("fault") or {}).get("atoms") or []
            fault_decisions[str(atoms[0].get("kind", "unknown"))
                            if atoms and isinstance(atoms[0], dict)
                            else "unknown"] += 1

    cross_split = [h for h, row in seen_prompts.items()
                   if len(row["splits"]) > 1]
    _require(not cross_split, "cross_split_identical_prompts:" +
             ",".join(sorted(cross_split)[:3]))
    conflicts = [(h, row) for h, row in seen_prompts.items()
                 if len(row["labels"]) > 1]
    if strict_conflicts:
        _require(not conflicts, "conflicting_labels_for_identical_prompt:" +
                 ",".join(sorted(h for h, _ in conflicts)[:3]))
    per_scene = sorted(scene_counts.values())
    warnings = []
    exact_extras = sum(count - 1 for count in exact.values())
    if exact_extras:
        warnings.append("identical_prompt_and_completion_repeated")
    if conflicts:
        warnings.append("identical_prompt_has_conflicting_completions")
    if source_counts["untrusted"] > 9 * max(1, source_counts["clean"]):
        warnings.append("untrusted_decisions_outnumber_clean_by_over_9_to_1")
    return {
        "schema": SCHEMA,
        "scene_counts": dict(sorted(Counter(scene_to_split.values()).items())),
        "case_count": len(cases),
        "decision_count": len(decisions),
        "decision_split_counts": {s: split_counts[s] for s in SPLITS},
        "source_counts": dict(sorted(source_counts.items())),
        "source_by_split": {
            s: {src: source_split[(s, src)]
                for src in ("clean", "untrusted")} for s in SPLITS
        },
        "action_counts": dict(sorted(action_counts.items())),
        "generation_modes": dict(sorted(mode_counts.items())),
        "fault_cases_by_kind": dict(sorted(fault_cases.items())),
        "fault_decisions_by_kind": dict(sorted(fault_decisions.items())),
        "unique_prompt_count": len(seen_prompts),
        "repeated_prompt_groups": sum(
            len(v["ids"]) > 1 for v in seen_prompts.values()
        ),
        "exact_duplicate_decision_rows": exact_extras,
        "conflicting_prompt_groups": len(conflicts),
        "conflict_examples": [{
            "prompt_sha256": h,
            "decision_ids": row["ids"][:8],
            "distinct_completion_count": len(row["labels"]),
        } for h, row in sorted(conflicts)[:10]],
        "cross_split_prompt_groups": len(cross_split),
        "per_scene_decisions": {
            "min": per_scene[0] if per_scene else 0,
            "median": median(per_scene) if per_scene else 0,
            "max": per_scene[-1] if per_scene else 0,
            "top_scenes": [
                {"scene_id": scene, "decisions": count}
                for scene, count in sorted(
                    scene_counts.items(), key=lambda pair: (-pair[1], pair[0])
                )[:10]
            ],
        },
        "warnings": warnings,
    }


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError("data_quality:missing_file:" + str(path))
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            item = json.loads(line)
            _require(isinstance(item, dict),
                     "invalid_jsonl_row:" + str(path) + ":" + str(line_number))
            rows.append(item)
    return rows


def audit_output(output_dir: Path, *, check_parquet: bool = False,
                 strict_conflicts: bool = False) -> dict:
    """Audit persisted JSONL and optionally verify Parquet adapter round trips."""
    output_dir = Path(output_dir)
    split_file = output_dir / "splits.json"
    split_data = json.loads(split_file.read_text(encoding="utf-8"))
    scene_to_split = split_data["scene_to_split"]
    cases, decisions, sft_rows = [], [], []
    for split in SPLITS:
        cases.extend(_read_jsonl(output_dir / "cases" / (split + ".jsonl")))
        split_decisions = _read_jsonl(output_dir / "decisions" / (split + ".jsonl"))
        adapters = _read_jsonl(
            output_dir / "adapters" / "da-sft" / (split + ".jsonl")
        )
        decisions.extend(split_decisions)
        sft_rows.extend((split, row) for row in adapters)
        _require(len(split_decisions) == len(adapters),
                 "decision_adapter_split_length_mismatch:" + split)
        if check_parquet:
            parquet_name = "eval" if split == "val" else split
            path = output_dir / "adapters" / "da-sft" / (parquet_name + ".parquet")
            if adapters:
                import pyarrow.parquet as pq
                _require(path.is_file(), "missing_sft_parquet:" + str(path))
                _require(pq.read_table(path).to_pylist() == adapters,
                         "sft_parquet_round_trip_mismatch:" + split)
            else:
                _require(not path.exists(),
                         "unexpected_sft_parquet_for_empty_split:" + split)
    return audit_dataset(
        cases, decisions, sft_rows, scene_to_split,
        strict_conflicts=strict_conflicts,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--check-parquet", action="store_true")
    parser.add_argument("--strict-conflicts", action="store_true")
    args = parser.parse_args(argv)
    result = audit_output(
        args.output_dir, check_parquet=args.check_parquet,
        strict_conflicts=args.strict_conflicts,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
