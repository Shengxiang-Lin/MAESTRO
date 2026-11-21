"""Base MAESTRO dataset integrity audit (no baseline split or Parquet policy).

Audits only the method-independent Case, Main decision and private label
contract. Repeated prompt/label observations are descriptive; baselines may
choose different deduplication or weighting policies.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
from statistics import median

from tqdm.auto import tqdm

from .source_io import digest, read_jsonl

SCHEMA = "maestro.base_quality.v1"


def _check(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError("base_data_quality:" + reason)


def _hash(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def _action_hash(value: dict) -> str:
    return _hash(json.dumps(value, ensure_ascii=False, sort_keys=True))


def _validate_action(action: dict, identity: str) -> None:
    _check(isinstance(action, dict) and
           set(action) == {"action", "reasoning", "params"},
           "action_fields:" + identity)
    _check(action["action"] in {"delegate_task", "complete"},
           "action_type:" + identity)
    _check(isinstance(action["reasoning"], str) and
           bool(action["reasoning"].strip()), "reasoning:" + identity)
    params = action["params"]
    _check(isinstance(params, dict), "action_params:" + identity)
    if action["action"] == "delegate_task":
        tasks = params.get("tasks")
        _check(isinstance(tasks, list) and bool(tasks),
               "delegate_tasks:" + identity)
    else:
        _check(isinstance(params.get("answer"), str),
               "complete_answer:" + identity)


def audit_dataset(cases: list[dict], decisions: list[dict],
                  labels: list[dict], *, progress: bool = False) -> dict:
    """Raise for integrity/privacy boundary violations; report distributions."""
    _check(len(decisions) == len(labels), "decision_teacher_count_mismatch")
    case_by_id = {}
    scenes = set()
    published = set()
    cases_by_kind, fault_by_kind = Counter(), Counter()
    for case in tqdm(cases, desc="Audit base Cases", unit="case",
                     disable=not progress, leave=False):
        identity, scene = case.get("case_id"), case.get("scene_id")
        _check(isinstance(identity, str) and identity and
               identity not in case_by_id, "duplicate_case_identity")
        _check(isinstance(scene, str) and scene, "case_missing_scene")
        kind = case.get("kind")
        _check(kind in {"clean", "fault_injected"}, "case_kind:" + identity)
        _check("split" not in case, "method_specific_split_in_base_case")
        if kind == "clean":
            _check(identity == "clean:" + scene and scene not in scenes,
                   "clean_scene_identity")
            scenes.add(scene)
        else:
            recovery = case.get("recovery") or {}
            if (recovery.get("decision_accepted") is True
                    or (recovery.get("decision_accepted") is None
                        and recovery.get("globally_recovered") is True)):
                published.add(identity)
            atoms = (case.get("fault") or {}).get("atoms") or []
            if atoms and isinstance(atoms[0], dict):
                fault_by_kind[str(atoms[0].get("kind", "unknown"))] += 1
            else:
                fault_by_kind["unknown"] += 1
        cases_by_kind[kind] += 1
        case_by_id[identity] = case
    _check(bool(scenes), "no_clean_scenes")
    _check(all(c["scene_id"] in scenes for c in cases),
           "fault_scene_missing_clean_pair")

    teacher_by_id = {}
    for item in tqdm(labels, desc="Audit base labels", unit="label",
                     disable=not progress, leave=False):
        did = item.get("decision_id")
        _check(isinstance(did, str) and did and
               did not in teacher_by_id, "duplicate_private_teacher_id")
        _check("split" not in item, "method_specific_split_in_teacher")
        _check(isinstance(item.get("ground_truth"), str),
               "missing_private_ground_truth")
        teacher_by_id[did] = item

    seen_ids, seen_rounds = set(), set()
    prompts = defaultdict(lambda: {"actions": set(), "ids": []})
    identical = Counter()
    sources, source_mode, modes, actions = Counter(), Counter(), Counter(), Counter()
    fault_decisions, scene_decisions = Counter(), Counter()
    for row in tqdm(decisions, desc="Audit base decisions", unit="decision",
                    disable=not progress, leave=False):
        did, cid, scene = (
            row.get("decision_id"), row.get("case_id"), row.get("scene_id")
        )
        _check(isinstance(did, str) and did and did not in seen_ids,
               "duplicate_decision_id")
        seen_ids.add(did)
        _check(cid in case_by_id and
               case_by_id[cid]["scene_id"] == scene,
               "decision_case_mismatch:" + did)
        _check("split" not in row, "method_specific_split_in_decision")
        kind = case_by_id[cid]["kind"]
        source = row.get("source")
        _check((source == "clean" and kind == "clean") or
               (source == "untrusted" and kind == "fault_injected" and
                cid in published),
               "decision_is_not_published_positive:" + did)
        number = row.get("decision_round")
        _check(type(number) is int and number >= 1,
               "invalid_decision_round:" + did)
        _check((cid, number) not in seen_rounds,
               "duplicate_case_round:" + did)
        seen_rounds.add((cid, number))
        prompt = row.get("prompt")
        _check(isinstance(prompt, list) and len(prompt) == 1 and
               isinstance(prompt[0], dict) and
               prompt[0].get("role") == "user" and
               isinstance(prompt[0].get("content"), str) and
               bool(prompt[0]["content"]),
               "invalid_o1_prompt:" + did)
        state = row.get("visible_state")
        _check(isinstance(state, dict) and
               set(state) == {"subtask_history"} and
               isinstance(state["subtask_history"], str),
               "visible_history:" + did)
        _check(isinstance(row.get("question"), str),
               "missing_scene_question:" + did)
        _validate_action(row.get("reference_action"), did)
        teacher = teacher_by_id.get(did)
        _check(teacher is not None and
               teacher.get("case_id") == cid and
               teacher.get("scene_id") == scene and
               teacher.get("reference_action") == row["reference_action"],
               "teacher_decision_alignment:" + did)
        h = _hash(prompt[0]["content"])
        ah = _action_hash(row["reference_action"])
        prompts[h]["actions"].add(ah)
        prompts[h]["ids"].append(did)
        identical[(h, ah)] += 1
        sources[source] += 1
        mode = str(row.get("generation_mode", "unknown"))
        modes[mode] += 1
        source_mode[(source, mode)] += 1
        actions[row["reference_action"]["action"]] += 1
        scene_decisions[scene] += 1
        if source == "untrusted":
            atoms = (case_by_id[cid].get("fault") or {}).get("atoms") or []
            if atoms and isinstance(atoms[0], dict):
                fault_decisions[str(atoms[0].get("kind", "unknown"))] += 1
            else:
                fault_decisions["unknown"] += 1
    _check(seen_ids == set(teacher_by_id), "unmatched_private_teacher")
    per_scene = sorted(scene_decisions.values())
    conflicts = [(key, value) for key, value in prompts.items()
                 if len(value["actions"]) > 1]
    warnings = []
    extra = sum(n - 1 for n in identical.values())
    if extra:
        warnings.append("identical_prompt_and_reference_action_repeated")
    if conflicts:
        warnings.append("identical_prompt_has_conflicting_references")
    if sources["untrusted"] > 9 * max(1, sources["clean"]):
        warnings.append("untrusted_decisions_outnumber_clean_by_over_9_to_1")
    return {
        "schema": SCHEMA,
        "scene_count": len(scenes),
        "case_count": len(cases),
        "cases_by_kind": dict(sorted(cases_by_kind.items())),
        "decision_count": len(decisions),
        "source_counts": dict(sorted(sources.items())),
        "source_generation_modes": {
            source: {mode: n for (s, mode), n in sorted(source_mode.items())
                    if s == source}
            for source in ("clean", "untrusted")
        },
        "generation_modes": dict(sorted(modes.items())),
        "action_counts": dict(sorted(actions.items())),
        "fault_cases_by_kind": dict(sorted(fault_by_kind.items())),
        "fault_decisions_by_kind": dict(sorted(fault_decisions.items())),
        "unique_prompt_count": len(prompts),
        "repeated_prompt_groups": sum(
            len(v["ids"]) > 1 for v in prompts.values()
        ),
        "exact_duplicate_decision_rows": extra,
        "conflicting_prompt_groups": len(conflicts),
        "conflict_examples": [{
            "prompt_sha256": key,
            "decision_ids": value["ids"][:8],
            "distinct_completion_count": len(value["actions"]),
        } for key, value in sorted(conflicts)[:10]],
        "per_scene_decisions": {
            "min": per_scene[0] if per_scene else 0,
            "median": median(per_scene) if per_scene else 0,
            "max": per_scene[-1] if per_scene else 0,
            "top_scenes": [{
                "scene_id": scene, "decisions": count
            } for scene, count in sorted(
                scene_decisions.items(), key=lambda item: (-item[1], item[0])
            )[:10]],
        },
        "warnings": warnings,
    }


def audit_output(output_dir: Path, *, verify_manifest: bool = False,
                 progress: bool = False) -> dict:
    root = Path(output_dir)
    files = {
        "cases.jsonl": root / "cases.jsonl",
        "decisions.jsonl": root / "decisions.jsonl",
        "private/decision_labels.jsonl": root / "private/decision_labels.jsonl",
    }
    if verify_manifest:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        for name, path in files.items():
            _check(digest(path) == manifest["base_file_sha256"].get(name),
                   "base_source_checksum_changed:" + name)
    if verify_manifest:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if "private_context" in manifest:
            from .private_context import verify_private_context
            verify_private_context(root, require_complete=False)
    return audit_dataset(
        read_jsonl(files["cases.jsonl"], progress=progress, desc="Read base Cases"),
        read_jsonl(files["decisions.jsonl"], progress=progress, desc="Read base decisions"),
        read_jsonl(files["private/decision_labels.jsonl"],
                   progress=progress, desc="Read base labels"),
        progress=progress,
    )


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--verify-manifest", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(audit_output(
        args.output_dir, verify_manifest=args.verify_manifest,
        progress=not args.no_progress,
    ), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
