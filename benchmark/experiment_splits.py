"""Rebuild and verify MAESTRO's frozen experiment task-ID selections.

Selected from official OmniGAIA-360 / MM-BrowseComp-400 metadata using
stratified sampling. The evaluated IDs are immutable; this module deliberately
does not re-roll a random seed or publish decrypted MM-BrowseComp content.

Run: python benchmark/experiment_splits.py [--check] [--verify-data]
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OMNI_PRETEST_60 = (5,12,20,28,35,39,45,46,60,61,64,71,72,74,75,77,80,91,95,100,115,117,118,121,127,153,164,165,172,179,181,182,186,194,195,206,208,209,212,224,230,234,243,245,249,265,273,280,283,288,292,312,313,318,328,342,345,347,358,359)
OMNI_FINAL_100 = (1,2,8,13,16,19,22,26,30,34,36,37,38,49,50,51,53,65,67,69,78,79,82,86,87,97,102,107,108,116,122,125,126,128,133,135,136,137,143,145,149,150,151,155,167,169,171,175,180,183,190,197,201,205,213,215,216,217,226,227,228,244,251,253,254,259,261,269,270,277,278,281,285,287,289,299,301,302,303,305,307,311,314,319,321,323,324,325,326,332,334,335,339,340,348,350,352,353,355,360)
MM_FINAL_100 = (1,3,5,9,17,19,28,30,33,48,52,54,67,68,71,73,81,87,88,97,104,109,111,118,122,123,125,141,142,144,145,149,156,157,160,161,167,176,180,187,188,207,208,210,211,218,228,232,233,237,308,312,314,315,1001,1004,1009,1015,1018,1022,1024,1031,1036,1048,1049,1050,1054,1060,1062,1070,1071,1073,1077,1081,1086,1088,1090,1096,1099,1100,1106,1109,1117,1119,1120,1121,1126,1127,1130,1131,1133,1135,1147,1153,1157,1159,1163,1166,1175,1301)
MANIFESTS = {
    "OmniGAIA/splits/pretest_60_ids.txt": OMNI_PRETEST_60,
    "OmniGAIA/splits/final_100_ids.txt": OMNI_FINAL_100,
    "MM-BrowseComp/splits/final_100_ids.txt": MM_FINAL_100,
}


def load_task_ids(path: Path | str) -> list[str]:
    """Read a one-ID-per-line manifest; fail on duplicate/malformed IDs."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Task IDs file not found: {path}")
    ids: list[str] = []
    seen: set[str] = set()
    for line_no, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        task_id = raw.split("#", 1)[0].strip()
        if not task_id:
            continue
        if any(ch.isspace() for ch in task_id) or "," in task_id:
            raise ValueError(f"{path}:{line_no}: one task ID per line is required")
        if task_id in seen:
            raise ValueError(f"{path}:{line_no}: duplicate task ID {task_id}")
        ids.append(task_id)
        seen.add(task_id)
    if not ids:
        raise ValueError(f"No task IDs in {path}")
    return ids


def _read_records(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig") as stream:
        if path.suffix.lower() == ".json":
            rows = json.load(stream)
            if isinstance(rows, dict):
                rows = rows.get("data", rows.get("tasks"))
        elif path.suffix.lower() == ".jsonl":
            rows = [json.loads(line) for line in stream if line.strip()]
        else:
            raise ValueError(f"Expected JSON/JSONL metadata: {path}")
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"Expected a list of records in {path}")
    return rows


def _index(rows: list[dict], name: str) -> dict[str, dict]:
    lookup: dict[str, dict] = {}
    for row in rows:
        task_id = row.get("id", row.get("task_id"))
        if task_id is None or str(task_id) in lookup:
            raise ValueError(f"{name}: missing/duplicate task id {task_id}")
        lookup[str(task_id)] = row
    return lookup


def verify_metadata(omnigaia_meta: Path, mm_meta: Path) -> None:
    omni = _index(_read_records(omnigaia_meta), "OmniGAIA")
    mm = _index(_read_records(mm_meta), "MM-BrowseComp")
    if len(omni) != 360 or len(mm) != 400:
        raise ValueError("Expected OmniGAIA-360 and MM-BrowseComp-400 official datasets")
    for label, selected, source in (
        ("OmniGAIA pretest", OMNI_PRETEST_60, omni),
        ("OmniGAIA final", OMNI_FINAL_100, omni),
        ("MM-BrowseComp final", MM_FINAL_100, mm),
    ):
        absent = set(map(str, selected)) - set(source)
        if absent:
            raise ValueError(f"{label}: missing IDs {sorted(absent)}")

    media_used: set[str] = set()
    for label, selected, desired_levels, desired_types in (
        ("pretest", OMNI_PRETEST_60, {"Easy": 20, "Medium": 20, "Hard": 20},
         {"video w/ audio": 30, "image + audio": 30}),
        ("final", OMNI_FINAL_100, {"Easy": 34, "Medium": 33, "Hard": 33},
         {"video w/ audio": 50, "image + audio": 50}),
    ):
        rows = [omni[str(i)] for i in selected]
        levels = Counter(row.get("Level", row.get("level")) for row in rows)
        task_types = Counter(row.get("task_type") for row in rows)
        categories = Counter(row.get("category") for row in rows)
        if levels != desired_levels or task_types != desired_types:
            raise ValueError(f"OmniGAIA {label}: wrong level/task-type distribution")
        if len(categories) != 9 or max(categories.values()) - min(categories.values()) > 1:
            raise ValueError(f"OmniGAIA {label}: expected 9 balanced categories")
        for row in rows:
            for media in row.get("omni_modal_input") or []:
                media_id = media.get("id") or media.get("path")
                if media_id:
                    if media_id in media_used:
                        raise ValueError(f"OmniGAIA pretest/final shared media: {media_id}")
                    media_used.add(media_id)

    mm_rows = [mm[str(i)] for i in MM_FINAL_100]
    mm_categories = Counter(row["category"] for row in mm_rows)
    mm_levels = Counter(str(row["level"]) for row in mm_rows)
    if len(mm_categories) != 5 or set(mm_categories.values()) != {20}:
        raise ValueError("MM-BrowseComp: expected 20 questions per category")
    if mm_levels != {"1": 85, "2": 15}:
        raise ValueError("MM-BrowseComp: expected 85/15 levels")
    if len({row["subtask"] for row in mm_rows}) != 22:
        raise ValueError("MM-BrowseComp: expected coverage of all 22 subtasks")
    print("Verified against 360 OmniGAIA and 400 MM-BrowseComp source records")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT / "data")
    parser.add_argument("--check", action="store_true", help="Verify committed manifests, do not change them")
    parser.add_argument("--verify-data", action="store_true", help="Audit IDs against both metadata files")
    parser.add_argument("--omnigaia-meta", type=Path, default=ROOT / "data/OmniGAIA/test_metadata.json")
    parser.add_argument("--mm-meta", type=Path, default=ROOT / "data/MM-BrowseComp/data/MMBrowseComp_400.jsonl")
    args = parser.parse_args()

    if len(OMNI_PRETEST_60) != 60 or len(OMNI_FINAL_100) != 100 or len(MM_FINAL_100) != 100:
        raise ValueError("Wrong frozen manifest size")
    if len(set(OMNI_PRETEST_60)) != 60 or len(set(OMNI_FINAL_100)) != 100 or len(set(MM_FINAL_100)) != 100:
        raise ValueError("Duplicated frozen task IDs")
    if set(OMNI_PRETEST_60) & set(OMNI_FINAL_100):
        raise ValueError("OmniGAIA splits overlap")

    for relative_path, selected in MANIFESTS.items():
        path = args.output_root / relative_path
        expected = "".join(f"{i}\n" for i in selected)
        if args.check:
            if path.read_text(encoding="utf-8") != expected:
                raise ValueError(f"Manifest differs from frozen IDs: {path}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(expected, encoding="utf-8")
        print(f"{'Checked' if args.check else 'Wrote'} {path} ({len(selected)} IDs)")
    if args.verify_data:
        verify_metadata(args.omnigaia_meta, args.mm_meta)


if __name__ == "__main__":
    main()
