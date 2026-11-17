"""Reconstruct full baseline O1 prompts from compact Stage-3 training turns.

Default main_turns.jsonl keeps only cumulative Sub-Agent feedback in
input.subtask_history; the task-specific Question and source catalog are
stored once in Stage-1 scenes, and model/attempt settings once in the
Stage-3 manifest. This optional exporter restores an ordinary prompt/action
dataset without introducing private Oracle or Macro graph content.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from environment.generate_question.stages import read_jsonl
from .generate_turns import _digest
from .o1_compat import o1_prompt_from_turn


def export_full_prompts(*, scenes_path: Path, turns_path: Path,
                        manifest_path: Path, output_path: Path) -> dict:
    scenes_path, turns_path = Path(scenes_path), Path(turns_path)
    manifest_path, output_path = Path(manifest_path), Path(output_path)
    if output_path.resolve() in {
        scenes_path.resolve(), turns_path.resolve(), manifest_path.resolve(),
    }:
        raise ValueError("stage3_render_output_must_not_replace_inputs")
    meta = json.loads(manifest_path.read_text(encoding="utf-8"))
    if _digest(scenes_path) != meta["scenes_sha256"]:
        raise ValueError("stage3_prompt_render_scene_manifest_mismatch")
    # Refuse to claim byte-exact O1 reconstruction after relevant renderer
    # changes. Resume/checkpoint code uses the same code signature.
    saved_code = meta["code_sha256"].get("o1_compat.py")
    if saved_code != _digest(Path(__file__).with_name("o1_compat.py")):
        raise ValueError("stage3_prompt_render_code_signature_mismatch")
    root = Path(__file__).resolve().parent
    for source in (
        "../../orchestra/base_main_agent.py",
        "../../orchestra/prompts/omnigaia.py",
        "../generate_question/execution/evidence_tools.py",
        "../../benchmark/omnigaia/tools/google_search.py",
        "../../benchmark/omnigaia/tools/extract_url_jina.py",
        "../../benchmark/omnigaia/tools/multimodal_toolkit.py",
        "../../benchmark/omnigaia/tools/audio_analysis.py",
        "../../benchmark/omnigaia/tools/video_analysis.py",
        "../../benchmark/omnigaia/tools/execute_code.py",
    ):
        expected = meta["code_sha256"].get(source)
        if expected is None or expected != _digest(root / source):
            raise ValueError("stage3_prompt_render_code_signature_mismatch")

    rows = read_jsonl(scenes_path, limit=meta.get("max_scenes"))
    scenes = {row["task_id"]: row for row in rows}
    if len(scenes) != len(rows):
        raise ValueError("stage3_prompt_render_duplicate_scenes")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp = output_path.with_suffix(output_path.suffix + ".tmp")
    exported = 0
    seen = set()
    try:
        with turns_path.open(encoding="utf-8") as reader, temp.open(
            "w", encoding="utf-8",
        ) as writer:
            for line in reader:
                if not line.strip():
                    continue
                turn = json.loads(line)
                sid = turn["task_id"]
                if sid not in scenes:
                    raise ValueError("stage3_prompt_render_unknown_scene:" + sid)
                sample_id = turn["sample_id"]
                if sample_id in seen:
                    raise ValueError("stage3_prompt_render_duplicate_sample")
                seen.add(sample_id)
                prompt = o1_prompt_from_turn(
                    scenes[sid], turn,
                    max_attempts=meta["max_attempts"],
                    sub_model=meta["sub_model"],
                    model_alias=meta["model_alias"],
                )
                output = {**turn, "input": {"prompt": prompt}}
                writer.write(json.dumps(output, ensure_ascii=False) + "\n")
                exported += 1
        os.replace(temp, output_path)
    finally:
        if temp.exists():
            temp.unlink()
    return {"exported_turns": exported, "output": str(output_path)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", required=True, type=Path)
    parser.add_argument("--turns", required=True, type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    result = export_full_prompts(
        scenes_path=args.scenes, turns_path=args.turns,
        manifest_path=args.manifest or args.turns.parent / "manifest.json",
        output_path=args.output,
    )
    print(f"Rebuilt {result['exported_turns']} exact O1 prompts: "
          f"{result['output']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
