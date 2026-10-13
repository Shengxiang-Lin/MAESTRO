"""Prepare dataset-specific inputs for the shared OmniGAIA JSONL reader."""
from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
import tempfile
from urllib.parse import unquote, urlparse


MM_QUESTION_PREFIX = (
    "Please answer the following question and also provide your problem-solving "
    "roadmap. Question: "
)


def normalize_dataset_type(value: str) -> str:
    name = str(value).strip().lower().replace("-", "").replace("_", "")
    if name == "omnigaia":
        return "omnigaia"
    if name == "mmbrowsecomp":
        return "mm_browsecomp"
    raise ValueError(f"Unsupported dataset_type {value!r}; use OmniGAIA or MM-BrowseComp")


def _read_records(source: Path) -> list[dict]:
    with source.open(encoding="utf-8-sig") as handle:
        if source.suffix.lower() == ".json":
            rows = json.load(handle)
            if isinstance(rows, dict):
                # Common JSON exports wrap the record list in one of these keys.
                rows = next((rows[key] for key in ("data", "tasks") if key in rows), rows)
            if not isinstance(rows, list):
                raise ValueError(f"{source}: JSON must contain a record list (or data/tasks list)")
        elif source.suffix.lower() == ".jsonl":
            rows = []
            for number, line in enumerate(handle, 1):
                if line.strip():
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"{source}: invalid JSONL at line {number}") from exc
        else:
            raise ValueError(f"{source}: expected a .json or .jsonl file")
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"{source}: expected a non-empty list of record objects")
    return rows


def _decrypt_mm_record(row: dict) -> dict:
    """Recognize official encrypted questions; plaintext exports also work."""
    question = row.get("question")
    password = row.get("canary")
    if not isinstance(question, str) or not password or question.startswith(MM_QUESTION_PREFIX):
        return row
    key = hashlib.sha256(str(password).encode()).digest()

    def decrypt(text: str) -> str:
        data = base64.b64decode(text, validate=True)
        return bytes(byte ^ key[index % len(key)] for index, byte in enumerate(data)).decode()

    # Ordinary unwrapped questions are left intact. Official ciphertext decrypts
    # to the known prompt wrapper, which makes recognition deterministic.
    try:
        decoded = decrypt(question)
    except (ValueError, UnicodeError):
        if len(question) > 80 and not any(char.isspace() for char in question):
            raise ValueError(f"MM-BrowseComp task {row.get('id')}: question may be encrypted but cannot be decrypted")
        return row
    if not decoded.startswith(MM_QUESTION_PREFIX):
        return row
    result = dict(row)
    result["question"] = decoded
    try:
        if isinstance(result.get("answer"), str):
            result["answer"] = decrypt(result["answer"])
        if isinstance(result.get("checklist"), list):
            result["checklist"] = [decrypt(item) for item in result["checklist"]]
    except (ValueError, UnicodeError, TypeError) as exc:
        raise ValueError(f"MM-BrowseComp task {row.get('id')}: invalid encrypted answer/checklist") from exc
    return result


def _local_image_path(value: str, attachments_dir: Path, task_id: str) -> str:
    root = attachments_dir.resolve()
    parsed = urlparse(value)
    if parsed.scheme in ("http", "https"):
        url_path = unquote(parsed.path)
        if "/MMBC_images/" not in url_path:
            raise ValueError(f"MM-BrowseComp task {task_id}: unsupported image URL {value!r}")
        path = root / "MMBC_images" / url_path.split("/MMBC_images/", 1)[1]
    elif parsed.scheme:
        raise ValueError(f"MM-BrowseComp task {task_id}: unsupported image reference {value!r}")
    else:
        path = Path(value)
        if not path.is_absolute():
            path = root / path
            if not path.is_file() and len(Path(value).parts) == 1:
                path = root / "MMBC_images" / value
    path = path.resolve()
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"MM-BrowseComp task {task_id}: image lies outside attachments_dir") from exc
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"MM-BrowseComp task {task_id}: missing/empty image {path}")
    return relative.as_posix()


def _normalize_mm_records(rows: list[dict], attachments_dir: Path) -> list[dict]:
    normalized = []
    seen = set()
    for number, original in enumerate(rows, 1):
        row = dict(_decrypt_mm_record(original))
        identifier = row.get("id", row.get("task_id"))
        if identifier is None or str(identifier) == "":
            raise ValueError(f"MM-BrowseComp record {number}: id/task_id is required")
        task_id = str(identifier)
        if task_id in seen:
            raise ValueError(f"MM-BrowseComp: duplicate task id {task_id}")
        seen.add(task_id)
        question = row.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"MM-BrowseComp task {task_id}: question is required")
        if question.startswith(MM_QUESTION_PREFIX):
            question = question[len(MM_QUESTION_PREFIX):]
        if not question.strip():
            raise ValueError(f"MM-BrowseComp task {task_id}: empty question after removing wrapper")
        if "answer" not in row:
            raise ValueError(f"MM-BrowseComp task {task_id}: answer is required for scoring")
        images = row.get("images", [])
        if not isinstance(images, list) or any(not isinstance(image, str) or not image.strip() for image in images):
            raise ValueError(f"MM-BrowseComp task {task_id}: images must be a list of non-empty strings")
        row.update(
            id=task_id,
            task_id=task_id,
            dataset_type="MM-BrowseComp",
            question=question.strip(),
            Level=str(row.get("level", row.get("Level", ""))),
            task_type=row.get("subtask", row.get("task_type", "")),
            omni_modal_input=[
                {"type": "image", "id": Path(path).name, "path": path}
                for path in (_local_image_path(image, attachments_dir, task_id) for image in images)
            ],
        )
        # Reference answer/checklist/source remain in records for evaluation;
        # the shared environment exposes only question and modal attachments.
        normalized.append(row)
    return normalized


def prepare_dataset(
    source: Path, attachments_dir: Path, dataset_type: str = "omnigaia",
    output: Path | None = None,
) -> Path:
    """Validate and atomically write a shared-format JSONL before model startup.

    Existing OmniGAIA JSONL paths pass through. Converted content is compared on
    every run, so source edits are reflected while unchanged output is reused.
    """
    source = Path(source).resolve()
    dataset_type = normalize_dataset_type(dataset_type)
    if not source.is_file():
        raise FileNotFoundError(f"Dataset source not found: {source}")
    if dataset_type == "omnigaia" and source.suffix.lower() == ".jsonl":
        return source
    rows = _read_records(source)
    if dataset_type == "mm_browsecomp":
        rows = _normalize_mm_records(rows, Path(attachments_dir))
    if output is None:
        output = (source.with_name(source.stem + ".prepared.jsonl")
                  if dataset_type == "mm_browsecomp" else source.with_suffix(".jsonl"))
    output = Path(output).resolve()
    if output == source:
        raise ValueError("dataset_jsonl_path must differ from the original dataset_path")
    if output.suffix.lower() != ".jsonl":
        raise ValueError("dataset_jsonl_path must have a .jsonl extension")
    content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode("utf-8")
    if output.is_file() and output.read_bytes() == content:
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, prefix=output.name + ".", suffix=".tmp", delete=False) as handle:
            temporary_path = Path(handle.name)
            handle.write(content)
        os.replace(temporary_path, output)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return output
