"""Lossless, cited public record prose shared by rendering and visibility checks."""
from __future__ import annotations

import json
import re

_HEADER = "The returned records are:\n"
_RECORD = re.compile(r"Record (\d+) contains ")
_FIELD = re.compile(r"([a-zA-Z][a-zA-Z0-9_ ]*) = ")


def render_records(rows: list[dict]) -> str:
    """Keep every delivered value and citation in Main-visible record sentences."""
    if not rows:
        return "No records matched the assigned task."
    lines = []
    for index, row in enumerate(rows, 1):
        fields = "; ".join(
            key.replace("_", " ") + " = " + json.dumps(value, ensure_ascii=False)
            for key, value in row["values"].items()
        )
        refs = ", ".join(row.get("source_fact_refs", [])) or "none"
        lines.append(f"Record {index} contains {fields}. Evidence: {refs}.")
    return _HEADER + "\n".join(lines)


def parse_records(text: str) -> list[dict] | None:
    """Parse only explicit public records, including nested final answers and nulls."""
    if text == "No records matched the assigned task.":
        return []
    if not isinstance(text, str) or not text.startswith(_HEADER):
        return None
    decoder = json.JSONDecoder()
    rows = []
    for index, line in enumerate(text[len(_HEADER):].splitlines(), 1):
        match = _RECORD.match(line)
        if match is None or int(match[1]) != index:
            return None
        position = match.end()
        values = {}
        while True:
            field = _FIELD.match(line, position)
            if field is None:
                return None
            name = field[1].replace(" ", "_")
            if name in values:
                return None
            try:
                value, length = decoder.raw_decode(line[field.end():])
            except ValueError:
                return None
            values[name] = value
            position = field.end() + length
            if line.startswith("; ", position):
                position += 2
                continue
            if not line.startswith(". Evidence: ", position) or not line.endswith("."):
                return None
            refs = line[position + len(". Evidence: "):-1]
            refs = [] if refs == "none" else refs.split(", ")
            if any(not re.fullmatch(r"[A-Za-z0-9_.-]+:[A-Za-z0-9_.-]+", ref)
                   for ref in refs):
                return None
            rows.append({"values": values, "source_fact_refs": refs})
            break
    return rows or None


def delivered_records(result: str) -> list[dict] | None:
    """Read cited record prose or existing JSON deliveries without consulting the World."""
    parsed = parse_records(result)
    if parsed is not None:
        return parsed
    try:
        rows = json.loads(result)
    except (TypeError, ValueError):
        return None
    if isinstance(rows, dict):
        rows = [rows]
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        return None
    return [
        row if isinstance(row.get("values"), dict) else
        {"values": row, "source_fact_refs": []}
        for row in rows
    ]
