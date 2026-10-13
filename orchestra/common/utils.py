"""Common utility functions."""
from __future__ import annotations
import json
import re
from typing import Any, Dict, List


def parse_json_response(resp: str) -> Dict[str, Any]:
    s = resp.strip()
    if "```" in s:
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", s, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                pass
    start, end = s.find("{"), s.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(s[start:end + 1])
        except json.JSONDecodeError:
            pass
        for end_pos in range(end, start - 1, -1):
            if s[end_pos] == '}':
                try:
                    return json.loads(s[start:end_pos + 1])
                except json.JSONDecodeError:
                    continue
    if start != -1:
        candidate = s[start:]
        missing = candidate.count('{') - candidate.count('}')
        if missing > 0:
            try:
                return json.loads(candidate + '}' * missing)
            except json.JSONDecodeError:
                pass
    raise json.JSONDecodeError(f"Failed to parse JSON response: {s[:200]}...", s, 0)


def indent_text(text: str, indent: str = "   ") -> str:
    return "\n".join(indent + line for line in text.strip().split("\n"))


def format_tools_description(tools: List[Any], verbose: bool = False) -> str:
    if not tools:
        return "No tools available."
    if verbose:
        return "\n\n".join(f"Tool Name: {t.name}\nDescription: {t.description}\nParameters: {json.dumps(t.parameters, indent=2)}" for t in tools)
    return "\n\n".join(f"{t.name}: {t.description}\nParams: {json.dumps(t.parameters, indent=2)}" for t in tools)
