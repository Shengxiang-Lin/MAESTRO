"""Format simulated clean Sub-Agent feedback exactly as O1 Main task history.

The baseline O1 reads the cumulative task_entries list and renders a
human-readable SUBTASK HISTORY, including results, summaries and compressed
traces from every preceding attempt. The gold simulator must present the same
view, rather than serializing private supervision or duplicating raw JSON.
"""
from __future__ import annotations

from typing import Any, Mapping


def _indent(text: str, prefix: str) -> str:
    return "\n".join(prefix + line for line in text.split("\n"))


def _result_line(value: Any) -> str:
    if value and value != "-":
        return f'"{value}"'
    return "(no result)"


def format_o1_subtask_history(trajectory: list[Mapping[str, Any]]) -> str:
    """Cumulative O1 Main-visible history, never just the most recent round."""
    entries = [step for step in trajectory if step.get("action") == "delegate_task"]
    if not entries:
        return "No subtasks completed yet."

    lines: list[str] = []
    done = 0
    total = 0
    for step in entries:
        attempt = step.get("attempt", "?")
        result = step.get("result", {})
        subtasks = result.get("subtask_results", ()) if isinstance(result, Mapping) else ()
        if not isinstance(subtasks, (list, tuple)):
            subtasks = ()
        total += len(subtasks)

        # The gold simulator produces one native O1 parallel batch per round.
        batch = [f"[Attempt {attempt}] 🔄 Parallel Batch ({len(subtasks)} subtasks)"]
        for index, sub in enumerate(subtasks):
            finish = sub.get("finish_result") or {}
            status = str(finish.get("status", "partial"))
            model = sub.get("model", "?")
            # Simulation deliberately does not invent tool steps. O1's
            # formatter already uses '?' when steps_taken is unspecified.
            steps = f'{sub.get("steps_taken", "?")}/{sub.get("max_steps", 30)}'
            emoji = "✅" if status == "done" else "⚠️"
            batch.append(
                f"  [{index+1}] {emoji} {status} | Model: {model} | Steps: {steps}"
            )
            batch.append(f'  ├─ Task: {sub.get("task_instruction", "N/A")}')
            batch.append(f'  ├─ Result: {_result_line(finish.get("result", ""))}')
            if finish.get("summary"):
                batch.append(f'  ├─ Summary: {finish["summary"]}')
            trace = sub.get("trace_summary", "")
            if trace and trace != "N/A":
                batch.append(f'  └─ Trace summary:\n{_indent(str(trace), "     ")}')
            else:
                batch[-1] = batch[-1].replace("├─", "└─")
            if status == "done":
                done += 1
        lines.append("\n".join(batch))
    lines.append(f"---\nSummary: {done}/{total} subtasks done")
    return "\n\n".join(lines)


__all__ = ["format_o1_subtask_history"]
