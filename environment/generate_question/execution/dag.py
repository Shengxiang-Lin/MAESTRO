"""Deterministic dependency waves for source-scoped Macro Tasks."""
from __future__ import annotations

MAX_CONCURRENCY = 10


class MacroPlanError(ValueError):
    pass


def compile_waves(tasks: list[dict], *, max_concurrency: int = MAX_CONCURRENCY) -> list[list[str]]:
    if max_concurrency != MAX_CONCURRENCY:
        raise MacroPlanError("this experiment fixes maximum concurrency at 10")
    by_id = {t["task_id"]: t for t in tasks}
    if len(by_id) != len(tasks):
        raise MacroPlanError("duplicate macro task IDs")
    completed, waves = set(), []
    while len(completed) < len(tasks):
        ready = sorted(
            tid for tid, item in by_id.items()
            if tid not in completed and set(item["depends_on"]) <= completed
        )
        if not ready:
            raise MacroPlanError("missing dependency or cycle")
        for start in range(0, len(ready), max_concurrency):
            wave = ready[start:start + max_concurrency]
            waves.append(wave)
            completed.update(wave)
    return waves


