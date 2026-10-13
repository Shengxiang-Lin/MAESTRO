"""
Global cost monitoring for LLM calls using contextvars.

Usage:
    with CostMonitor() as monitor:
        await pipeline.run(...)
        print(f"Total cost: ${monitor.total_cost:.4f}")
        monitor.save()  # saves to workspace/costs/<timestamp>.json
"""

import contextvars
import json
import threading
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from base.engine.api_pricing import (
    PRICING_DATE, PRICING_REGION, TOOL_CATEGORIES, USD_TO_CNY, estimate_usage, to_dict,
)

_journal_lock = threading.Lock()
COST_CSV_FIELDS = (
    "image_cost", "audio_cost", "video_cost", "search_cost", "browse_cost",
    "tool_cost", "evaluation_cost", "tracked_cost", "tracked_cost_cny", "total_cost_cny",
    "cost_currency", "usd_to_cny", "cost_complete", "cost_note", "api_call_count",
    "web_search_calls", "pricing_region", "pricing_date", "usage_log",
)

_current_monitor: contextvars.ContextVar["CostMonitor | None"] = contextvars.ContextVar(
    "cost_monitor", default=None
)


@dataclass
class CostRecord:
    """Single LLM call cost record."""

    model: str
    input_tokens: int
    output_tokens: int
    cost: float
    category: str = "agent"
    coverage_complete: bool = True
    is_api: bool = True
    details: dict = field(default_factory=dict)


@dataclass
class CostMonitor:
    """Aggregates LLM costs during a context scope."""

    records: list[CostRecord] = field(default_factory=list)
    _token: contextvars.Token | None = field(default=None, repr=False)
    task_id: str = ""
    journal_path: Path | None = None

    @property
    def total_cost(self) -> float:
        return sum(r.cost for r in self.records)

    @property
    def total_input_tokens(self) -> int:
        return sum(r.input_tokens for r in self.records)

    @property
    def total_output_tokens(self) -> int:
        return sum(r.output_tokens for r in self.records)

    @property
    def call_count(self) -> int:
        return len(self.records)

    def record(self, model: str, input_tokens: int, output_tokens: int, cost: float,
               *, category: str = "agent", coverage_complete: bool = True,
               is_api: bool = True, **details) -> None:
        """Record usage immediately; task scopes also cover async child tasks/threads."""
        record = CostRecord(model, input_tokens, output_tokens, cost, category,
                            coverage_complete, is_api, details)
        with _journal_lock:
            self.records.append(record)
            if self.journal_path:
                event = {
                    "timestamp": datetime.now().isoformat(), "task_id": self.task_id,
                    "model": model, "category": category, "is_api": is_api,
                    "input_tokens": input_tokens, "output_tokens": output_tokens,
                    "cost_usd": cost, "cost_cny": cost * USD_TO_CNY,
                    "coverage_complete": coverage_complete, "details": details,
                    "pricing_date": PRICING_DATE, "pricing_region": PRICING_REGION,
                    "currency": "USD", "usd_to_cny": USD_TO_CNY,
                    "estimate_basis": "public_list_price",
                }
                try:
                    self.journal_path.parent.mkdir(parents=True, exist_ok=True)
                    with self.journal_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
                except OSError as exc:
                    record.coverage_complete = False
                    logging.getLogger(__name__).warning("Could not persist API usage: %s", exc)

    def cost_columns(self, main_cost: float, sub_cost: float) -> dict:
        grouped = {name: sum(r.cost for r in self.records if r.category == name)
                   for name in (*TOOL_CATEGORIES, "evaluation")}
        incomplete = {r.category for r in self.records if not r.coverage_complete}
        tool_cost = sum(grouped[name] for name in TOOL_CATEGORIES)
        tracked = main_cost + sub_cost + tool_cost + grouped["evaluation"]
        complete = not incomplete
        return {
            **{name + "_cost": None if name in incomplete else grouped[name]
               for name in (*TOOL_CATEGORIES, "evaluation")},
            "tool_cost": None if incomplete.intersection(TOOL_CATEGORIES) else tool_cost,
            "tracked_cost": tracked,
            "tracked_cost_cny": tracked * USD_TO_CNY,
            "total_cost": tracked if complete else None,
            "total_cost_cny": tracked * USD_TO_CNY if complete else None,
            "cost_currency": "USD", "usd_to_cny": USD_TO_CNY,
            "cost_complete": complete,
            "cost_note": "; ".join(sorted({
                r.category + ": " + note
                for r in self.records if not r.coverage_complete
                for note in (r.details.get("notes") or ["usage_not_complete"])
            })),
            "api_call_count": sum(r.is_api for r in self.records),
            "web_search_calls": sum(r.details.get("web_search_calls", 0) for r in self.records),
            "pricing_region": PRICING_REGION, "pricing_date": PRICING_DATE,
            "usage_log": str(self.journal_path) if self.journal_path else "",
        }

    def summary(self) -> dict:
        """Get aggregated summary."""
        return {
            "total_cost": self.total_cost,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "call_count": self.call_count,
            "by_model": self._group_by_model(),
            "coverage_complete": all(r.coverage_complete for r in self.records),
            "currency": "USD", "usd_to_cny": USD_TO_CNY,
            "by_category": {
                name: sum(r.cost for r in self.records if r.category == name)
                for name in sorted({r.category for r in self.records})
            },
        }

    def _group_by_model(self) -> dict[str, dict]:
        result: dict[str, dict] = {}
        for r in self.records:
            if r.model not in result:
                result[r.model] = {"cost": 0.0, "input_tokens": 0, "output_tokens": 0, "calls": 0}
            result[r.model]["cost"] += r.cost
            result[r.model]["input_tokens"] += r.input_tokens
            result[r.model]["output_tokens"] += r.output_tokens
            result[r.model]["calls"] += 1
        return result

    def save(self, save_dir: str = "workspace/costs") -> Path:
        """Save cost summary to JSON file."""
        cost_dir = Path(save_dir)
        cost_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = cost_dir / f"{ts}.json"
        path.write_text(
            json.dumps(self.summary(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return path

    def __enter__(self) -> "CostMonitor":
        self._token = _current_monitor.set(self)
        return self

    def __exit__(self, *args) -> None:
        if self._token is not None:
            _current_monitor.reset(self._token)


def get_current_monitor() -> "CostMonitor | None":
    """Get the current cost monitor from context, if any."""
    return _current_monitor.get()


def record_cost(model: str, input_tokens: int, output_tokens: int, cost: float, **details) -> None:
    """Record cost to the current monitor if one is active."""
    monitor = get_current_monitor()
    if monitor is not None:
        try:
            monitor.record(model, input_tokens, output_tokens, cost, **details)
        except Exception as exc:
            # A bookkeeping failure must never retry a completed paid request.
            monitor.records.append(CostRecord(model, 0, 0, 0.0,
                details.get("category", "agent"), False, details.get("is_api", True),
                {"notes": ["usage_recording_failed"]}))
            logging.getLogger(__name__).warning("Could not record API usage: %s", exc)


def record_api_response(model: str, response, *, category: str, base_url=None,
                        audio_input=False, search_requested=False) -> dict:
    """Capture only provider usage/request IDs; never prompts or credentials."""
    usage = response.get("usage") if isinstance(response, dict) else getattr(response, "usage", None)
    request_id = (response.get("request_id") or response.get("id")) if isinstance(response, dict) else getattr(response, "id", None)
    try:
        estimate = estimate_usage(model, usage, base_url=base_url, audio_input=audio_input,
                                  search_requested=search_requested)
    except Exception as exc:
        record_unmeasured_call(model, category, "provider_usage_could_not_be_parsed")
        logging.getLogger(__name__).warning("Could not parse provider usage: %s", exc)
        return {"coverage_complete": False, "total_cost": None}
    record_cost(
        model, estimate["input_tokens"], estimate["output_tokens"], estimate["total_cost"],
        category=category, coverage_complete=estimate["coverage_complete"],
        request_id=request_id, raw_usage=to_dict(usage), notes=estimate["notes"],
        web_search_calls=estimate["web_search_calls"],
        cached_input_tokens=estimate["cached_input_tokens"],
        audio_input_tokens=estimate["audio_input_tokens"],
        audio_output_tokens=estimate["audio_output_tokens"],
    )
    return estimate


def record_unmeasured_call(model: str, category: str, reason: str) -> None:
    record_cost(model, 0, 0, 0.0, category=category, coverage_complete=False,
                notes=[reason])
