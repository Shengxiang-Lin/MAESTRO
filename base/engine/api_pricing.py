"""Reviewed Beijing DashScope list prices, in CNY per million tokens.

Checked 2026-10-13 against Alibaba Cloud's model-information and web-search
pages. These are estimates before account discounts, credits, and settlement.
USD columns retain the project's historical conversion of 7 CNY per USD.
"""
from __future__ import annotations

USD_TO_CNY = 7.0
PRICING_DATE = "2026-10-13"
PRICING_REGION = "cn-beijing"
TOOL_CATEGORIES = ("image", "audio", "video", "search", "browse")
SOURCE_URLS = {
    "qwen3.8-max": "https://help.aliyun.com/zh/model-studio/qwen3-8-max",
    "qwen3.7-plus": "https://help.aliyun.com/zh/model-studio/qwen3-7-plus",
    "qwen3.7-flash": "https://help.aliyun.com/zh/model-studio/qwen3-7-flash",
    "qwen3.5-omni-flash": "https://help.aliyun.com/zh/model-studio/qwen3-5-omni-flash",
    "web_search": "https://help.aliyun.com/zh/model-studio/web-search",
}
# Upper input-token threshold, ordinary input, output, cached input.
TEXT_PRICES = {
    "qwen3.8-max": ((None, 12.0, 36.0, 1.5),),
    "qwen3.7-plus": ((256000, 2.0, 8.0, 0.4), (None, 6.0, 24.0, 1.2)),
    "qwen3.7-flash": (
        (32000, 0.2, 0.8, 0.04),
        (256000, 0.6, 2.4, 0.12),
        (None, 1.2, 4.8, 0.24),
    ),
}


def to_dict(value):
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    return vars(value) if hasattr(value, "__dict__") else {}


def reviewed_model(model):
    for name in (*TEXT_PRICES, "qwen3.5-omni-flash"):
        if model == name or model.startswith(name + "-"):
            return name
    return None


def normalize_usage(usage):
    raw = to_dict(usage)
    inputs = raw.get("prompt_tokens", raw.get("input_tokens"))
    outputs = raw.get("completion_tokens", raw.get("output_tokens"))
    input_detail = to_dict(raw.get("prompt_tokens_details") or raw.get("input_tokens_details"))
    output_detail = to_dict(raw.get("completion_tokens_details") or raw.get("output_tokens_details"))
    # Responses API may return modality details under usage.x_details.
    details = raw.get("x_details") or []
    if details and not input_detail:
        for item in details:
            for key, value in to_dict(to_dict(item).get("input_tokens_details")).items():
                if isinstance(value, (int, float)):
                    input_detail[key] = input_detail.get(key, 0) + value
    if details and not output_detail:
        for item in details:
            for key, value in to_dict(to_dict(item).get("output_tokens_details")).items():
                if isinstance(value, (int, float)):
                    output_detail[key] = output_detail.get(key, 0) + value
    tools = to_dict(raw.get("x_tools") or raw.get("plugins"))
    if not tools:
        for item in details:
            for name, info in to_dict(to_dict(item).get("plugins")).items():
                count = to_dict(info).get("count", 0)
                tools.setdefault(name, {"count": 0})["count"] += count or 0
    return {
        "input_tokens": int(inputs or 0),
        "output_tokens": int(outputs or 0),
        "cached_input_tokens": int(input_detail.get("cached_tokens") or 0),
        "audio_input_tokens": int(input_detail.get("audio_tokens") or 0),
        "audio_output_tokens": int(output_detail.get("audio_tokens") or 0),
        "web_search_calls": int(to_dict(tools.get("web_search")).get("count") or 0),
        "has_audio_input_details": "audio_tokens" in input_detail,
        "has_audio_output_details": ("audio_tokens" in output_detail
                                     or output_detail.get("text_tokens") == outputs),
        "has_search_details": "web_search" in tools,
        "has_token_usage": inputs is not None and outputs is not None,
        "raw_usage": raw,
    }


def estimate_usage(model, usage, *, base_url=None, audio_input=False, search_requested=False):
    normalized = normalize_usage(usage)
    name = reviewed_model(model)
    notes = []
    if not normalized["has_token_usage"]:
        notes.append("missing_token_usage")
    if name is None:
        notes.append("unverified_model_price")
    if base_url and "dashscope.aliyuncs.com" not in base_url:
        notes.append("endpoint_price_not_verified_for_beijing")
    inputs, outputs = normalized["input_tokens"], normalized["output_tokens"]
    cached = min(inputs, max(0, normalized["cached_input_tokens"]))
    input_cny = output_cny = 0.0
    if name in TEXT_PRICES:
        _, input_rate, output_rate, cached_rate = next(
            tier for tier in TEXT_PRICES[name] if tier[0] is None or inputs <= tier[0]
        )
        input_cny = ((inputs - cached) * input_rate + cached * cached_rate) / 1_000_000
        output_cny = outputs * output_rate / 1_000_000
    elif name == "qwen3.5-omni-flash":
        audio_in = min(inputs, max(0, normalized["audio_input_tokens"]))
        audio_out = max(0, normalized["audio_output_tokens"])
        if audio_input and not normalized["has_audio_input_details"]:
            notes.append("missing_audio_input_breakdown")
        if audio_input and not normalized["has_audio_output_details"]:
            notes.append("missing_audio_output_breakdown")
        input_cny = ((inputs - audio_in) * 2.2 + audio_in * 18.0) / 1_000_000
        # With text+audio output, only output audio tokens are charged.
        output_cny = audio_out * 72.0 / 1_000_000 if audio_out else outputs * 13.3 / 1_000_000
    if search_requested and not normalized["has_search_details"]:
        notes.append("missing_web_search_count")
    search_cny = normalized["web_search_calls"] * 4.0 / 1000
    total_cny = input_cny + output_cny + search_cny
    return {
        **normalized,
        "input_cost": input_cny / USD_TO_CNY,
        "output_cost": output_cny / USD_TO_CNY,
        "search_surcharge": search_cny / USD_TO_CNY,
        "total_cost": total_cny / USD_TO_CNY,
        "cost_cny": total_cny,
        "coverage_complete": not notes,
        "notes": notes,
        "pricing_date": PRICING_DATE,
        "pricing_region": PRICING_REGION,
        "pricing_source": SOURCE_URLS.get(name),
    }
