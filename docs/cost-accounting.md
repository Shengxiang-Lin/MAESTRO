# API cost accounting

Prices were checked on 2026-10-13 for the Beijing DashScope endpoint. Values
are public list-price estimates, before account discounts, credits, taxes or
settlement adjustments. `cost_complete` describes whether all observed API
calls had usable token/modality/tool usage and a verified price; it does not
mean that Alibaba Cloud has confirmed the amount.

| Model | Ordinary input CNY / million tokens | Output CNY / million tokens | Cached input CNY / million tokens |
| --- | ---: | ---: | ---: |
| qwen3.8-max | 12 | 36 | 1.5 |
| qwen3.7-plus, input <= 256000 | 2 | 8 | 0.4 |
| qwen3.7-plus, input > 256000 | 6 | 24 | 1.2 |
| qwen3.7-flash, input <= 32000 | 0.2 | 0.8 | 0.04 |
| qwen3.7-flash, 32000 < input <= 256000 | 0.6 | 2.4 | 0.12 |
| qwen3.7-flash, input > 256000 | 1.2 | 4.8 | 0.24 |

Qwen3.5-Omni-Flash charges 2.2 CNY / million text/image/video input tokens,
18 CNY / million audio input tokens, and 13.3 CNY / million text output tokens.
For text+audio output it charges only audio output, at 72 CNY / million tokens.
Responses API web search adds 4 CNY / 1000 searches, using the provider's
`usage.x_tools.web_search.count` (or equivalent nested plugin statistics).

## Result columns

`main_cost`, `sub_cost`, `image_cost`, `audio_cost`, `video_cost`, `search_cost`,
`browse_cost`, `tool_cost`, `evaluation_cost`, `tracked_cost`, and `total_cost`
are USD estimates. The project retains its historical conversion of
**7 CNY per USD**, recorded in `usd_to_cny`; this is not a live exchange rate.
`tracked_cost_cny` and `total_cost_cny` expose CNY amounts directly.

`tool_cost` is the sum of image, audio, video, search and browse costs. Video
costs include frame analysis, audio-track analysis and combined summaries.
`total_cost = main_cost + sub_cost + tool_cost + evaluation_cost`.
Local vLLM inference has zero provider API cost; GPU rental is excluded.

Missing usage, unsupported prices or unmeasured failed requests set
`cost_complete=False`. The affected cost categories and complete totals are
blank, while `tracked_cost` retains the observed subtotal. Costs already
incurred are retained even if a benchmark task later fails.

Every observed response is recorded immediately in `api_usage_<timestamp>.jsonl`,
with task ID, model, category, request ID, original provider usage and prices.
Concurrent benchmark tasks have separate accounting contexts; child async tasks
and `asyncio.to_thread` tool calls inherit the relevant context.

Existing processes keep the code loaded at startup. They must be restarted to
produce the new accounting columns. Historical tool usage that was discarded
cannot be reconstructed from the existing CSV; compare the provider's billing
records for those calls.

## Reviewed sources

- [Qwen3.8-Max](https://help.aliyun.com/zh/model-studio/qwen3-8-max)
- [Qwen3.7-Plus](https://help.aliyun.com/zh/model-studio/qwen3-7-plus)
- [Qwen3.7-Flash](https://help.aliyun.com/zh/model-studio/qwen3-7-flash)
- [Qwen3.5-Omni-Flash](https://help.aliyun.com/zh/model-studio/qwen3-5-omni-flash)
- [Web search](https://help.aliyun.com/zh/model-studio/web-search)
