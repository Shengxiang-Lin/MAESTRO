# Trusted Sub-Agent report generation

```bash
python -m environment naturalize-subtasks --config environment/configs/subagent_reports.yaml
```


## Oracle execution and restart

`naturalize-subtasks` rebuilds its deterministic Oracle under
`<report-output>/oracle/` on every invocation. There is no Oracle cache
manifest or Scene/code/parameter fingerprint gate, and stale Oracle files
are overwritten automatically. This stage makes no model calls.

The **LLM report journal** is still protected separately: an interrupted run
can resume if its reviewed-report inputs and model settings are unchanged.
Previously completed LLM report output is not silently overwritten; use a
fresh output directory or explicitly enable `reporting.overwrite` when
changing the report inputs after a completed run.

## Current Task Plan: source-cited Result

For current `task_plan` Scenes, Stage 2 retains the exact Oracle WorkProduct
and writes a complete, deterministic JSON-list string into
`finish_result.result`. Each entry contains `values` and
`source_fact_refs`. The LLM writes the human-readable `summary` only.
No row or citation is truncated to meet a text budget; oversized reports
fail visibly. Legacy Scene reports keep their original readable Result.

## Stage-1 observations and natural Stage-2 traces

Stage 1 already creates `observed_content`, `observation_semantics`,
`document_context`, and named source titles/purposes/modalities within
synthetic World observations. Oracle realization now carries this **existing**
evidence into `trace_summary.steps[].evidence_atoms`, plus a scoped
`trace_summary.source_context` catalog. The original operator steps, full
WorkProduct, and provenance remain code-owned.

For current `result_encoding: typed_evidence_json_v1` tasks, one Qwen
Author invocation returns **both** `summary` and `trace_narrative`.
The Guard checks each field, and the independent Critic sees both alongside
the exact evidence. Only the accepted narrative is published as
`trace_summary.narrative` and shown to Main in Stage 3. The old operator
paraphrase function remains a fallback for older, non-typed reporting modes.
The frozen `finish_result.result` and fact references never change.

No actual image, audio, video, network or tool execution occurs in Stage 2;
the natural narrative must describe simulated, supplied observations. The
earlier Stage-1 `evidence_acquisition` contract remains optional: without it,
do **not** synthesize a fictitious tool-call history.

## Code-owned results and stage-local summaries

`reporting.result_policy: code_owned` (in `environment/configs/subagent_reports.yaml`)
enables a safer V1.4 report format. The model authors **only**
`finish_result.summary` from the current Macro's frozen step/evidence
contract; code renders `finish_result.result` and `status=done` from the
complete currently delivered WorkProduct. The model is not asked to describe
the overall question; the complete frozen question can contain downstream
conditions that a current Macro has not checked.

For example, a context-log filter returns candidate case IDs, NOT
independently confirmed report-year disagreements. Its Result enumerates the
actual candidate IDs, while the LLM's Summary must describe only context-log
filtering. Both the code result and authored summary run the normal Guard and
independent Critic. Existing scenes without the V1.4 marker retain the legacy
report mode.

The config's `implementation.scenes` is the accepted Question bank.
Without `--scenes`, the command automatically reads that path. This is
the **one-command** workflow:

1. Build exact Oracle Macro WorkProducts and operator/evidence traces from
   frozen Scenes, replaying source constraints into
   `<report-output>/oracle/`.
2. Naturalize every Macro's independent Finish Result and trace narrative
   with the configured Author/Critic models, preserving immutable Oracle
   answers, provenance and tools. Publish completed accepted Scenes into
   `<report-output>/subtasks.jsonl`.

The `reporting` YAML section owns model endpoints, concurrency, revisions,
output path, and resume behavior. Internal status, failure reasons and
source-preservation checks are written to summary and failure JSONL files.
Nothing here represents an actual Web/media tool invocation.

For offline Oracle-only inspection, use the internal
`environment.implement_subtask.pipeline.materialize_subtasks` function.
The maintained public Stage-2 command is `naturalize-subtasks`, which runs
Oracle execution and Author/Critic review together.
