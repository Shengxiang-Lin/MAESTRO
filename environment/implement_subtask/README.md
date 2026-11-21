# Stage 2 — verified Sub-Agent reports

The **only maintained** production command is:

```bash
python -m environment naturalize-subtasks --config environment/configs/subagent_reports.yaml
```

This runs two ordered, distinct operations:

1. Execute the deterministic Macro Oracle over each accepted Stage-1 Scene.
   The exact evidence rows, operator traces, source references and terminal
   outputs are kept privately under `<report-output>/oracle/`. This cheap
   Oracle is rebuilt on every invocation.
2. Have an LLM Author/Critic produce natural, **stage-local** Sub-Agent
   findings over the immutable executed WorkProduct. The published
   `subtasks.jsonl` contains only approved complete Scene sets; each
   Macro's review is journaled for restart safety.

For the maintained `task_plan` format with
`result_encoding: typed_evidence_json_v1`, the delivered
`finish_result.result` is a **code-owned exact serialization** of actually
returned typed evidence rows and `source_fact_refs`. The model writes the
stage-local human-readable Summary and narrative, but cannot change the
Result, invent values, fabricate source citations, or assert downstream work.
No private Oracle result is substituted when executing Stage-4 faults.

`implementation.scenes` points to the accepted Stage-1 question bank.
`reporting.output_dir` is the single reporting location; the Oracle
input paths are derived from `<report-output>/oracle/` by code and are
**not separate config switches**. `reporting.max_scenes` controls the
published Stage-1 subset in the one-command workflow.
`implementation.max_records` remains available for direct standalone
Oracle inspection but does not override the combined command's Scene cap.

The Stage-2 review is strict: an LLM-failed Macro means no published
complete Scene, not a fallback-positive example. Reviews may be retried
with the same frozen evidence; they may not rewrite Oracle rows or macro
dependencies. Keep failed journals and `summary.json` for diagnostics.

Stage 3 consumes only whole Scenes with every Macro report accepted.
Stage 4 reuses the exact reviewed Stage-2 results when checking its
independent Stage-3 Clean Reference. No real network or media decoding
occurs in this synthetic environment.

For direct offline Oracle inspection only, the internal function is
`environment.implement_subtask.pipeline.materialize_subtasks`.
