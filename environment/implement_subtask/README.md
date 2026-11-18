# Stage 2 — reviewed Sub-Agent reports

The maintained entry point is:

```bash
python -m environment naturalize-subtasks \
  --config environment/configs/subagent_reports.yaml
```

## One current evidence/report format

Stage 2 accepts the current Stage-1 `task_plan` with
`result_encoding: typed_evidence_json_v1`. There is **one** report path:
`review_subagent_report` in `reporting.py`. No
`reporting.result_policy` selection and no legacy `llm_authored` report
mode remain.

1. `combined.generate_subagent_reports` executes the frozen Scene's
   Macro tasks through `pipeline.materialize_subtasks` under
   `<output>/oracle/`. This deterministic step has no LLM calls.
2. `naturalize.review_subagent_reports` issues one Author request per
   Macro, asking for exactly `answer`, `summary`, and `trace_summary`.
   A separate Critic reviews those strings against **this Macro's**
   immutable evidence and local operator scope. Retries repair unsupported
   prose; exhausted attempts fail instead of silently publishing a fallback.
3. Only complete accepted Scenes are published to
   `<output>/subtasks.jsonl`. The reviewed public handoff contains
   `finish_result.result` (natural Answer), `finish_result.summary`
   (short handoff), and `trace_summary.narrative` (detailed evidence
   sequence). Stage 3 renders these into the real O1-format history.

The exact `answer`, `answer_with_provenance`,
`trace_summary.steps` and source FactRefs remain frozen.
`reporting.serialize_verified_result` writes the complete typed relation
to `answer_text` **for private audit/replay**; it is not substituted for
the natural Main-visible Answer. Stage 4 can therefore compare or
reconstruct faulty feedback without rewriting the trusted Oracle.

## Grounding invariants

- The Answer must preserve every distinct scalar identifier, date, number
  and other required value from the current Macro. A precise one-field
  result (including a short ID or number) is valid without filler.
- Structured Author output is accepted as a human-readable Answer **only**
  after exact typed equality with the immutable relation. Partial,
  type-changed, altered and extra-field objects fail.
- The Guard rejects invented sources, tool/media execution, private operator
  identifiers, source-FactRef misattribution, premature downstream checks,
  and incorrect stage-local filtering claims. A public source handle is
  permitted only when present in this Macro's visible source catalog.
- The Critic receives source-scoped observations and the exact local steps
  rather than treating the whole user question as already completed.
  Natural statements of successful work are fine when supported by
  actual current-stage evidence.
- A narrow surface normalizer replaces technical phrasing such as
  "Joined" with "Cross-referenced" without modifying identifiers,
  numerical values, trace facts, or provenance.

The synthetic image/audio/video annotations and source handles are
**simulated evidence**, not records of actual network or media-tool calls.

## Resuming and testing

The deterministic Oracle is rebuilt from the latest accepted Scene bank on
each one-command run, without trusting a stale Oracle cache. The reviewed
report writer has a separate crash-safe, signature-checked journal; interrupted
runs can resume if the model, source files and prompt code are unchanged.
Use a new output directory after changing any of these, or explicitly enable
`reporting.overwrite`. Incomplete Scenes remain private.

Useful focused checks:

```bash
python -m unittest environment.tests.test_subtask_reporting \
  environment.tests.test_combined_stage2 \
  environment.tests.test_stage3_task_contract
```

The CLI name `naturalize-subtasks` and the Stage-2 output schema remain
unchanged. The internal Python functions use current, explicit names.
