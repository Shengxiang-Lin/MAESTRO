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

## Current Task Plan: O1-style Answer, Summary and Trace Summary

For the maintained `task_plan` and `result_encoding:
typed_evidence_json_v1` Scenes, Stage 2 uses two representations:

- **Offline Oracle (private):** `answer`, `answer_with_provenance`,
  `trace_summary.steps`, and `answer_text` (the exact typed JSON-string
  serialization). These are immutable and remain available for Replay,
  fact checking, fault construction and regression tests.
- **Main-visible delivery:** `finish_result.result` is the natural-language
  **Answer**, `finish_result.summary` briefly explains what this delegated
  task examined, found and left unresolved, while
  `trace_summary.narrative` records the more detailed execution evidence
  and sequence. Stage 3 renders all three in its original O1 history.

One Qwen Author request now returns exactly `{"answer": "...",
"summary": "...", "trace_summary": "..."}`; it is still **one API call**.
This matches the information hierarchy in real Orchestra-o1 traces:
`finish.summary` is an informative 1–3 sentence handoff, not a generic
completion message or the first trace bullet. The longer, task-adaptive
`trace_summary` retains actual supported observations, checks, outcomes
and limitations. The Guard validates all three prose fields for grounded
stage-local claims; the independent Critic checks their consistency.
When applicable, the existing code-owned, source-scoped witnessed digest
may be appended to `finish.summary` after approval, without altering
Oracle facts.

For these **clean, successful** Stage-2 reports, natural O1-style positive
language is welcome: the Author may describe an observation as successfully
recovered, an identifier as correctly identified or the task as smoothly
completed. It may briefly note no apparent issues if supported by the
current evidence. Such language is **optional and model-authored**, never a
post-processing template or required stock sentence. The Critic must not
reject genuine successful wording merely for being affirmative, but the
existing evidence and tool-use guards still reject fabricated web/media
access, fake verification steps, unsupported efficiency measurements,
incorrect source claims and premature downstream success.

The Guard requires all distinct identifiers,
dates and numeric scalar values from the frozen delivered rows to occur
verbatim in the Answer (up to 6,500 characters), and rejects invented
tool/media execution or internal procedural identifiers. The independent
Critic checks whether the Answer correctly associates entities with
their values and whether both texts are grounded and stage-local.
An answer missing a required value is repaired or rejected; the Oracle
is never changed merely to make a report pass. Any Macro (including intermediate ID lookups) may
return its exact single-row, single-field scalar directly (e.g.
`PRJ-6C3E1089F38D`, `132` or a short date/name) even when it is shorter
than 18 characters; this narrow exception requires an exact match to the
immutable Oracle value. When the local model returns a JSON number instead
of a string, the code converts it to canonical text only after verifying
the same exact one-field Oracle value. A structured multi-field object
is rendered as natural-language Answer text **only if** every key, value
and type matches the complete current Oracle answer. This includes plans
whose Oracle wraps a typed payload in a single `answer` field: when the
Author supplies the exact inner object, it can safely be converted to
a readable field/value sentence. Partial, modified or extra-field objects
still fail the Guard, and the result remains independently Critic-reviewed.
A private `verified_structured_answer_rendered` flag records conversion.

For multi-source investigations, the immutable execution trace supplies
`source_scoped_filter_steps` to the Critic: source-local filtering steps
are paired to their actual source reads. A later filter on source B does not
invalidate an earlier, different filter on source A. The reporter allows
visible source identifiers such as `M3_a` and `relation_2` **only when
listed in the current Macro's visible evidence-source catalog**.
Invented IDs, private operator IDs, `Macro` or `Oracle` leakage and
unperformed actions still fail. Revision hints explain how to remove
internal wording without concealing authentic source distinctions.

Before the exact-value Guard and independent Critic, the reporter
normalizes a small whitelist of *surface-only* operator-style wording
(e.g. "Scanned" -> "Reviewed", "Joined" -> "Cross-referenced",
"Projected" -> "Retained", "rows" -> "records"). This does not change
structured rows, IDs, measured values or provenance. Internal source keys,
private Oracle claims, invented tool actions and premature downstream
conclusions are still rejected by the original checks.

The full Oracle JSON is **not** inserted into Main's `finish_result.result`.
Stage 4 similarly renders actually delivered *faulty* or *partial* typed
rows into readable prose, never substituting the private clean result.
Its visible-claim parser recovers only explicitly communicated facts:
structured prose it can parse, or literal case-ID lists; otherwise it
abstains rather than inferring hidden facts.

## Reusing Stage-1 observation language

Stage 1 has already generated `observed_content`,
`observation_semantics`, `document_context`, and source titles,
purposes and modalities within its synthetic World. Stage 2 passes
these through the scoped `trace_summary.steps[].evidence_atoms`
and `trace_summary.source_context`. They ground the Answer and Trace,
rather than forcing the model to paraphrase `scan/project/join` names.

**No real image, audio, video, network or tool execution happens here.**
Without Stage 1 evidence-acquisition contracts, do not fabricate a tool
call history. The natural Trace reports work performed over supplied
simulated observations and only limitations established by that work.

## Report generation and resumability

The single maintained reporting path keeps status and exact typed
evidence owned by the application; it no longer means that the Main-visible
Answer is a raw JSON WorkProduct. Non-typed legacy report modes are
left as compatibility paths. An interrupted Author/Critic journal may
resume with unchanged inputs, model and prompt; use a fresh report
output directory or `reporting.overwrite: true` for a full rerun.

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
