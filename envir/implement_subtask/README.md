# MAESTRO Stage 2 — Implement Subtask

The Question-generation stage already executed every Macro Task and stored
its exact workproduct and provenance under task_feedback; it also froze the
fine-grained gold_trace and source observations. Stage 2 materializes this
existing oracle evidence into one consistent per-Macro Sub-Agent response
without inventing new World values or asking a model to recalculate answers.

## Run

Edit [config.yaml](config.yaml) to select a private **accepted**
Stage-1 `questions/scenes.jsonl` and a new output directory. Then:

~~~bash
python -m envir implement-subtasks \
  --config envir/implement_subtask/config.yaml
~~~

Or override just the dataset and output directory:

~~~bash
python -m envir implement-subtasks \
  --config envir/implement_subtask/config.yaml \
  --scenes envir/data/maestro_macro_question_aligned_91/scenes.jsonl \
  --output-dir envir/data/maestro_stage2_aligned_91
~~~

The output files are:

- `subtasks.jsonl`: **one answer + trace_summary per Macro Task**
- `scene_index.jsonl`: frozen Question, terminal Macro and dependency waves
- `failed.jsonl`: private invalid scene records, never counted as success
- `summary.json`: scene/Macro counts and validation results

### Subtask record

`answer` contains the exact typed output rows (possibly an empty list).
`answer_with_provenance` preserves the fact references of every output row.
`trace_summary` includes a plain-English objective and grounded summary,
plus `steps`: primitive operator, exact input dependencies, observed row
count, schema, source fact references and sampled underlying evidence atoms.

Evidence atoms are read from the frozen synthetic World and checked against
the original fact IDs. `evidence_limit` caps only the *inlined* raw evidence
per primitive step; the complete fact-reference list is retained. If the
evidence is truncated, this is explicitly recorded.

**This is a deterministic oracle/reference Sub-Agent trace, not a real
LLM Sub-Agent's autonomous reasoning transcript.** It can serve as a
gold-grounded training/evaluation target for a later LLM naturalization
or action-taking Sub-Agent. Such naturalization must preserve answer,
source permissions and fact references, and should be reviewed separately.

The entire accepted Stage-1 scene is validated by strict replay before
any of its Macros is exported. If a Macro fails, the entire scene is
withheld, with a reason in `failed.jsonl`. The Stage-1 scene, Program,
World, and Macro DAG remain unmodified.

Stage 3 (Subtask composition & scheduling) is not implemented here.
It can consume `subtasks.jsonl` and `scene_index.jsonl` to simulate
dependency-respecting handoffs.


## Natural-language Sub-Agent finish results (Author + Critic)

In the two reference OmniGAIA runtime trajectories, a Sub-Agent returns a
compact finish contract with a detailed result, a completion status, and a
short evidence-oriented summary:

~~~json
{
  "result": "Concrete findings and the supporting observations",
  "status": "done",
  "summary": "Actions taken, what was established, and significant limitations."
}
~~~

A true runtime Sub-Agent can return status partial when evidence is missing.
For this Stage-2 fully executed oracle, even an empty matching relation is
a complete answer (done), not a partial or invented success.

To mimic the real Main-Agent-facing format, Stage 2 now offers a second
pass that works directly from the ALREADY EXPORTED oracle Subtasks. LLM
Author writes the natural result/summary; independent LLM Critic reviews
grounding, unchanged answer scope, source permissions and natural style.
A deterministic guard blocks fabricated tool/media use, invented fact IDs,
internal Teacher jargon and incorrect completion status. Revisions receive
focused previous failure feedback.

New report output preserves, UNCHANGED, the exact typed answer,
answer_with_provenance, and all trace_summary.steps; it adds:

- `answer_text`: compact Main-facing finding or handoff summary
- `trace_summary.narrative`: deterministic description of executed steps and output counts
- `finish_result`: concise result, completion status and stage-local Author summary
- `report_authoring`: acceptance, attempts, independent Critic verdict and presentation version

Use `answer` for the complete typed handoff. Finish prose and the trace
narrative are separate projections of the same execution.

If any Macro report fails, the **whole scene** is withheld, ensuring the
published DAG remains complete; a fallback NEVER counts as success.
This is a teacher-grounded *simulated* Sub-Agent report, not a genuine
LLM Sub-Agent's autonomous tool-use trajectory.

### Naturalize existing 529 Macro Oracle results (no stage reruns)

~~~bash
python -m envir naturalize-subtasks \
  --config envir/implement_subtask/config.yaml
~~~

First try a five-scene sample:

~~~bash
python -m envir naturalize-subtasks \
  --config envir/implement_subtask/config.yaml \
  --max-scenes 5 \
  --output-dir envir/data/maestro_subagent_reports_trial_5
~~~

The reporting configuration uses the existing
`maestro_subtask_teacher_100/subtasks.jsonl` and `scene_index.jsonl`.
It writes separate `subtasks.jsonl`, `scene_index.jsonl`,
`failed.jsonl`, and `summary.json` under
`envir/data/maestro_subagent_reports_100`.


## Structured result compatibility and review audit

In the five-scene local-Qwen pilot, 21/31 Macro reports failed. Of the
63 attempt-level reasons recorded in the failure file, 60 were the exact
guard code result_invalid_length_or_type: Qwen produced structured JSON
arrays (often correct Oracle rows), while the guard required a string.
This is primarily a response-shape mismatch, not 60 independent wrong
answers.

The Author may now return: a natural English result string; an exact
array of row objects; one exact row object; or a list of values for a
single-column output. EVERY structured result is compared to the
complete frozen Oracle answer, without changing field/value types.
Wrong subsets, extra rows, lost columns or invented values FAIL. Exact
structured answers are rendered into a readable result sentence before
independent Critic. The complete verified answer stays in the structured
answer field. Model summaries remain reviewed for factual claims.

The new PRIVATE review_audit.jsonl includes each Macro's accepted or
failed status, revision attempts, reasons, and accepted preview—even if
one failed Macro prevents the scene's complete DAG from being
published. summary.json includes rejection_causes_across_attempts.
Only complete reviewed scenes are retained in public subtasks.jsonl.

Retry the five-scene trial in a NEW output directory:

~~~bash
python -m envir naturalize-subtasks \
  --config envir/implement_subtask/config.yaml \
  --max-scenes 5 \
  --output-dir envir/data/maestro_subagent_reports_trial_5_v2
~~~


## Intermediate Macro scope guard (2026-11 second-pilot follow-up)

The second five-scene Qwen trial improved to 30/31 accepted Macro
reports, 4/5 complete scenes. The only failed Macro was a join-stage
handoff (work_mapped_entries): its *correct* Oracle answer contained
inactive and zero-valued candidate records, because later Macros apply
the eligibility checks. Critic objected to the Author's inaccurate
"all qualify" summary but then recommended deleting the candidate
rows, causing the last attempt to violate Oracle exact equality.

The Author/Critic instructions now explicitly enforce the **current
Macro stage** rather than the final overall question. A source scan
cannot claim unrelated approval/clearance conditions; a join does not
implicitly prove that all records are active or within time windows.
Contradictions against actual active/approved values and common
unexecuted filter claims are guarded deterministically before Critic.
A Critic's "filter out the rows" suggestion is never forwarded to
Author as an instruction: only the **narrative** can be repaired, not
the Oracle answer.

Use a NEW output directory for the same five-scene A/B test:

~~~bash
git pull origin main
python -m envir naturalize-subtasks \
  --config envir/implement_subtask/config.yaml \
  --max-scenes 5 \
  --output-dir envir/data/maestro_subagent_reports_trial_5_v3
~~~

The stricter rules may initially reduce accepted reports, but prevent
misleading intermediate handoffs from being published. Report counts
alone cannot establish semantic quality; inspect the private
review_audit.jsonl and accepted finish_result summaries.


### Durable live Sub-Agent report streaming

The language pass naturalize-subtasks no longer waits for all 529
Macros before saving. Its progress bar shows accepted, rejected,
and written counts for individual Macro Author/Critic reviews.
These counts are distinct from complete published scenes.

During a run, the following files are updated:

- subagent_reports_accepted.jsonl: provisional accepted Macro results,
  including Macros belonging to not-yet-complete scene DAGs
- subagent_reports_rejected.jsonl: provisional failed Macro records
- subagent_reports_audit.jsonl: per-Macro ordered private audit journal
- subagent_reports_progress.json: durable checkpoint, hashes and counts
- review_audit.jsonl: detailed audit for every reviewed Macro
- subtasks.jsonl and scene_index.jsonl: complete successful scene DAGs,
  appended as soon as their final Macro decision is committed
- failed.jsonl: failed scenes as soon as all their Macros are reviewed
- summary.json: running snapshot on each completed scene

A single writer preserves stable input order despite parallel LLM
completions. Crash recovery truncates uncommitted tails; rerunning the
SAME command and output directory reuses all committed decisions.
The provisional accepted Macro file is PRIVATE and must not be used
as a published complete-DAG bank.

The reporting.flush_every config key defaults to 1. A new streamed
full run can be started with:

~~~bash
python -m envir naturalize-subtasks \
  --config envir/implement_subtask/config.yaml \
  --flush-every 1 \
  --output-dir envir/data/maestro_subagent_reports_streamed_100
~~~

Use a NEW output folder when migrating from the old all-at-once
implementation: old outputs have no checkpoint. Completed runs do not
silently overwrite existing files; choose a new directory or explicitly
enable overwrite to restart from scratch.


## Full-run 529-Macro audit: precise all-active Guard + rejected-only retry

A full 97-scene, 529-Macro run accepted 512 Macros but published only
81 complete scenes (431 Macros) because 17 Macro reports failed.
Detailed review: 14 distinct failed Macros produced 42 attempts
rejected by the overbroad old all ... active guard, even when the
report said "all rows containing the active field" or "including inactive
entries". The rule now rejects only explicit claims that all result
rows ARE active, not source-field names or descriptions of mixed states.
The Oracle relation and evidence were never at issue.

Other failures include structured results that did not equal Oracle
and Critic disputes where sampled evidence was mistaken for the full
answer. The Critic is now reminded that exact_oracle_answer is the
authoritative full relation; sampled_evidence_atoms is non-exhaustive.

To avoid paying for all 529 LLM calls again, reuse the fully checked
accepted Macros from the PREVIOUS completed stream and retry only
the 17 rejected Macros:

~~~bash
git pull origin main
python -m envir naturalize-subtasks \
  --config envir/implement_subtask/config.yaml \
  --reuse-from envir/data/maestro_subagent_reports_streamed_100 \
  --output-dir envir/data/maestro_subagent_reports_repaired_100
~~~

The original directory is READ-ONLY. The new run checks the previous
checkpoint's SHA256 and row order, requires its full 529-Macro
completion, and validates every reused reviewed record against the
CURRENT exact Oracle Answer, provenance, executed steps, and Critic
acceptance. Only previously rejected Macros trigger new model calls.
Previously accepted but withheld (because their scene had another failed
Macro) are also reused. The new output has its OWN checkpoint and
scene-atomic publication. Summary fields are reused_accepted_macros
and freshly_reviewed_macro_tasks.

This does NOT automatically claim that all failed Macros will pass. If
any still fail, inspect the new review audit and distinguish genuine
incorrect Oracle-result proposals from semantic reviewer mistakes.
Never modify the old completed run in place; --reuse-from must point to
a DIFFERENT, previously completed reporting directory.


## Recommended one-command Stage 2

To go from accepted Stage-1 scenes directly to the realistic
Sub-Agent reports, without manually running the Oracle command:

~~~bash
python -m envir naturalize-subtasks \
  --config envir/implement_subtask/config.yaml \
  --scenes envir/data/maestro_question_fresh_100/questions/scenes.jsonl \
  --output-dir envir/data/maestro_subagent_reports_all_in_one_100
~~~

What happens internally:

1. Deterministically replay/verify each accepted Scene, exporting
   private answer/evidence ground truth under OUTPUT_DIR/oracle/.
2. Generate and independently review a stage-local finish_result for
   each Macro, streaming accepted/rejected reports with checkpoints.
3. Publish only complete reviewed scene DAGs to OUTPUT_DIR/subtasks.jsonl
   and OUTPUT_DIR/scene_index.jsonl.

OUTPUT_DIR/oracle/manifest.json binds the scene input bytes and the
oracle executor to the exported Oracle files. A crash can be resumed
with the exact SAME command and output directory, without rerunning
finished Oracle or already checkpointed report generation. Source or
executor code changes cannot reuse the old cached Oracle.

Previous commands continue to work: implement-subtasks exports only
Oracle data, and naturalize-subtasks without --scenes consumes already
materialized Oracle JSONL files. The combined mode forbids mixing
--scenes with --source-subtasks/--source-scene-index. Use a NEW output
directory when switching generation modes.
