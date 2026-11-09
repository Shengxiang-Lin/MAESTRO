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
