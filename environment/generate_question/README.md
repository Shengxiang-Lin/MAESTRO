# Stage 1 — Current Macro Task Plan

**Current supported production route:** a single Macro Task Plan is sampled, a
canonical synthetic World is composed, scoped evidence source views are generated,
and the existing Compiler executes Gold/Replay and necessity audits.

Run:

```bash
python -m environment generate-questions \
  --config environment/configs/questions_task_plan.yaml \
  --output-dir environment/data/stage1_current_20
```

The currently maintained Stage-1 data generation profile is
`knowledge_variant: task_plan` and `knowledge_world_mode: compose`.
The default pilot contains 20 tasks; set `--target` explicitly for larger runs.
YAGO is used as a vocabulary/structural seed, not as an assertion that
synthetic entities or media were real.

## What is generated?

| Artifact | Contents | Initial Main visibility |
|---|---|---|
| Macro Task Plan | roles, dependency edges, source scope, terminal goal | **No** (private teacher plan) |
| Canonical World | entities, required contextual association, project-event links | **No** |
| Evidence Views | candidate observations, keyed/browse access modes, FactRefs | Only the **source catalog**; contents acquired later |
| User Question | concise final goal and essential event/verification constraints | **Yes** |
| Initial Task Inputs | target-specific simulated observations (image/audio/video/text/table) | **Yes**, as text descriptions |
| Gold Program / Trace | executable source-scoped operators and provenance | **No** |
| Gold Answer | final teacher answer | **No** |

**Question** describes the user's goal, not the internal subtask sequence.
**Initial Task Inputs** describe observations *already provided* with the
request. They are not images/audio files and do not imply successful live
perception. **Available Evidence Sources** are still-unqueried synthetic
archives/indices. Each can be browsed or retrieved with an upstream key
as declared in the source catalog.

In the Orchestra-o1 Main prompt, these are two separate blocks:

```text
==== QUESTION ====
[user-facing question]

==== INITIAL TASK INPUTS (PROVIDED SIMULATED OBSERVATIONS) ====
[image_1 / observed image] The photographed folio ...
[audio_1 / observed audio] The recording includes ...

==== AVAILABLE EVIDENCE SOURCES (NOT INITIAL ATTACHMENTS) ====
- M1_a [image, browse]: ...
- M1_b [table, browse]: ...
...
```

The renderer does **not** expose evidence source handles as attached physical
media. No source rows, hidden entity IDs, or Gold outputs enter the initial
Main prompt. A Sub-Agent may investigate the source catalog, but should not
re-extract observations already supplied by the user.

## Core implementation

| File | Responsibility |
|---|---|
| `task_plan.py` | sample one Macro graph, freeze semantic goal/World binding and initial reference observations |
| `world_motifs.py` | canonical entity, contextual and event relationships |
| `task_plan_runtime.py` | derive scoped evidence views, Gold operator chains and counterfactual checks |
| `question.py` | render one concise task-level Question; ensure required constraints |
| `stages.py` | deterministic construction, user-task dedup and optional author/reviewer |
| `compiler.py` / `program_ir.py` | scoped execution and provenance |
| `quality.py` / `pipeline.py` | strict replay and quality report |
| `../base_orchestration/o1_compat.py` | reconstruct actual Main prompt and visibility for Stage 3/4 |

No second Planner, per-stage Shadow World or extra runtime retrieval graph is
introduced. The existing Task Plan and World remain the only truth sources.

## Important quality checks

- **Gold Replay** reproduces World, Program, Trace, answer and Macro graph.
- **Necessary discovery clues**: a full set singles out one dossier; removing
  a clue leaves ambiguity rather than simply making an API call fail.
- **World relation necessity**: the required shared association affects the
  choice of target rather than being decorative context.
- **Keyed Event dependency**: the prior Identification macro supplies
  `project_id` to the Event Ledger. Wrong but valid keys
  produce nonempty incorrect downstream answers.
- **Public surface**: identifying observations stay in the separate
  `task_input`, with no answer/project/case keys.
- **Deduplication** uses normalized (Question, initial observation content);
  random IDs and observation ordering do not create new tasks.
- **Prompt separation**: first Main turn knows which observations were already
  supplied, which sources have not been queried, and has no fictitious
  physical file attachment.

For a small offline verification:

```bash
python -m unittest environment.macro_v3.tests.test_single_task_plan -v
python -m environment generate-questions \
  --config environment/configs/questions_task_plan.yaml \
  --output-dir environment/data/stage1_audit_current
```

Check `constructed/scenes.jsonl`, `constructed/summary.json`
and `quality_report.json`. The 20-case seed should yield all
ordinals 0–19, 20/20 replay, 20/20 information necessity and 20 public
reference cards. These are **synthetic task feasibility checks**, not a proof
of Main-Agent independent solving or recovery.

## Stage 2–4

Stage 2 generates and reviews source-scoped Sub-Agent reports from the
frozen Macro graph. Stage 3 reconstructs the real O1-style Main prompt using
the public question, already-supplied observations and available-source
descriptions. Stage 4 branches histories under controlled untrusted
feedback; it must use only reports delivered by that round, not hidden
World state.

Older experimental generator modules and tests should not be used as the
**current Stage-1 entrypoint**. Their retained shared compiler and regression
interfaces are maintained separately until their downstream dependencies are
migrated; they are not needed to author current `task_plan + compose`
scenes.


## Current-only cleanup and downstream pilot

The default generator remains \`task_plan + compose\`. Obsolete question
renderers shadowed by the frozen semantic goal and the old scenario-domain
fallback have been removed from the production Task Plan authoring path.
Historical **other** generator families keep their exact Replay/compiler
implementations for regression tests; they are not the current default.

For newly generated homogeneous Task Plan compose scene banks,
\`quality_report.json\` keeps general replay, graph, acquisition and
reasoning diagnostics plus \`knowledge_world_grounding.single_task_plan\`.
It omits empty, retired sub-generator report sections. Quality reports for
old scene banks retain their historical schema to avoid reinterpreting
old results as current experiments.

To inspect orchestration with the same frozen 5 scenes after Stage 1:

\`\`\`bash
python -m environment naturalize-subtasks \
  --config environment/configs/subagent_reports.yaml \
  --scenes environment/data/stage1_current_20/questions/scenes.jsonl \
  --output-dir environment/data/stage2_pilot_5 \
  --max-scenes 5

python -m environment generate-main-turns \
  --config environment/configs/main_orchestration.yaml \
  --scenes environment/data/stage1_current_20/questions/scenes.jsonl \
  --reports environment/data/stage2_pilot_5/subtasks.jsonl \
  --output-dir environment/data/stage3_llm_5 \
  --max-scenes 5 --mode llm --workers 4
\`\`\`

Stage 3 authors prose for **frozen** legally dispatchable task batches; it
does not independently discover new orchestration choices.
