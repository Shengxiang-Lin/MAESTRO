# Stage 1 — output-first task generation

Stage 1 has one maintained generator. The input is a bounded YAGO motif
sample that supplies predicate vocabulary and structural statistics; the
generated task's World and all media observations are **synthetic**, never
reused OmniGAIA question/answer/media content.

## Workflow

```bash
python -m environment generate-questions \
  --config environment/configs/questions.yaml \
  --output-dir environment/data/stage1
```

This command builds a frozen **Goal → evidence acquisition → Macro DAG →
canonical World → Gold Program → Question** pipeline. Question Author and
Critic may paraphrase the request, but may not edit the factual World, Goal
answer schema, evidence sources or Gold.

The active Stage-1 code is grouped by function:

| Package | Responsibility |
|---|---|
| `planning/` | Canonical output-first `goal_plan`, `evidence_plan`, `task_plan` |
| `grounding/scene_semantics.py` | One cohesive home for domain clue profiles, goal referents, readable simulated-media observations and source titles |
| `execution/` | `task_plan_runtime` and shared `program_ir` Gold executor |
| `authoring/` | Question/scenario prompts, field guards, semantic and surface checks |
| `auditing/` | Decision-opportunity, grounded observation and structural-quality audits |
| `stages.py`, `pipeline.py` | Resumable generation, Question Author/Critic and reports |

Root-level `goal_plan.py`, `evidence_plan.py`, `task_plan.py`,
`task_plan_runtime.py`, and `program_ir.py` are compatibility facades
for historical Stage-2/3/4 imports; the implementations live only in the
corresponding packages. Older experimental `hybrid_world*` and
`task_graph_first*` modules remain for historical snapshot Replay and
their independently maintained regression suites. Do not use them for new
Stage-1 question generation.

The active code is grouped by responsibility: `grounding/` holds domain-aware
synthetic observation profiles, `authoring/` holds question/scenario
naturalization and validation helpers, and `auditing/` holds structural
decision-opportunity and difficulty diagnostics. Thin compatibility shims
preserve older Stage-2/3/4 import paths for shared surface contracts. The CLI, typed solvers and
Stage-2/3/4 data contracts remain at their established import paths.

Numeric, date and coordinate values can be acquired from simulated
image/audio observations rather than only generic tables. Browse-only media
Macros are now independent of target identification; the final synthesis
joins their records to the resolved entity. Keyed sources still require an
upstream key. Each source observation also contains a modality-specific
`observed_content` rendering and a typed `observation_semantics` marker,
strictly derived from the original executable values. No real binary media
decoding is claimed.

Candidate reference reconciliation and independently necessary verification
remain **two distinct Main-visible Macros**, while the pair of directory
lookups is a single coherent Sub-Agent investigation. This allows event and
verification routes with typed Goals inside the 11-Macro budget.

The quality report adds `macro_decision_opportunities` for parallel batches,
key-dependent handoffs, reusable intermediate results and true evidence
corroboration (distinct from a Goal comparison operator). These are *structural
opportunities*, not proof that an LLM made adaptive decisions.

The output structure is sampled **before** operations. Individual semantic
fields require compatible producers: Lookup; Arithmetic; Temporal
Difference/Midpoint; Geographic Distance; Count/Aggregate; Compare/Select;
bounded Verify/Explain. Independent fields are composable. A common
prerequisite is fetched once, and two Goal graphs may be compared by a
typed operation subgraph partial order. The current bounded grammar
requires 2–3 hops for the entity key and at most 11 Macro tasks; this is
not an unconstrained planner.

## Quality gates

Candidate Scenes must pass the frozen goal/World hash, source schemas,
typed execution, unique target answer, nonempty alternative candidate
behavior, Gold Replay, source-level counterfactuals and information
necessity. All accepted Scenes must pass the Question Guard and LLM
Author/Critic, if configured. Non-accepted Question attempts stay in
`failed.jsonl`; acceptance and fallback counts are always separate.

Multi-field Goals share a compatible numeric, temporal or geographic
measurement family. Entity identity and observation counts can be reused as
common prerequisites, but unrelated measurement families are not randomly
bundled into one user request.

Numerical facts (counts, ratios, dates, distances) are **seed-conditioned**
and vary across questions; the algorithm preserves the same operation and
source-dependency contract. Quality reports record `answer_value_diversity`
per semantic field: number of samples, unique values and modal answer share.
A deterministic warning flags high-frequency numeric shortcut candidates.
A genuinely binary `higher_source` or `conclusion` field is excluded from
the numeric-collapse warning.

Expected outputs include `constructed/scenes.jsonl` and, with LLM
naturalization, `questions/scenes.jsonl`. Stage 2–4 consume only
accepted frozen Scenes and their Macro/WorkProduct contracts. Never point a
new generation run at an existing output directory unless intentionally
overwriting it.

## Validation

```bash
python -m unittest environment.tests.test_stage1_generator
python -m environment quality --config environment/configs/questions.yaml \
  --scenes environment/data/stage1/questions/scenes.jsonl
```

Stage 2, trusted orchestration and untrusted orchestration maintain their
existing public data contracts. Version strings in persisted data are
**schema identifiers**, not separately selectable generator implementations.

## 100-task pilot, reproducibility and limits

Offline, network-free CI and local smoke benchmark:

```bash
python -m environment.tests.stage1_pilot \
  --target 100 --output-dir environment/data/stage1_offline_100
```

This command uses a deliberately tiny **test RDF motif**; outputs are
`constructed/scenes.jsonl`, `pilot_summary.json` and
`quality_report.json`. The GitHub workflow
`.github/workflows/stage1-pilot-100.yml` also uploads those files as an
artifact. This validates 100 frozen Question/World/Gold Scenes and strict
Replay but **does not** run Qwen Author/Critic or establish benchmark
realism. You can use `--knowledge-yago /path/to/real/YAGO-4.6` to test
with your local graph (still offline deterministic Question fallback).

For the full Author/Critic experiment use the existing production command
at the start of this README, with the YAGO corpus and Qwen endpoint
configured in `environment/configs/questions.yaml`. Use a NEW output
directory to avoid mixing incompatible frozen Worlds.

Review the new `grounded_evidence` and `macro_decision_opportunities`
fields under `knowledge_world_grounding.single_task_plan`. In particular:
`with_independent_media_acquisition` is structural source-first work,
not conditional feedback dispatch; `with_evidence_corroboration`
means an independently sourced verification gate, not an arithmetic
`verify` Goal operator. The static trusted Stage-1 DAG does not yet
encode branching policy decisions based on the *contents* of future
Sub-Agent messages; those require Stage-3 trajectory integration.
