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

The internal Stage 1 code is organized by responsibility:

| File | Responsibility |
|---|---|
| `goal_plan.py` | Select output structure, recursively compose typed operations, normalize shared dependencies, synthesize diverse measurements and independently validate Gold |
| `evidence_plan.py` | Select feasible discovery/verification producers and source affordances; build the Macro acquisition DAG |
| `task_plan.py` | Compose a single frozen ProblemSpec, source catalog, modality observations, World binding and contract validator |
| `task_plan_runtime.py` | Materialize source views, compile Gold IR and test executable information necessity |
| `program_ir.py` | Shared typed relational executor and answer formatter |
| `question.py`, `intent.py`, `authoring/surface_quality.py` | Author/Critic prompt contract, exact output-field checking, semantic naturalization and fallback handling |
| `stages.py`, `pipeline.py` | Resumable generation and quality/audit reports |

The active code is grouped by responsibility: `grounding/` holds domain-aware
synthetic observation profiles, `authoring/` holds question/scenario
naturalization and validation helpers, and `auditing/` holds structural
decision-opportunity and difficulty diagnostics. The CLI, typed solvers and
Stage-2/3/4 data contracts remain at their established import paths.

Numeric, date and coordinate source annotations can now be acquired from
simulated image/audio evidence rather than only generic tables. A media source
is browse-only and its rows are joined to the resolved entity; no real media
decoding is claimed. Evidence mode selection is budget-based rather than
forcing Trace for every multi-source Goal.

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
