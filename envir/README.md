# MAESTRO — Task-First Environment

`envir/` is the **only** data-driven Task-First training-environment
generator. It builds independent multi-domain problems from the existing
**MultiModalQA, KQA-Pro and optional YAGO-4.6** corpora. No original
MultiModalQA question is a generated target question or WorkProduct goal.

## Architecture

```text
MultiModalQA source annotations ───► Scene Bank
YAGO optional exact matches ──────► scene context
Independent curated Goal Bank ────┐
                                  ├─► seeded Scene × Goal selection
Typed Work-Product Recipe Bank ───┘             │
                                                ▼
                              recursive backwards WorkProduct planner
                                ├─ choose alternative output recipes
                                ├─ recursively expand required typed inputs
                                ├─ memoize shared intermediate products
                                ├─ attach ALL original MMQA source handles
                                └─ validate 3–9 task DAG, source closure
                                                ▼
                              terminal Macro Task question blueprint
                                                ▼
                           NEW deterministic question (3 styles/goal)
                                                │
                                  optional LLM Author → independent Critic
                                                ▼
                         simulated multi-source EvidenceGraph + Task Feedback
                                                ▼
                             replay, proof closure and diversity audit
```

**Macro Tasks are substantial work packages, not individual KoPL operations.**
Alternative recipes for `EvidenceBundle`, `AlignedCandidates`,
`EvidenceReview`, `GoalAssessment` and `GoalReport` make the DAG
structurally variable. Each recipe declares its input/output WorkProduct
types, operation and deliverable. When several downstream tasks require one
product, the compiler reuses it. The source documents do NOT determine
Macro topology; the MMQA nested type is a source-side prior only.

The **Scene Bank** contains MMQA entities/topic, broad domain hints,
modalities, source IDs and optional YAGO matches. The **Goal Bank** contains
eight independent problem-solving needs, including comparison, counting,
reference linkage, reconciliation, chronology, prioritization and
feasibility. A compatibility check rejects unusable source combinations.

## Canonical Python modules

| Module | Responsibility |
| --- | --- |
| `scene_catalog.py`, `scene_sources.py` | Extract and group scene/source handles |
| `source_contexts.py`, `priors.py` | Optional MMQA excerpts and KQA ProgramIR |
| `goal_grammar.py`, `recipes.py` | Reusable Goal and typed Recipe Banks |
| `recipe_planner.py` | Backward recursive WorkProduct planning |
| `graph.py` | Typed edges, coarsening, waves and exact DAG topology signature |
| `question.py`, `question_review.py` | Graph-first new question and Author–Critic |
| `world.py`, `evidence.py`, `validation.py` | Synthetic observations, trace execution, replay |
| `builder.py`, `cli.py`, `question_streaming.py`, `question_audit.py` | Batch pipeline, CLI, checkpointing and diagnostics |

The canonical command-line entry is **`python -m envir`**.

## Build (from the MAESTRO repo root)

```bash
python -m envir build \
  --kqa data/world-source/KQA-Pro/train.json \
  --mmqa data/world-source/MultiModalQA/MMQA_train.jsonl.gz \
  --yago data/world-source/YAGO-4.6/yago-facts.ttl \
  --max-kqa-records 10000 --max-mmqa-records 30000 \
  --target 1000 --seed 17 \
  --output-dir envir/data/taskfirst_1000
```

YAGO is optional: omit `--yago` if the local file is not available.
Source-bank text/table/caption excerpts are read from MMQA's parent directory
when present, or specify `--mmqa-context-dir`. They are never treated as
decoded image pixels.

Generate questions without LLM calls by default. To optionally naturalize
the prebuilt question bank using a local OpenAI-compatible Author–Critic:

```bash
python -m envir naturalize \
  --scenes envir/data/taskfirst_1000/scenes.jsonl \
  --output-dir envir/data/taskfirst_1000_natural \
  --question-api-base http://127.0.0.1:8807/v1 \
  --question-model local-model-name \
  --question-workers 4 --max-revisions 2
```

To replay and check internal evidence:

```bash
python -m envir replay \
  --scenes envir/data/taskfirst_1000/scenes.jsonl

python -m unittest discover -s envir/tests -p 'test_*.py' -v
```

## Output files

- `scene_bank.jsonl`: extracted scene/source catalog **without MMQA
  question text**.
- `goal_bank.json`: curated, scene-independent Goal and question patterns.
- `recipe_bank.json`: all typed WorkProduct producers and their alternatives.
- `source_manifest.jsonl`: selected scene/goal, original MMQA source
  provenance and generated question; original dataset question is
  teacher-side audit metadata only.
- `scenes.jsonl`: private DAG, KQA ProgramIR, source worlds, WorkProducts,
  EvidenceGraphs and Sub-Agent Feedback/summary_trace.
- `questions.jsonl`: model-facing questions and source descriptions.
- `summary.json`: source coverage, work types, recipe diversity,
  **exact unlabeled DAG isomorphism count**, depths, failure/rejection reasons,
  question provenance and review status.
- `question_audit.jsonl` and `question_checkpoint.json`: only for
  optional LLM question review/streamed resumption.

The structured source facts and internally computed values are synthetic
task observations. They provide deterministic typed operation semantics,
but they do **not** prove the original MMQA answer or perform actual
image/audio/video perception. KQA ProgramIR is an unexecuted
inside-Sub-Agent reasoning prior; KoPL nodes are not Main-Agent Tasks.
Clean replay proves internal consistency and complete source reference
closure—not real-world factual correctness.

## Diversity and safety invariants

The full generator checks: no copied MMQA questions; no duplicate generated
question text; complete coverage of original supporting source IDs;
typed WorkProduct input/output closure; no orphan Macro tasks; source facts
never leak to the public question file; reproducible recipe selection;
and exactly 3–9 Macro Tasks per accepted scene.

The `unlabeled_dag_topologies` statistic is a **true DAG isomorphism
measure**, invariant to node ID and Work Type. It should be considered
alongside `unique_recipe_sequences`, actual Goal distribution, graph
depth, breadth and question-style variation. More tasks/edges do not
automatically mean better orchestration problems.
