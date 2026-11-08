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
python -m envir build 
  --kqa data/world-source/KQA-Pro/train.json 
  --mmqa data/world-source/MultiModalQA/MMQA_train.jsonl.gz 
  --yago data/world-source/YAGO-4.6/yago-facts.ttl 
  --max-kqa-records 10000 --max-mmqa-records 30000 
  --target 1000 --seed 17 
  --output-dir envir/data/taskfirst_1000
```

YAGO is optional: omit `--yago` if the local file is not available.
Source-bank text/table/caption excerpts are read from MMQA's parent directory
when present, or specify `--mmqa-context-dir`. They are never treated as
decoded image pixels.

Generate questions without LLM calls by default. To optionally naturalize
the prebuilt question bank using a local OpenAI-compatible Author–Critic:

```bash
python -m envir naturalize 
  --scenes envir/data/taskfirst_1000/scenes.jsonl 
  --output-dir envir/data/taskfirst_1000_natural 
  --question-api-base http://127.0.0.1:8807/v1 
  --question-model local-model-name 
  --question-workers 4 --max-revisions 2
```

### Question-only semantic naturalization policy

The Author receives the **frozen Macro TaskGraph** and an explicit
`semantic_contract` derived from the selected Goal, its operation and typed
final WorkProduct, and the already authorized source inventory. This contract
is **independent of the original MMQA question** and does not expose private
source observations, answers or Sub-Agent Feedback. The Prompt allows a
natural user expression, but not any new entity, fact, source, media content,
operation, threshold or deliverable. In particular, synthetic scores cannot
be treated as real-world importance or prestige.

The independent Critic returns five explicit checks:
`objective_equivalent`, `solution_preserved`,
`required_work_preserved`, `source_faithful`, and
`answer_equivalent`. **All five must be `yes`** with acceptable
naturalness, user framing and no unsupported addition before an LLM question
is accepted. Uncertainty, protocol failures, or uncorrected violations
return the pre-existing deterministic Question without changing the DAG.

The review is an LLM-based screening decision, **not a formal proof** that
each subtask is necessary or that original media has been understood.
The semantic contract is fingerprinted and replay-checked. Previous
naturalization results remain auditable under their original review policy.

**Stage boundary:** Question naturalization does **not** rewrite formal
Macro Tasks or regenerate any `summary_trace` or `answer` field. A later
stage will take the accepted Question and frozen contracts to jointly
realize Task descriptions and build/check the associated trace and answer.

A fresh 50-question comparison after updating the code (use the ORIGINAL
`taskfirst_1000/scenes.jsonl`, not already naturalized output):

```bash
python -m envir naturalize 
  --scenes envir/data/taskfirst_recursive_1000/scenes.jsonl 
  --output-dir envir/data/taskfirst_recursive_strict50 
  --max-records 50 
  --question-api-base http://127.0.0.1:8807/v1 
  --question-model local-qwen3.5-9b 
  --question-workers 2 --max-revisions 2 --flush-every 1

python -m envir replay 
  --scenes envir/data/taskfirst_recursive_strict50/scenes.jsonl
python -m envir audit-questions 
  --scenes envir/data/taskfirst_recursive_strict50/scenes.jsonl 
  --output envir/data/taskfirst_recursive_strict50/audit.json
```

To replay and check internal evidence:

```bash
python -m envir replay 
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
