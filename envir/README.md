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

### Semantic Scenario Contract (v4, default post-graph grounding)

**Purpose:** give the already-compiled Macro TaskGraph, synthetic World and
EvidenceGraph ONE coherent and reusable domain interpretation. It is not
creative background decoration, original MultiModalQA answer reconstruction
or new source perception. The frozen Goal/operation/source inventory remains
authoritative, so the same contract can later inform the Question, Task,
Summary Trace and Answer surfaces.

`python -m envir ground-scenes` now uses **v4 Semantic Scenario Contract**.
The Constructor receives:
- Topic/domain, immutable Goal operation/required fields/threshold/answer
  scope, and task-derived reference Question **as a semantic boundary**;
- exact synthetic candidate/flag/score/order/group values and
  `source_id:fact_id` references;
- compact compiled TaskGraph operations, dependencies and computed results;
- a leave-one-source-out diagnostic (not proof of task necessity).

**The Constructor does NOT see the original MMQA Question, or original
MMQA source excerpt text.** This prevents the previous Pichichi Trophy /
The Shape of Water failures, where original MMQA question details were
mistaken for evidence in a separately sampled Goal. The original source
files, source inventory and frozen evidence remain unchanged.

The LLM produces this **proposal**, never new source facts:

```json
{
  "status": "realizable",
  "reason": "none",
  "scenario": {
    "setting": "In a hypothetical Subject X sports archive...",
    "user_role": "Sports archive curator",
    "motivation": "Report the number of internally marked submissions"
  },
  "semantic_bindings": {
    "candidate_kind": "fictional review submissions",
    "score": "internal evaluation points",
    "positive_flag": "recorded positive review marker",
    "relative_time_index": "relative submission order",
    "group_key": "fictional review cohort"
  },
  "source_bindings": [
    {
      "source_id": "EXACT_EXISTING_SOURCE_ID",
      "candidate_label": "Submission A",
      "source_role": "hypothetical image-source metadata record"
    }
  ],
  "user_need": "Count positively marked submissions for Subject X"
}
```

For `status="not_realizable"`, the LLM must provide a specific reason
instead of inventing missing facts or operations. Such a scene is retained
in its original form, with an auditable `not_realizable` status.

After syntactic checks and **independent six-axis semantic review**, Python
builds the accepted `scene_grounding` overlay:
- `scenario`: setting, user role and why the requested output matters;
- `semantic_bindings`: stable candidate/score/flag/order/group meanings;
- `source_cards`: model-supplied *fictional labels and source roles* plus
  **Python-sourced exact fact values and references** (the model cannot
  supply or overwrite values);
- `question_intent`: **derived by Python**, immutable Goal ID, operation,
  required fields, hard constraints, answer scope, WorkProduct type and
  authorized sources;
- `review`, fingerprints and source-ablation diagnostic.

The reviewer screens domain plausibility, semantic coherence, source
faithfulness, Goal compatibility, evidence boundary and user-need alignment.
Replay compares full materialized overlays with original World, Goal,
TaskGraph, EvidenceGraph and Feedback signatures. Legacy v1-v3 overlays
remain replay-compatible. A reviewer PASS remains **a model judgment, not
a proof** of realistic semantics or multi-source necessity. The real media
was not viewed.

Run a small pilot with a **fresh output directory**:

```bash
python -m envir ground-scenes \
  --scenes envir/data/taskfirst_recursive_1000/scenes.jsonl \
  --output-dir envir/data/taskfirst_semantic_v4_20 \
  --max-records 20 \
  --grounding-api-base http://127.0.0.1:8807/v1 \
  --grounding-model local-qwen3.5-9b \
  --grounding-workers 4 \
  --max-revisions 2 --flush-every 1

python -m envir replay \
  --scenes envir/data/taskfirst_semantic_v4_20/scenes.jsonl

python -m envir naturalize \
  --scenes envir/data/taskfirst_semantic_v4_20/scenes.jsonl \
  --output-dir envir/data/taskfirst_semantic_v4_natural20 \
  --question-api-base http://127.0.0.1:8807/v1 \
  --question-model local-qwen3.5-9b \
  --question-workers 2 --max-revisions 2 --flush-every 1
```

The `summary.json` includes `accepted`, `rejected`,
`not_realizable`, revision counts, exact validation error counts and
the source-ablation diagnostic. Inspect both
`grounding_audit.jsonl` (failed attempts) and accepted
`scene_grounding` (actual semantic contract) before scaling up.

#### Legacy v1–v3 notes (backward compatibility only)

### Post-TaskGraph World concretization (optional LLM stage)

After `build` and **before** `naturalize`, run `ground-scenes`.
This stage is a source-bound **synthetic interpretation** of the existing
World, not original MMQA gold annotation, real visual understanding, or
a reconstruction of the actual historical event.

The Scene Constructor receives the deterministic reference Question,
the original MMQA Question as a weak inspiration cue, the existing source
facts (all source IDs, candidate IDs, simulated scores, boolean flags,
relative time indices and grouping keys), and compact **per-Task EvidenceGraph
operations, input dependencies and computed operation results**. The LLM
proposes a concrete hypothetical `scene_context`, a domain-appropriate
`candidate_kind`, field meanings and an exactly source-matched list of
fictional display labels and source roles. It is not asked to invent or
recompute facts.

Python independently materializes `scene_grounding` as a fact-bound
overlay on `scenes.jsonl`:

```json
{
  "scene_grounding": {
    "mode": "synthetic_interpretation_not_original_media_annotation",
    "scene_context": "A hypothetical editor compares ...",
    "candidate_kind": "fictional evaluation entries",
    "field_meanings": {
      "score": "evaluation points",
      "positive_flag": "positive review indicator",
      "relative_time_index": "entry sequence",
      "group_key": "evaluation group"
    },
    "source_cards": [
      {
        "source_id": "EXISTING_SOURCE_ID",
        "candidate_key": "EXISTING_CANDIDATE_ID",
        "candidate_label": "Entry A",
        "source_role": "fictional table reference record",
        "fact_bindings": {
          "score": {
            "value": 42,
            "fact_ref": "EXISTING_SOURCE_ID:F6",
            "verification": "consistent_synthetic_record"
          }
        }
      }
    ],
    "frozen_solution_fingerprint": "..."
  }
}
```

The `fact_bindings` object above is **abridged**: the actual record also
contains entry ID, positive mark, relative sequence, group and their exact
original fact references and values. No fake values or extra sources are
invented. `grounding_audit.jsonl` stores the model's original proposal,
Critic verdict, rejection reasons and revision attempts.

**What counts as acceptable:** Every source appears once; every referenced
fact points to its exact original `source_id:fact_id`, and its value equals
the original World; all TaskGraph, original World, Feedback, EvidenceGraph
and complete replay stay unchanged; field meanings and context must be
hypothetical but plausible; there can be no new operation, source, media
content claim or historical assertion. The independent Critic must return
five `yes` judgments (domain plausibility, frozen facts, source roles,
solution compatibility and reference-Question alignment). Failing proposals
are rejected and the original scene is preserved with an audit record.
This is a screening decision, **not a proof** of real-world correctness or
multi-source necessity. Each overlay also includes a leave-one-source-out
goal-level influence diagnostic (source IDs that change / do not change the
Goal-pattern result when omitted). This helps identify mostly cosmetic
multi-source questions, but does NOT prove any Task is redundant or
repair source-linkage computation. The current simulator's operation
semantics may still be abstract; the next Task/Trace/Answer stage must
address those independently.

Pilot command using the **original, non-naturalized** 1000-scene bank:

```bash
python -m envir ground-scenes \
  --scenes envir/data/taskfirst_recursive_1000/scenes.jsonl \
  --output-dir envir/data/taskfirst_recursive_ground50 \
  --max-records 50 \
  --grounding-api-base http://127.0.0.1:8807/v1 \
  --grounding-model local-qwen3.5-9b \
  --grounding-workers 2 \
  --max-revisions 2 \
  --flush-every 1

python -m envir replay \
  --scenes envir/data/taskfirst_recursive_ground50/scenes.jsonl

python -m envir naturalize \
  --scenes envir/data/taskfirst_recursive_ground50/scenes.jsonl \
  --output-dir envir/data/taskfirst_recursive_ground_natural50 \
  --max-records 50 \
  --question-api-base http://127.0.0.1:8807/v1 \
  --question-model local-qwen3.5-9b \
  --question-workers 2 \
  --max-revisions 2 \
  --flush-every 1

python -m envir replay \
  --scenes envir/data/taskfirst_recursive_ground_natural50/scenes.jsonl
```

`ground-scenes` writes ordered, resumable `scenes.jsonl`,
`questions.jsonl`, `grounding_audit.jsonl`, `summary.json`, and
`grounding_checkpoint.json`. Use a fresh output directory for a new
model/prompt version, or `--overwrite` only when discarding a previous run.
No re-execution or modification of the original Macro TaskGraph is needed.

#### Grounding source-role ordering fix (v3)

The 100-scene v2 pilot reached 95 accepted. Three of the remaining
five failures were unrelated to semantic content: the LLM returned a valid
`source_roles` array in a different order from the canonical order used
by Python when writing `source_cards`. The prior hash treated list order
as substantive and falsely rejected the World overlay with both
`grounding_proposal_changed` and `grounding_fact_bindings_changed`.

In v3, source roles are canonically sorted by immutable `source_id`
*only for the proposal fingerprint*. Every source must still occur exactly
once, values still bind to the original World, and old v1/v2 accepted
overlays retain their original replay hash convention. A malformed JSON
constructor response receives a bounded retry.

Importantly, 95% acceptance is not proof of substantive grounding:
some accepted scenarios still attach specific historical events or visual
details not supported by existing simulated facts. Manually inspect the
actual `scene_context`, `user_need` and `source_roles` before using
the new examples for supervised training. A separate stronger semantic
grounding audit is needed; v3 addresses the deterministic false rejects.

#### Grounding validation notes (v2)

The first 100-scene pilot produced 11 accepted proposals and 89 rejections,
mainly because the old validator restricted all `field_meanings` and
`source_role` descriptions to 110 characters, and `user_need` to 220
characters. These limits did not represent semantic correctness. Version 2
uses 320/360-character description limits (candidate labels 140), and
accepts longer Critic explanations. Source IDs, source coverage, fact values,
World/TaskGraph/EvidenceGraph/Feedback fingerprints and Replay constraints
remain strict.

A v2-specific safety check now rejects interpretations claiming that visual
features were actually confirmed or detected in unobserved original media.
An internal synthetic flag can mean a *fictional* review marker, but cannot
certify content in original image/video pixels. Accepted v1 overlays retain
their earlier validation policy for backwards Replay compatibility; a fresh
v2 run should use a **new output directory**, not reuse its old checkpoint.

`summary.json` now reports `validation_error_counts` and explicit
`rejection_reasons`; each audit row also includes
`last_validation_errors`. These make prompt failures and genuine
source/provenance violations separately diagnosable. Improved acceptance
alone is not evidence of realistic scene quality: inspect accepted
`scene_context`, `field_meanings`, `source_roles` and the
independent Critic assessments before scaling.

### Question-only natural language realization (v2)

Question Author and Critic have deliberately **different views**:

- **Author:** `realization_brief.py` produces a short, user-facing description
  of the topic, actual information need and complete result. It does NOT
  contain the frozen deterministic Question template, raw schema field
  names, Task IDs, verbose Macro Task text, private EvidenceGraph or feedback.
  The Author may choose completely different natural wording, including
  'the records' instead of listing every text/table/image source.
- **Critic:** receives the complete frozen `semantic_contract` and semantic
  Macro TaskGraph, then checks objective, solution, necessary work, source
  boundary and answer completeness. It must **not** demand wording overlap,
  source-type enumeration or explicit workflow steps. A quoted claimed
  violation must actually occur in the candidate Question. Contradictory
  'pass' judgments trigger repair instead of being mistaken for a successful
  semantic review.
- **Audit:** reports reference-question similarity, near-template wording
  and leftover machine-like schema jargon. It does NOT flag omission of
  internal setting labels such as 'MultiModalQA reference collection' as a
  naturalization defect.

Strictly immutable: source scope, Goal operation, conditions, final result
and the already compiled TaskGraph. No new accounts, source subtypes, media
contents, dates, real-world significance or additional facts are allowed.
Ordinary conversational framing **without new evidence claims** is permitted.

The new review policy is `taskfirst_natural_question_v2`; previously
naturalized Question files remain replayable under their original policy.
Review still involves model judgment and is **not** a formal proof.
Question naturalization never changes Tasks, `summary_trace` or `answer`.
Those will be realized together in a separate future stage.

Run a **new** 50-question comparison against the ORIGINAL deterministic
scene bank (not the previous naturalized 50):

```bash
python -m envir naturalize \
  --scenes envir/data/taskfirst_recursive_1000/scenes.jsonl \
  --output-dir envir/data/taskfirst_recursive_flexible50 \
  --max-records 50 \
  --question-api-base http://127.0.0.1:8807/v1 \
  --question-model local-qwen3.5-9b \
  --question-workers 2 --max-revisions 2 --flush-every 1

python -m envir replay \
  --scenes envir/data/taskfirst_recursive_flexible50/scenes.jsonl

python -m envir audit-questions \
  --scenes envir/data/taskfirst_recursive_flexible50/scenes.jsonl \
  --max-examples 50 \
  --output envir/data/taskfirst_recursive_flexible50/audit.json
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
