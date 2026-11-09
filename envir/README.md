# MAESTRO — Task-First Semantic World Environment

**New:** [Reasoning-First V2 pipeline](v2/README.md) provides end-to-end
build-v2, replay-v2, audit-v2 and naturalize-v2 commands. The original
Task-First commands below remain available for prior runs.

`envir/` generates **synthetic**, replayable, multi-source Macro TaskGraph
problems. MultiModalQA supplies topic/source *handles*; KQA-Pro provides
unexecuted structural priors. Our independent Goal/Recipe Bank composes
new problems. No source image, video or audio is actually perceived, and
no original MMQA answer is asserted.

## Canonical pipeline

```text
MMQA Topic + Source Handles       Independent Goal + Recipe Bank
                  \                    /
                   Recursive typed Macro TaskGraph (3–9 tasks)
                                 |
                 LLM Semantic World Spec + independent Critic
                (NO old random values / original MMQA question)
                                 |
                 Deterministic World and relational fact sampling
                    | candidate observation != entity identity
                    | group category != entity identity
                    | reference-to-source != same entity
                    | ordinal precedes != historical calendar date
                                 |
                 Execute per-Task EvidenceGraph and Feedback
                                 |
                 Independent Question Author + Critic
                                 |
                     Replay / Quality / Coverage Audit
```

**Single active semantic-realization entry point:** `realize-world`.
The obsolete post-World `ground-scenes` / Scenario Contract modules and
command have been removed. Use the original Macro TaskGraph bank as input;
old synthetic values are deliberately ignored and replaced for successful
semantic specs.

All Goal, Recipe, Macro Task IDs, typed dependencies, authorized source IDs
and source modalities remain frozen during semantic realization. What
changes: the synthetic World, its source facts, relational graph and
derived EvidenceGraph/Feedback, which are completely re-executed.

## Run

From the MAESTRO repository root, using a local OpenAI-compatible vLLM
service:

### 1. Prepare the original Macro TaskGraph bank (once)

```bash
python -m envir build \
  --kqa data/world-source/KQA-Pro/train.json \
  --mmqa data/world-source/MultiModalQA/MMQA_train.jsonl.gz \
  --target 1000 --seed 17 \
  --output-dir envir/data/taskfirst_recursive_1000
```

The existing `envir/data/taskfirst_recursive_1000/scenes.jsonl` from
earlier runs is also suitable, provided that it is an original
pre-semantic TaskGraph bank (no `semantic_world_spec`,
`scene_grounding` or `question_authoring` overlay). **It need not pass
the NEW executor's strict `replay` before import**: its old
`feedback_bank` and `task_evidence_graphs` were generated using
different derived result fields. The `realize-world` importer instead
checks all original MacroPlan/source authorizations, provenance, source
fact IDs and values (recreated deterministically from the original
plan/seed) and structural evidence closure, and waives only the two
known differences in the OLD derived feedback/graph payloads. This is
an input migration allowance only, not relaxed validation for new outputs.

For accepted semantic specs, the World, relational facts, EvidenceGraph
and Feedback are regenerated. For rejected or `not_realizable` specs,
the original World and MacroPlan remain, but EvidenceGraph and Feedback
are **re-executed using the current interpreter**; therefore even
fallback rows in `scenes.jsonl` must pass NEW strict `replay`.
Run `replay` on the resulting `semantic_relational_100/scenes.jsonl`,
rather than the old TaskGraph bank.

### 2. Define a semantic World **before** generating values

```bash
python -m envir realize-world \
  --scenes envir/data/taskfirst_recursive_1000/scenes.jsonl \
  --output-dir envir/data/semantic_relational_100 \
  --max-records 100 \
  --world-api-base http://127.0.0.1:8807/v1 \
  --world-model local-qwen3.5-9b \
  --world-workers 8 \
  --max-revisions 2 --flush-every 1

python -m envir replay \
  --scenes envir/data/semantic_relational_100/scenes.jsonl
```

Model inputs: Topic; weak domain hint (which may be wrong for
cross-domain original MMQA questions); frozen Goal with Python-authored
execution rules; Macro Task operations/dependencies; source ID/modalities.
**No original MMQA question, answer, source excerpt, real media content or
old synthetic World observations are passed to this LLM.**

The LLM proposes the fictional `scenario`, stable
`semantic_bindings` (candidate kind, internal score, positive mark,
relative order, group category), `source_bindings` (one semantic source
role and display label per authorized source) and domain-specific
`user_need`. The model does **not** choose the Goal, numeric values,
source IDs, ranking direction, eligibility threshold or identity links.
An independent Critic reviews domain/topic coherence, field/source
coherence, Goal compatibility and synthetic provenance. Invalid proposals
can be revised; `not_realizable` is permitted only for genuine semantic
incompatibility. **Empty eligibility/intersection results are valid.**

Python then deterministically samples the synthetic observation IDs,
0–100 internal scores, binary marks, relative ordinals, categorical
groups, shared entity keys and directed record references using the
approved spec fingerprint and source IDs. In addition to legacy
executable fields, accepted Worlds include:

| Field | Meaning |
| --- | --- |
| `synthetic_entity_key` | Explicit synthetic identity; independent of candidate observation ID |
| `synthetic_group_key` | Shared category/cohort; **not** identity |
| `synthetic_reference_target_source_ids` | Directed, explicit reference to an authorized source |
| `synthetic_precedes_source_ids` | Explicit ordering link consistent with ordinal indices |
| `world.relations` | Derived source-cited typed edges: `same_entity`, `shared_category`, `references`, `precedes` |

These relations carry `source_id:fact_id` witnesses and are not invented
by the Question writer.

The typed Evidence executor treats:
- `composition` as explicitly directed source-reference linkage;
- `intersection` as category/group overlap **across distinct sources**;
- `comparison` and `prioritization` as DESCENDING internal score order;
- `temporal` as ASCENDING relative order, not real-world chronology;
- `feasibility` as `positive_flag=true AND score>=50`;
- `aggregation` as an exact positive-mark count.

The final GoalReport exposes **complete** ranking, chronology, category
witnesses or reference links, not only a top candidate. `replay`
deterministically regenerates the entire synthetic World (including
relations), EvidenceGraphs and Feedback, and detects forged values.

### 3. Naturalize only successfully realized scenes

```bash
python -m envir naturalize \
  --scenes envir/data/semantic_relational_100/accepted_scenes.jsonl \
  --output-dir envir/data/semantic_relational_natural_100 \
  --question-api-base http://127.0.0.1:8807/v1 \
  --question-model local-qwen3.5-9b \
  --question-workers 8 \
  --max-revisions 2 --flush-every 1

python -m envir replay \
  --scenes envir/data/semantic_relational_natural_100/scenes.jsonl

python -m envir audit-questions \
  --scenes envir/data/semantic_relational_natural_100/scenes.jsonl
```

Author and independent Critic receive the **same frozen semantic World
spec** and Python Goal execution rules. Deterministic guards reject
internal API/WorkProduct names (e.g., `PriorityReport`), machine field
names (e.g., `relative_time_index`), raw `candidate-` IDs,
unsupported perceived-image assertions and obvious Goal polarity changes
(e.g., "ascending" rather than required descending score ranking).
Naturalness and full semantic equivalence remain model judgments, not
formal proofs. Bounded retries and auditable fallback preserve valid
Question/World consistency.

## Output

`realize-world` writes:

| File | Role |
| --- | --- |
| `scenes.jsonl` | All source TaskGraphs; accepted semantic Worlds plus unmodified rejected fallbacks |
| `accepted_scenes.jsonl` | **Only** successfully semantic-first instantiated scenes; preferred training input |
| `questions.jsonl` | Corresponding public projections for all input scenes |
| `semantic_world_audit.jsonl` | Constructor/Critic proposals, failures, revisions, `not_realizable` |
| `summary.json` | Success counts, errors, model calls and completion metadata |
| `semantic_world_checkpoint.json` | Ordered, resume-safe completion index with code/input/model signature |

The `naturalize` command creates its own `scenes.jsonl`,
`questions.jsonl`, `question_audit.jsonl`, `summary.json` and checkpoint.
Use a fresh output directory for each prompt/model/code revision.

## Key modules

| Module | Role |
| --- | --- |
| `scene_catalog.py`, `scene_sources.py` | Extract MMQA Topic and authorized source inventory |
| `goal_grammar.py`, `goal_invariants.py` | Independent Goals and immutable execution rules |
| `recipes.py`, `recipe_planner.py`, `graph.py` | Recursive typed Macro TaskGraph |
| `semantic_world.py`, `semantic_world_cli.py` | Semantic spec generation, review, World reinstantiation and resumable CLI |
| `world.py`, `relational_world.py` | Deterministic numeric and typed relational synthetic facts |
| `evidence.py`, `validation.py` | Authorized per-task operation execution, source-cited EvidenceGraphs and Replay |
| `semantic_contract.py`, `realization_brief.py` | Frozen goal/question meaning and domain-scene projection |
| `question_review.py`, `question_streaming.py`, `question_audit.py` | LLM Question Author/Critic, precise guards, streaming and audit |
| `cli.py`, `builder.py` | CLI entry point and initial MacroPlan bank |

Tests:
`python -m unittest discover -s envir/tests -p "test_*.py" -v`.

## Scope and limitations

This is a **controlled synthetic orchestration environment**, not true
multimodal question answering. A source with `modality_hint=image` only
provides simulated metadata; it does not mean any image pixels were
analyzed. Generic 0–100 scores and binary flags are not real sports,
film, business, medical or historical facts. The semantic World schema
does not prove all Macro Tasks were necessary, nor does it add real
untrusted-source attacks, conflict recovery or routing behavior.

Compared with the old synthetic-bank generation, new World observations
and final EvidenceGraph answers are *different*, so do not compare them as
paired answers. Evaluate semantic diversity, real source/operation
dependence, evidence relation quality and naturalness **separately** from
Replay correctness or Critic acceptance rates.
