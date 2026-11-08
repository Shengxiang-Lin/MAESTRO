# MAESTRO `envir/`: V4.5 Dataset-driven Task-First

This directory builds **multi-domain, multi-source orchestration questions** from
the three existing corpora only: **MultiModalQA**, **KQA-Pro**, and optionally
**YAGO-4.6**. It does not import GAIA/OmniGAIA questions or additional datasets.

## What is generated, and by whom?

1. **Scenario/need catalog (MultiModalQA):** read official question, original
   question type (e.g. `Compare(Compose(TableQ,ImageQ),TextQ)`), source roles,
   `supporting_context` IDs, entity hints, and topic. Do not use the annotated
   factual answer as the model-facing Question or as a fake perception result.
   Questions with duplicate wording are deduplicated.
2. **Optional YAGO enrichment:** bounded scan of the local YAGO TTL file;
   attach only *unique exact normalized entity-label* matches, with RDF
   subject IDs and observed facts. An unmatched entity stays unmatched.
   The YAGO context is a teacher-side annotation, **not** a magically
   available MMQA document and not proof of entity equivalence.
3. **Internal KQA prior:** load full KoPL ProgramIR, preserving program shape.
   It is a seed for future *inside-Sub-Agent* operations and trace builders.
   KoPL nodes are **not** delegated as Macro Tasks, and the saved program is
   not claimed to have executed in the current clean environment.
4. **Macro TaskGraph:** use MMQA's nested composition and supporting-source
   requirements to construct source-scoped evidence investigation work
   packages, typed cross-source work-product handoffs and a terminal
   information-need resolution work product. Create only legitimate
   3–9-task DAGs; reject unsupported or single-source cases instead of
   padding them with meaningless tasks.
5. **Initial Question:** deterministically derive the original MMQA user
   information need from the **terminal Macro Task goal**. It cannot silently
   drift away from the graph, but graph/Question semantic equivalence still
   requires independent review.
6. **Optional V4.5 Author–Critic:** rewrite the public Question with a local
   OpenAI-compatible model. Author and Critic see sanitized semantic Macro
   goals/inputs, not private evidence, Gold answers or the actual KoPL program.
   Retain the task graph/evidence/feedback unchanged during naturalization.
7. **Synthetic clean evidence and replay:** existing V4 evidence executor
   creates *text surrogate* source observations, internal evidence nodes,
   deterministic work products and trace entries. This checks structural
   consistency/source-reference closure, **not** external factual correctness,
   real image/video/audio perception or a computed MultiModalQA Gold answer.

This is an orchestration-question generator. It does **not** claim to solve
the original MMQA question or perform an independent raw-media inference.

## Run from MAESTRO repository root

Prepare 1,000 candidates using full MMQA training data and the local KQA file:

```bash
python -m envir.macro.v4_cli build \
  --kqa data/world-source/KQA-Pro/train.json \
  --mmqa data/world-source/MultiModalQA/MMQA_train.jsonl.gz \
  --yago data/world-source/YAGO-4.6/yago-facts.ttl \
  --max-kqa-records 10000 \
  --max-mmqa-records 30000 \
  --max-yago-triples 500000 \
  --target 1000 \
  --seed 17 \
  --output-dir envir/data/macro_v45_data_1000
```

`--yago` is optional. Skip it when no usable local YAGO fact file is
available. A bounded prefix scan may match few MMQA entities; the result
reports actual coverage. The number of eligible multi-source compositions
depends on the loaded MMQA sample. If the requested target cannot be
reached, build **fails explicitly with a rejection breakdown** rather
than duplicating questions.

Question-only naturalization of the saved, structurally validated scenes:

```bash
python -m envir.macro.v4_cli naturalize \
  --scenes envir/data/macro_v45_data_1000/private_clean_scenes_v4.jsonl \
  --output-dir envir/data/macro_v45_data_1000_natural \
  --question-api-base http://127.0.0.1:8807/v1 \
  --question-model local-qwen3.5-9b \
  --question-workers 4 --max-revisions 2 --flush-every 1
```

Or supply the Author–Critic options directly to `build`.
To replay the clean evidence and graph/question contracts:

```bash
python -m envir.macro.v4_cli replay \
  --scenes envir/data/macro_v45_data_1000/private_clean_scenes_v4.jsonl

python -m unittest discover -s envir/tests -p 'test_*.py' -v
```

## Output contracts

- `scenario_seed_manifest.jsonl`: only the **selected** MMQA record IDs,
  original question, source IDs/modalities, composition type and optional
  YAGO provenance. It is teacher-side metadata.
- `private_clean_scenes_v4.jsonl`: complete TaskGraph, actual source
  *surrogates*, per-Task EvidenceGraph, clean work-product feedback, KQA
  ProgramIR and original MMQA source provenance.
- `public_questions_v4.jsonl`: user Question and publicly described,
  simulated source handles; no hidden DAG, KoPL program, evidence values,
  teacher answers or feedback.
- `question_review_audit_v4.jsonl`: only when Author–Critic is enabled,
  candidate/revision/verdict metadata.
- `macro_v4_summary.json`: corpus coverage, rejected compositions,
  MacroGraph statistics and data policies.
- `question_progress_state_v4.json`: when naturalizing with the incremental
  checkpoint writer, for safe deterministic resume.

## Why this differs from the previous pilot

Previous V4.5 used a closed set of 9 domains, 36 predefined incident
descriptions and 8 handcrafted business intents; MMQA was only a weak
recipe-selection weight. The new build path is **MMQA need first, Macro DAG
second, initial Question third**. It preserves the original Task-First
principle: no LLM invents the TaskGraph topology, and all Macro edges are
typed work-product dependencies. The original fixed-scene generator is
not used by the default CLI path.

An MMQA composition is a *prior for macro work*, not a guarantee that each
dataset-level micro QA operation deserves one Sub-Agent. Source investigation
is one bounded, independently meaningful work package. KoPL retrieval,
filtering, and arithmetic belong to a later verified **internal** execution
layer.

## Validation scope and future boundary

Current deterministic clean EvidenceGraph nodes do **not** execute the
original KQA ProgramIR or the original MMQA multimodal QA computation;
the simulated observations are intentionally clearly labeled. Do not use the
current clean feedback as a factual answer oracle. Before training Main
Agents on attacks/retries/rollback, implement a richer internal executor
and semantically sufficient observations. The present task is focused on
**dataset ingestion → diverse scenario/need selection → Macro TaskGraph →
Question generation**.
