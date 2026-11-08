# MAESTRO `envir/`: V4.5 Task-First with V4.6 semantic-work improvements

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
4. **Macro TaskGraph:** use MMQA's nested composition and all supporting
   sources to construct a typed Work-Product DAG. Semantic work includes
   source investigation, reference/entity alignment, attribute verification,
   temporal reconstruction, aggregation, comparison/intersection, evidence
   consistency review and answer synthesis. Shared source leaves are reused;
   repeated documents are never silently dropped. Accept only 3–9
   meaningful Macro Tasks per question.
5. **Initial Question:** deterministically derive the original MMQA user
   information need from the **terminal Macro Task goal**. It cannot silently
   drift away from the graph, but graph/Question semantic equivalence still
   requires independent review.
6. **Optional V4.5 Author–Critic:** rewrite the public Question with a local
   OpenAI-compatible model. Author and Critic see sanitized semantic Macro
   goals/inputs, not private evidence, Gold answers or the actual KoPL program.
   Retain the task graph/evidence/feedback unchanged during naturalization.
7. **Source content + typed internal execution:** whenever original MMQA
   context-bank files are already present, read text/table excerpts or image
   caption proxies by exact supporting `doc_id`. For every source, produce
   distinct, question-scoped **synthetic** candidate observations (never
   invented facts about actual image pixels). Evaluate typed operations over
   these observations and write the computed result to `summary_trace` and
   the WorkProduct; this is not execution of the saved KQA KoPL program and
   is **not** the original MMQA Gold answer.
8. **Replay and source audit:** validate every original supporting document
   handle appears in the private source world and in reachable Macro Tasks;
   check typed dependencies, operator traces and deterministic results.

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
  --output-dir envir/data/macro_v46_data_1000
```

`--yago` is optional. MMQA context-bank files are loaded automatically from
`--mmqa`'s parent directory when present; override with
`--mmqa-context-dir PATH`, cap excerpts with `--mmqa-excerpt-chars 1200`, or
use `--no-mmqa-context`. No new datasets are needed. Skip YAGO when no usable local YAGO fact file is
available. A bounded prefix scan may match few MMQA entities; the result
reports actual coverage. The number of eligible multi-source compositions
depends on the loaded MMQA sample. If the requested target cannot be
reached, build **fails explicitly with a rejection breakdown** rather
than duplicating questions.

Question-only naturalization of the saved, structurally validated scenes:

```bash
python -m envir.macro.v4_cli naturalize \
  --scenes envir/data/macro_v45_data_1000/private_clean_scenes_v4.jsonl \
  --output-dir envir/data/macro_v46_data_1000_natural \
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
- `private_clean_scenes_v4.jsonl`: complete TaskGraph, synthetic source
  observations plus source-matched excerpts when present, per-Task EvidenceGraph,
  executable synthetic operator results in Feedback/Trace, KQA
  ProgramIR and original MMQA source provenance.
- `public_questions_v4.jsonl`: user Question and publicly described,
  simulated source handles; no hidden DAG, KoPL program, evidence values,
  teacher answers or feedback.
- `question_review_audit_v4.jsonl`: only when Author–Critic is enabled,
  candidate/revision/verdict metadata.
- `macro_v4_summary.json`: corpus coverage, rejected compositions, Macro
  depth/size/topology distributions, missing original support counts,
  number of original MMQA context excerpts attached, and data policies.
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

Current deterministic clean EvidenceGraph nodes **do execute the new typed
synthetic source-attribute operations**, but do **not** execute the original
KQA ProgramIR or independently solve the original MMQA multimodal questions;
the simulated observations are intentionally clearly labeled. Do not use the
current clean feedback as a factual answer oracle. Before training Main
Agents on attacks/retries/rollback, implement a richer internal executor
and semantically sufficient observations. The present task is focused on
**dataset ingestion → diverse scenario/need selection → Macro TaskGraph →
Question generation**.
