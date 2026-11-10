# MAESTRO Environment

Current supported workflow: **generate validated questions with frozen Macro
plans, then produce grounded Sub-Agent reports**. Every World and media
observation is synthetic. A successful Replay verifies internal evidence
consistency, not independent multimodal perception or Agent performance.

## Generate questions (Stage 1)

Configure `envir/config.yaml`, then run:

```bash
python -m envir generate-questions \
  --config envir/config.yaml \
  --target 100 \
  --output-dir envir/data/maestro_current_100
```

Active settings under `generation`:

| Setting | Default | Meaning |
| --- | --- | --- |
| `macro_first_v2_share` | 100 | Use the current Macro-first World for Medium/Hard; Easy keeps the approval baseline |
| `difficulty_mix` | `[17, 33, 50]` | Easy/Medium/Hard desired **accepted** counts for a 100-item dataset |
| `quota_refill` | `true` | Refill missing difficulty groups after both Scenario and Question review |
| `max_candidate_multiplier` | `3` | Hard upper limit: attempted candidates ≤ target × multiplier |
| `seed` | 17 | Reproducible seed for dataset sampling and frozen World construction |

You may set `--max-candidate-multiplier 4` to raise the budget without
changing the requested difficulty ratio. A quota is never filled by relabeling
a problem, duplicating an answer, or inventing an accepted Critic decision.
If the budget is exhausted, `status=partial` and
`summary.json.difficulty_quotas.deficit` reports the shortfall.

Completed batches are saved under `rounds/01`, `rounds/02`, etc.
with independent Scenario/Question checkpoints. Runs interrupted between
batches can reuse already verified batches if their signatures match.
Use a **new output directory** when changing generation settings.

### Canonical Stage 1 outputs

```text
maestro_current_100/
├── summary.json                   # achieved quotas and accepted counts
├── quality_report.json            # strict Replay and difficulty proxies
├── constructed/
│   ├── scenes.jsonl                # selected frozen scenes
│   └── failed.jsonl                # Scenario failures across rounds
├── questions/
│   ├── scenes.jsonl                # accepted final Questions + frozen evidence
│   ├── questions.jsonl             # Main-visible Question records
│   ├── difficulty_audit.jsonl
│   └── failed.jsonl                # Question failures across rounds
└── rounds/
    ├── 01/                         # full intermediate stage output
    └── 02/                         # as many bounded rounds as necessary
```

The `constructed/scenes.jsonl` root file represents selected scenes after
quota selection; intermediate attempted/fallback records remain in each
round's folder. The accepted `questions/scenes.jsonl` is the authoritative
input for Stage 2. Its records include typed Goal, scoped Macro DAG,
deterministic Gold Solvers, evidence World, provenance and final Question.

## Build Sub-Agent reports (Stage 2)

```bash
python -m envir naturalize-subtasks \
  --config envir/implement_subtask/config.yaml \
  --scenes envir/data/maestro_current_100/questions/scenes.jsonl \
  --output-dir envir/data/maestro_current_reports_100
```

This creates exact Oracle WorkProducts and checked natural-language
Sub-Agent finishes. Stage 2 never changes the underlying Gold Answer.
The deterministic `oracle_review.py` maps Entry IDs to cited FactRefs
and their observed Active status, including full answer provenance when
evidence atom samples are truncated. A provably mistaken Critic assessment
may trigger one targeted independent re-review; it **never** auto-passes.
A wrong citation in an Author summary is rejected.

Reports are published under `complete_scene_only`: one rejected Macro
prevents publishing an incomplete scene.

## Contracts, validation and complexity

The current generation pipeline is organized around:

- `macro_first_v2.py`, `macro_first_v2_world.py`: frozen Macro DAG,
  typed Goal, synthetic, modality-sensitive observations and witness generation.
- `semantic_evidence.py`, `question.py`: executable predicate and
  **single-answer-slot** contracts. Question text may paraphrase, not add
  unlicensed outputs, time windows or media requirements.
- `source_title_contract.py`, `scenario_fields.py`, `scenario.py`:
  immutable source modality/title contract; audited, field-level repair
  of overlong/missing narrative fields; independent semantic Critic.
- `information_necessity.py`: check that independent multimodal branches
  each retain ambiguity and the intersection uniquely identifies a target
  with a distinct possible final answer.
- `difficulty_proxy.py`: observe Macro depth, evidence volume, primitive
  operation count, index hops and candidate ambiguity. Structural proxies
  **do not** establish true Hard-vs-Medium Agent failure rates.
- `quota_pipeline.py`, `stages.py`, `pipeline.py`: bounded,
  checkpointed generation rounds and final accepted-quota publication.
- `implement_subtask/oracle_review.py`, `implement_subtask/reporting.py`:
  deterministic Oracle evidence checks and independently reviewed prose.

The active CLI/config no longer exposes superseded
`clue_chain_share`, `evidence_graph_share` or `macro_first_share`
generation experiments. Frozen historical scene readers and underlying
operators remain because they are still necessary to audit prior
experiments. They are not part of the current user-facing generation API.

Run a read-only audit of an accepted or historical scene bank:

```bash
python -m envir quality \
  --config envir/config.yaml \
  --scenes envir/data/maestro_current_100/questions/scenes.jsonl \
  --output envir/data/maestro_current_100/quality_recheck.json
```

## Limits

The current Media World uses synthetic structured observations, not
original decoded audio, images, video or external web browsing. Macro
information necessity is checked for the frozen candidate World, not
against every imaginable independent plan. Scene/Question Author-Critic
acceptance and 100% Oracle replay do **not** measure real Main-Agent
solving difficulty, cost, accuracy or user-perceived naturalness.
