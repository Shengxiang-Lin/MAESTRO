# MAESTRO environment

The maintained environment has **four stages and four canonical YAML files**.
This branch configures one coherent batch of **500 Stage-1 candidate
ProblemSpecs** under `environment/data/batches/candidate_500/`.
Older 2/10-scene experimental configuration files are not production entries.

## Production commands

Start the local OpenAI-compatible `local-qwen3.5-9b` server on
`http://127.0.0.1:8807/v1`, and ensure
`data/world-source/YAGO-4.6` is present. Then run from the repository root:

```bash
python -m environment generate-questions
python -m environment quality
python -m environment naturalize-subtasks
python -m environment generate-main-turns
python -m environment generate-untrusted-turns
```

These commands consume `environment/configs/{questions,subagent_reports,
main_orchestration,untrusted_orchestration}.yaml`, respectively. Stage 1's
`quality` is a **read-only audit**, not an additional generation stage.

### Batch data dependencies

| Stage | Input | Main output |
| --- | --- | --- |
| 1 | YAGO vocabulary + seeded ProblemSpec composer | `stage1/questions/scenes.jsonl` |
| 2 | Stage-1 **accepted** Scenes | `stage2/subtasks.jsonl`; private `stage2/oracle/` |
| 3 | Stage-1 accepted Scenes + reviewed Stage-2 reports | `stage3/episodes.jsonl`, `stage3/main_turns.jsonl`, `stage3/accepted_scenes.jsonl` |
| 4 | Stage-3 accepted Scenes + Stage-2 reports + Stage-3 reviewed Clean References | `stage4/main_turns.jsonl`, private audits |

A Stage-1 `generation.target: 500` defines **500 frozen candidate
ProblemSpecs**, not a promise of 500 Author/Critic-accepted questions.
Stage-2 report review may reject additional Scenes, and Stage-3 review may
reject entire trajectories. Each stage publishes only valid output. Stage 3
rebuilds `accepted_scenes.jsonl` from its complete, accepted scene journals
so Stage 4 never silently treats an unreviewed Stage-3 episode as a reference.

Stage 4 freezes historical Stage-3 dispatches and prior returned reports,
applies **one** atomic fault to the current feedback batch, checks only
**Main-visible** issues, and publishes **one** reviewed next `delegate_task`
decision. It is not a multi-round recovery rollout. Stage 4's
`target_main_turns: 2000` is a **global positive decision budget**, not
the number of input Scenes or a guaranteed yield; `max_candidates_per_round`
and `max_distractors_per_fault` deliberately bound coverage.

### Resume, validation and non-fallback policy

The four output directories and per-stage journals are independent. Do not
reuse a previous pilot output directory with different source files, YAML or
code signatures. Failures and insufficient evidence remain explicit; never
promote template or fallback reports to positive LLM reviewed training data.
Inspect each stage's `summary.json` before starting the next command:

- Stage 1: `generated`, `rejected`, `final_scenes`
- Stage 2: `oracle_accepted_scenes`, accepted reviewed reports, failures
- Stage 3: `accepted_scenes`, `rejected_scenes`, `fully_llm_reviewed_scenes`
- Stage 4: `trusted_prefixes_rejected`, `available_scenarios`,
  `published_scenario_types`, `main_training_turns`

`Stage 4 status: done` means candidate processing ended, **not** that
every candidate passed language review or that recovery succeeded globally.

## Packages

- `generate_question/` — Stage 1 task plans, sources, evidence, auditing
- `implement_subtask/` — Stage 2 Oracle and independently reviewed reports
- `base_orchestration/` — Stage 3 frozen DAG dispatch, public O1 actions
- `untrusted_orchestration/` — Stage 4 one-fault scenario-first decisions
- `data_export/` — generic dataset export, not a train/test splitting policy

For exact caps, paths, and safe rollout advice see
[`configs/README.md`](configs/README.md). `python -m environment.data_export`
produces generic `cases.jsonl`, `decisions.jsonl` and private labels; each
baseline is responsible for scene-disjoint splits and its own training adapter.
