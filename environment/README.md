# MAESTRO environment

This package builds executable, reviewable data for trusted and untrusted
multi-agent orchestration. Configuration and data directories are organized
**by function**, not by experimental revision.

## Canonical pipeline

```bash
# Stage 1: questions, synthetic sources, typed Macro dependency graphs
python -m environment generate-questions \
  --config environment/configs/questions.yaml

# Stage 2: execute the Macro Oracle, then review Sub-Agent feedback
python -m environment naturalize-subtasks \
  --config environment/configs/subagent_reports.yaml

# Stage 3: trusted Main-Agent decisions (optional for Stage 4)
python -m environment generate-main-turns \
  --config environment/configs/main_orchestration.yaml

# Stage 4: single-fault evidence recovery, reviewed Main decisions
python -m environment generate-untrusted-turns \
  --config environment/configs/untrusted_orchestration.yaml
```

Stage 1 uses the maintained output-first Task Plan builder and loads real
YAGO relations via `generate_question/grounding/yago.py`. Obsolete
`macro_v3/` synthesizers and the unused `scenarios/` and `sources/`
packages have been removed. Stage 1 Question Author/Critic and Stage 2–4
language review use DashScope `qwen3.8-flash`.

The current Stage 2–4 pilot consumes the first 10 accepted Scenes from
`environment/data/stage1/questions/scenes.jsonl`, without regenerating
Stage 1; see `configs/README.md`. Stage 2 deterministically recompiles
its Oracle every run and keeps separately resumable LLM report journals.
Stage 4 validates and reuses the Stage 3 Clean Reference when
`input.clean_episodes` is configured.

## Stage 4 supervision

- `untrusted_orchestration/visible_claim_extraction.py`: parses explicit
  typed assertions only from the actually returned Sub-Agent
  `finish_result.result` and `summary`. Unparseable text remains unknown.
- `untrusted_orchestration/visible_claim_state.py`: constructs the
  per-round Claim–Source–Dependency state and relational evidence graph,
  including source lineage, ambiguous provenance, observed contradictions,
  partial deliveries and retry replacement.
- `private/claim_state_supervision.jsonl`: per-round state labels for
  **globally recovered and fully reviewed** branches.
- `private/claim_state_diagnostics.jsonl`: additional private states for
  reviewed failures and unresolved branches; these are NOT positive SFT
  labels.
- `main_turns.jsonl`: O1-compatible Main input/output; never contains
  private state, Gold values, injected fault labels or future report text.

The state representation is *evidential*, not a hidden truth oracle:
`unverified` and `disputed` do not mean the source is incorrect; only
explicit independent provenance may establish `corroborated`. Task
dependencies are teacher-side structural annotations, never invented
visible evidence. Oracle/Gold is used only to validate the execution
outcome and remains private.

Stage 4 uses synchronous completed Sub-Agent batches and **one atomic
fault per branch**. The pilot samples early/middle/late feedback forks,
performs paired no-repair/recovery replays, and language-reviews a bounded
number of genuine failure branches for observable state supervision.
Only globally recovered branches can become positive Main SFT labels. No asynchronous messages or multi-fault combination
are generated. Successful recovery is selected by executable replay,
not by a model asserting that it recovered.

## Functional packages

- `generate_question/`: Stage-1 source construction, questions, operators
- `implement_subtask/`: Stage-2 executable oracle and reviewed messages
- `base_orchestration/`: Stage-3 legal trusted Main dispatch and O1 formatting
- `untrusted_orchestration/`: Stage-4 fault execution, causal repair,
  evidence state supervision, naturalization, publication
- `configs/`: one stable YAML per function, shared output layout


## Unified base dataset (separate from training adapters)

`python -m environment.data_export` publishes unsplit generic
`cases.jsonl`, `decisions.jsonl`, and
`private/decision_labels.jsonl` into `environment/data/unified/`.
It does **not** produce DA-SFT/DA-GRPO Parquets or select
train/val/test Scenes. See `environment/data_export/README.md`.

Each baseline invokes its own `prepare_data.py`, chooses a frozen Scene
split, filters eligible reference actions and writes its own
`baseline/<method>/data/` training files.
