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

# Stage 3: trusted Main-Agent decisions (frozen reference for Stage 4)
python -m environment generate-main-turns \
  --config environment/configs/main_orchestration.yaml

# Stage 4: Main-observable one-fault, single-turn next-decision supervision
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

- `untrusted_orchestration/inspection/visible_claim_extraction.py`: parses explicit
  typed assertions only from the actually returned Sub-Agent
  `finish_result.result` and `summary`. Unparseable text remains unknown.
- `untrusted_orchestration/inspection/visible_claim_state.py`: constructs the
  per-round Claim–Source–Dependency state and relational evidence graph,
  including source lineage, ambiguous provenance, observed contradictions,
  partial deliveries and retry replacement.
- `private/claim_state_supervision.jsonl`: pre-action state labels for
  **accepted single-turn next decisions**, not verified global recovery.
- `private/claim_state_diagnostics.jsonl`: private states for
  reviewed failures and rejected candidates; these are NOT positive SFT
  labels.
- `main_turns.jsonl`: O1-compatible Main input/output; never contains
  private state, Gold values, injected fault labels or future report text.

The state representation is *evidential*, not a hidden truth oracle:
`unverified` and `disputed` do not mean the source is incorrect; only
explicit independent provenance may establish `corroborated`. Task
dependencies are teacher-side structural annotations, never invented
visible evidence. Private Oracle/Gold values stay outside Main prompts;
Stage 4 does not claim to have executed a future task or proved recovery.

Stage 4 preserves the exact Stage-3 history preceding the latest
completed Sub-Agent batch and applies **one atomic fault per candidate**.
It first targets Main-observable A (evidence insufficiency) or B1/B2/B3
(evidence inconsistency), then validates the realized public witness before
selecting one legal next Main `delegate_task` decision. The future batch is
not executed: `single_turn_accepted` means a reviewed next-step label, **not**
globally recovered execution. Failed, unobservable, duplicate and budget-reserve
candidates remain private. The maintained YAML/CLI does not accept the retired
multi-round recovery mode.

## Functional packages

- `generate_question/`: Stage-1 source construction, questions, operators
- `implement_subtask/`: Stage-2 executable oracle and reviewed messages
- `base_orchestration/`: Stage-3 legal trusted Main dispatch and O1 formatting
- `untrusted_orchestration/`: Stage-4 one-atom latest-feedback faults,
  Main-visible issue detection, one-step delegation, review and publication
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
