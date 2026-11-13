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

The default Stage-1 target is 1,000 Scenes (accepted count may be lower).
See `configs/README.md` for the canonical input/output paths. Each stage
has resumable journals and input/code signatures. Old experiment directories
must not be reused with these paths. `python -m environment run` executes
Stages 1, 2 and 4; Stage 4 internally compiles its clean history from
Stage-1 scenes + Stage-2 reports, not from a Stage-3 artifact.

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
fault per branch**. No asynchronous messages or multi-fault combination
are generated. Successful recovery is selected by executable replay,
not by a model asserting that it recovered.

## Functional packages

- `generate_question/`: Stage-1 source construction, questions, operators
- `implement_subtask/`: Stage-2 executable oracle and reviewed messages
- `base_orchestration/`: Stage-3 legal trusted Main dispatch and O1 formatting
- `untrusted_orchestration/`: Stage-4 fault execution, causal repair,
  evidence state supervision, naturalization, publication
- `configs/`: one stable YAML per function, shared output layout
