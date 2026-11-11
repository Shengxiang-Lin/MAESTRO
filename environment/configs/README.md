# MAESTRO pipeline configuration

Edit paths and model endpoints here, then run from the repository root:

```bash
python -m environment generate-questions --config environment/configs/questions.yaml
python -m environment naturalize-subtasks --config environment/configs/subagent_reports.yaml
python -m environment generate-main-turns --config environment/configs/base_orchestration.yaml
python -m environment generate-untrusted-turns --config environment/configs/untrusted_orchestration.yaml
```

- `questions.yaml`: Candidate construction, replay, and Question Author/Critic.
- `subagent_reports.yaml`: Oracle Macro evidence and trusted Sub-Agent reports; both passes run with one command.
- `base_orchestration.yaml`: Base orchestration Author/Critic, round budget, and journal outputs.
- `untrusted_orchestration.yaml`: Round-local fault mode, clean inputs, branching window, retry budget, concurrency, and **independently configurable Stage-2-style Sub-Agent report vs Stage-3-style Main Action LLM Author/Critic**. It uses the accepted Base Episodes from the third workflow.

## Migrating existing generated data

Previously generated `envir/data/` outputs are ignored by Git and are **not**
moved automatically by `git pull`. To reuse your accepted Scene/report/Base
Episode files with the new default configurations, from the repository root
run:

```bash
mkdir -p environment/data
cp -a envir/data/. environment/data/
```

Then run the fourth workflow using
`python -m environment generate-untrusted-turns --config environment/configs/untrusted_orchestration.yaml`.
The old *tracked* `environment/` implementation is deleted by the rename;
local ignored/untracked files should be backed up separately. Do not resume
old `envir` checkpoints after code-path/signature changes.

Paths in YAML are relative to the repository root, not the YAML file's own directory. Keep output directories different from inputs. Previously generated outputs with old code signatures should be treated as archival snapshots; start a new output directory for regenerated data.

## Independent LLM use in the fourth workflow

In `untrusted_orchestration.yaml`, `generation.mode` sets the default for
both report and Main action authoring. `generation.report_mode` and
`generation.main_mode` can override it independently with `llm` or
`template` (`inherit` uses the common default).

- `subagent_model`: Author/Critic for the current attempt's **result,
  summary, and trace_summary**, grounded in the actual executed WorkProduct,
  not the Gold answer.
- `model`: Author/Critic for the **Main O1 delegate_task/complete**
  natural-language decision. The dynamic task planner remains deterministic.
- `generation.report_max_revisions` and
  `generation.main_max_revisions`: separate review/retry limits.

Use a new `generation.output_dir` when changing modes or source code; the
journals deliberately reject incompatible code and model signatures.
