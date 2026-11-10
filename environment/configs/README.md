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
- `untrusted_orchestration.yaml`: Round-local fault mode, clean inputs, branching window, retry budget, concurrency, naturalization mode, and journal outputs. It uses the accepted Base Episodes from the third workflow.

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
