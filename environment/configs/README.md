# MAESTRO data generation configuration

Use one maintained YAML per stage. Stage 1 defaults to the current
`task_plan + compose` profile with a 100-candidate Question Author/Critic pilot; increase
`--target` and select a new `--output-dir` for production. Retired generator YAML profiles are regression fixtures under
`environment/tests/fixtures/` and `environment/macro_v3/tests/fixtures/`,
not supported production configurations. Only the current Task Plan profile
is kept in the production Stage-1 config directory. Acceptance is verified by Replay and necessity tests.

| Stage | Config | Output | Main input |
|---|---|---|---|
| 1 — questions and executable Macro DAG | `questions_task_plan.yaml` (default) | `environment/data/stage1/` | YAGO-seeded synthetic Task Plan + composed World |
| 2 — Oracle and reviewed Sub-Agent reports | `subagent_reports.yaml` | `environment/data/stage2/` | Stage-1 `questions/scenes.jsonl` (reviewed) |
| 3 — trusted Main decisions | `main_orchestration.yaml` | `environment/data/stage3/` | Stage-1 scenes + Stage-2 `subtasks.jsonl` |
| 4 — untrusted recovery decisions | `untrusted_orchestration.yaml` | `environment/data/stage4/` | Stage-1 scenes + Stage-2 `subtasks.jsonl` |

Stage-2 reviewed reports are stored at
`environment/data/stage2/subtasks.jsonl`; its private deterministic Oracle
cache lives under `environment/data/stage2/oracle/`.
This is the **same layout** whether Stage 2 runs via
`naturalize-subtasks` (combined) or via separate
`implement-subtasks` and report naturalization.

Run each stage explicitly:

```bash
python -m environment generate-questions \
  --config environment/configs/questions_task_plan.yaml

python -m environment naturalize-subtasks \
  --config environment/configs/subagent_reports.yaml

python -m environment generate-main-turns \
  --config environment/configs/main_orchestration.yaml

python -m environment generate-untrusted-turns \
  --config environment/configs/untrusted_orchestration.yaml
```

`python -m environment run` is a convenience path for Stage 1 → Stage 2 →
Stage 4. Stage 4 **does not consume the Stage-3 generated Main dataset**; it
constructs its trusted execution prefix deterministically from Stages 1/2.

Stages retain their own manifest signatures and resumable journals.
**Do not reuse** directories containing previous 100-scene or differently
configured data. Archive/move those outputs before running these defaults.
Use `--output-dir` for exploratory pilots to avoid overwriting canonical
datasets.

The Stage-4 config fixes **one atomic injected fault per branch** and uses
synchronous Sub-Agent batches. Asynchronous responses and multi-fault
combinations are out of scope.
