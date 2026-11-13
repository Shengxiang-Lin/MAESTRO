# MAESTRO data generation configuration

There is **one YAML per functional stage**, using stable paths rather than
experiment/version suffixes. The default target is **1,000 Stage-1 Scenes**.
Rejections are valid: this is not a guarantee that exactly 1,000 accepted
Scenes or 1,000 Stage-4 recovered branches will be published.

| Stage | Config | Output | Main input |
|---|---|---|---|
| 1 — questions and executable Macro DAG | `questions.yaml` | `environment/data/stage1/` | MMQA scenarios (context only) |
| 2 — Oracle and reviewed Sub-Agent reports | `subagent_reports.yaml` | `environment/data/stage2/` | Stage-1 `questions/scenes.jsonl` |
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
  --config environment/configs/questions.yaml

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
