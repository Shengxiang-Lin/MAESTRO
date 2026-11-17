# MAESTRO pipeline configurations

Each stage has one maintained config. Stage 1 uses an output-first Goal
planner and an executable, provenance-aware Macro DAG. Do not select
retired generator variants through flags or configuration keys.

| Stage | Config | Default output | Input |
|---|---|---|---|
| 1 — questions and executable evidence DAG | `questions.yaml` | `environment/data/stage1/` | YAGO motif vocabulary; simulated evidence |
| 2 — reviewed Sub-Agent reports | `subagent_reports.yaml` | `environment/data/pilots/qwen_stage234_10/stage2/` | First 10 accepted Stage 1 Scenes |
| 3 — trusted Main dispatch | `main_orchestration.yaml` | `environment/data/pilots/qwen_stage234_10/stage3/` | Same 10 Scenes and Stage 2 reports |
| 4 — untrusted recovery | `untrusted_orchestration.yaml` | `environment/data/pilots/qwen_stage234_10/stage4/` | Stage 1 / Stage 2 and reviewed Stage 3 clean episodes |

```bash
python -m environment generate-questions --config environment/configs/questions.yaml
python -m environment naturalize-subtasks --config environment/configs/subagent_reports.yaml
python -m environment generate-main-turns --config environment/configs/main_orchestration.yaml
python -m environment generate-untrusted-turns --config environment/configs/untrusted_orchestration.yaml
```

**Current 10-Scene Qwen pilot:** The three maintained Stage 2/3/4 YAML
files temporarily select the first 10 rows from the existing Stage-1
`environment/data/stage1/questions/scenes.jsonl`. Do **not** run Stage 1 again
for this pilot. Each stage writes beneath
`environment/data/pilots/qwen_stage234_10/`. Stage 4 checks
`input.clean_episodes` against Stage 3's actual clean references and samples
only one returned-feedback fork per Scene, with up to 12 atom candidates per
round. API Author/Critic uses `qwen3.8-flash`.

The Stage 1 construction and Question review phases are independently
journaled. Its latest output is a single schema, not a selectable
v3/v4/v5/v6 series. Only accepted Scenes enter later stages; the
private Oracle and Gold results never enter public Main prompts.

Stage 2 always rebuilds its deterministic Oracle from current Scenes
(no Oracle cache signatures or manifest checks). Reviewed LLM reports keep
separate resume safeguards.

Stage 2 reviewed reports are stored in `environment/data/stage2/subtasks.jsonl`;
its private deterministic Oracle cache is under `stage2/oracle/`.
During this pilot, Stage 4 checks and reuses the reviewed Stage 3
`episodes.jsonl` against the frozen Stage 1/2 evidence bank.

Stages 1–4 use DashScope `qwen3.8-flash` for active LLM Author/Critic
calls. Export `DASHSCOPE_API_KEY` before running; the endpoint is
`https://dashscope.aliyuncs.com/compatible-mode/v1`. Stage 1 resolves it
through `config/model_config.yaml`, while Stages 2–4 set the same endpoint
explicitly. In Stage 3/4, `generation.sub_model` is the Sub-Agent model
label recorded in the simulated Main history, not another API invocation.

Keep output directories distinct between pilot runs, seeds and modified
configuration. `python -m environment run` remains the Stage 1→2→4
convenience command.
