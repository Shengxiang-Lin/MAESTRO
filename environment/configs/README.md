# MAESTRO pipeline configurations

Each stage has one maintained config. Stage 1 uses an output-first Goal
planner and an executable, provenance-aware Macro DAG. Do not select
retired generator variants through flags or configuration keys.

| Stage | Config | Default output | Input |
|---|---|---|---|
| 1 — questions and executable evidence DAG | `questions.yaml` | `environment/data/stage1/` | YAGO motif vocabulary; simulated evidence |
| 2 — reviewed Sub-Agent reports | `subagent_reports.yaml` | `environment/data/stage2/` | Stage 1 accepted `questions/scenes.jsonl` |
| 3 — trusted Main dispatch | `main_orchestration.yaml` | `environment/data/stage3/` | Stage 1 Scenes and Stage 2 reports |
| 4 — untrusted recovery | `untrusted_orchestration.yaml` | `environment/data/stage4/` | Stage 1 Scenes and Stage 2 reports |

```bash
python -m environment generate-questions --config environment/configs/questions.yaml
python -m environment naturalize-subtasks --config environment/configs/subagent_reports.yaml
python -m environment generate-main-turns --config environment/configs/main_orchestration.yaml
python -m environment generate-untrusted-turns --config environment/configs/untrusted_orchestration.yaml
```

The Stage 1 construction and Question review phases are independently
journaled. Its latest output is a single schema, not a selectable
v3/v4/v5/v6 series. Only accepted Scenes enter later stages; the
private Oracle and Gold results never enter public Main prompts.

Stage 2 reviewed reports are stored in `environment/data/stage2/subtasks.jsonl`;
its private deterministic Oracle cache is under `stage2/oracle/`.
Stage 4 reconstructs its trusted execution prefix from Stages 1/2,
rather than consuming Stage 3 generated Main data.

Stages 1–4 use DashScope `qwen3.8-flash` for active LLM Author/Critic
calls. Export `DASHSCOPE_API_KEY` before running; the endpoint is
`https://dashscope.aliyuncs.com/compatible-mode/v1`. Stage 1 resolves it
through `config/model_config.yaml`, while Stages 2–4 set the same endpoint
explicitly. In Stage 3/4, `generation.sub_model` is the Sub-Agent model
label recorded in the simulated Main history, not another API invocation.

Keep output directories distinct between pilot runs, seeds and modified
configuration. `python -m environment run` remains the Stage 1→2→4
convenience command.
