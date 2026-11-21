# MAESTRO pipeline configurations

Each stage has one maintained config. Stage 1 uses an output-first Goal
planner and an executable, provenance-aware Macro DAG. Do not select
retired generator variants through flags or configuration keys.

| Stage | Config | Default output | Input |
|---|---|---|---|
| 1 — questions and executable evidence DAG | `questions.yaml` | `environment/data/stage1/` | YAGO motif vocabulary; simulated evidence |
| 2 — reviewed Sub-Agent reports | `subagent_reports.yaml` | `environment/data/pilots/qwen_stage234_10/stage2/` | First 10 accepted Stage 1 Scenes |
| 3 — trusted Main dispatch | `main_orchestration.yaml` | `environment/data/pilots/qwen_stage234_10/stage3/` | Same 10 Scenes and Stage 2 reports |
| 4 — untrusted single-turn decisions | `untrusted_orchestration.yaml` | `environment/data/pilots/qwen_stage234_10/stage4/` | Stage 1 / Stage 2 and reviewed Stage 3 clean episodes |

```bash
python -m environment generate-questions --config environment/configs/questions.yaml
python -m environment naturalize-subtasks --config environment/configs/subagent_reports.yaml
python -m environment generate-main-turns --config environment/configs/main_orchestration.yaml
python -m environment generate-untrusted-turns --config environment/configs/untrusted_orchestration.yaml
```

**Current 10-Scene Qwen pilot:** Stage 2 and Stage 3 reuse the first
10 accepted Scenes and reviewed Sub-Agent reports. Stage 4 reads the checked
Stage-3 clean reference and enumerates one-atom variants on the latest
returned feedback wave. Stage 4 uses scenario-first A/B1/B2/B3
sampling by design; no legacy decision-mode configuration exists.
`target_main_turns` is a decision-label budget, not a number of complete
recovery episodes. Use new output directories after source-code changes.

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

Stages 1–4 call the already-running local vLLM `local-qwen3.5-9b` over
`http://127.0.0.1:8807/v1` for active LLM Author/Critic calls.
No API key is required. Stage 1 deliberately bypasses its API-only model
registry resolver; Stages 2–4 configure the endpoint explicitly. The vLLM
server is launched with thinking disabled by default. Stage 1's optional
Scenario Author/Critic remain disabled (`null`) as before.
In Stage 3/4, `generation.sub_model` is the Sub-Agent model label recorded
in the simulated Main history, not another API invocation.

Keep output directories distinct between pilot runs, seeds and modified
configuration. The maintained entrypoints are the four stage-specific
commands above; the old Stage 1→2→4 shortcut was removed because it skipped
the Stage 3 Clean Reference now required by the paired pilot.
