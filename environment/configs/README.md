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
up to three stratified feedback forks per Scene (early/middle/late),
with up to 12 executable one-atom candidates per fork. Recovery is bounded
to eight post-fork decision rounds; genuine non-recovered branches may be
reviewed for evidence-state supervision but never become positive SFT. API Author/Critic uses the locally served `local-qwen3.5-9b`.

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
