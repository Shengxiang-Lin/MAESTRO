# MAESTRO dataset environment

Three trusted workflows are runnable from the repository root. **Every input,
output path and LLM setting is configured in a YAML file** under
`environment/configs/`. There are no mandatory `--scenes`, `--reports` or
`--output-dir` command-line overrides.

```bash
python -m environment generate-questions --config environment/configs/questions.yaml
python -m environment naturalize-subtasks --config environment/configs/subagent_reports.yaml
python -m environment generate-main-turns --config environment/configs/base_orchestration.yaml
```

| Workflow | Python package | Config | Published file |
| --- | --- | --- | --- |
| Construct Questions | `environment/generate_question/` | `questions.yaml` | `questions/scenes.jsonl` |
| Report trusted Sub-Agents | `environment/implement_subtask/` | `subagent_reports.yaml` | `subtasks.jsonl` |
| Generate base orchestration trajectories | `environment/base_orchestration/` | `base_orchestration.yaml` | `main_turns.jsonl` |

Edit the `input` and `generation` paths in each YAML to connect your
experiment directories. YAML paths resolve relative to the **repository
root**. The default configs are wired together:
`questions.yaml` publishes to `environment/data/maestro_current_100`,
`subagent_reports.yaml` reads its `questions/scenes.jsonl`, and
`base_orchestration.yaml` reads that Scene file plus the reviewed
`subtasks.jsonl` from the Sub-Agent report directory.

## Contract and publication boundaries

**Question construction** generates exactly the requested initial candidate
pool, performs Scenario and Question Author/Critic checks, freezes a
source-scoped executable Macro DAG, and keeps every genuinely accepted
Question. No accepted-level quota refill. Summary, strict Replay/quality
report, and rejected candidate records are written next to the output.

**Sub-Agent reports** in `naturalize-subtasks` do not need an explicit
`--scenes` argument. The command first materializes deterministic Oracle
Macro evidence into `<report-output>/oracle/`, then authors and
independently reviews source-grounded Finish Results. Explicit
`--source-subtasks` and `--source-scene-index` remain available for
advanced reuse of a separately materialized Oracle bank. No real media
or Web tools are executed by dataset synthesis.

**Base orchestration** read the accepted Stage-1 Scenes and complete
reviewed Sub-Agent reports, dispatch dependency-unblocked Macro
waves (skipping evidence-free terminal extraction only when all answer
identities are already explicitly present in public reviewed results),
insert already delivered feedback into O1's cumulative
`subtask_history`, and use LLM Author/Critic for natural Main
`delegate_task` prose. Only finished accepted Episodes contribute
`main_turns.jsonl` labels. The final `complete` answer is teacher verified; its public-evidence
rationale is independently LLM-reviewed when possible, with a clearly
labeled deterministic fallback if the prose review fails.
Main-turn examples omit the repeated full O1 prompt; use:

```bash
python -m environment.base_orchestration.rebuild_prompts \
  --scenes environment/data/maestro_current_100/questions/scenes.jsonl \
  --turns environment/data/maestro_base_orchestration_100/main_turns.jsonl \
  --output environment/data/maestro_base_orchestration_100/full_prompts.jsonl
```

For all workflows, `summary.json` and `failed.jsonl` (when relevant)
hold diagnostic details. The terminal keeps concise live progress counters.

## Organization and future untrusted workflow

- `generate_question/macro_planning.py`: current Macro evidence DAG compiler
  and source access contracts (internal on-disk schema versions retained).
- `generate_question/macro_evidence_world.py`: deterministic typed evidence.
- `generate_question/prompt_templates.py`: live Author/Critic text templates.
- `implement_subtask/realize.py`, `reporting.py`, `combined.py`:
  trusted evidence replay, report review and one-command integration.
- `base_orchestration/trajectory_builder.py`, `action_author.py`,
  `o1_compat.py`: trusted Main decision supervision and real O1 formatting.
- `untrusted_orchestration/`: **reserved for future design**, not an
  implemented attack simulator. It will have a fourth independent config
  when its attack/visibility/recovery contracts are specified.

The old quota-refill engine and the obsolete private simulator/demo have
been removed. Legacy Macro Scene schema markers remain recognized to avoid
misinterpreting previously generated archived data. **Do not resume old
generation journals after this code organization change**; start new output
directories, or retain an archived Git revision for older runs.

Run tests with `python -m unittest discover -s environment/tests`.
