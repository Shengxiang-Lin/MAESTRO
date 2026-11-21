# Stage 3 — trusted Main-Agent orchestration

```bash
python -m environment generate-main-turns \
  --config environment/configs/main_orchestration.yaml
```

**Goal:** Build one valid, fully reviewed **clean orchestration trajectory**
for each Stage-1 Scene whose Stage-2 Macro reports were all accepted. This
stage does **not** inject faults, simulate Sub-Agent unreliability or plan
recovery actions.

## Frozen dispatch, independently reviewed language

The deterministic compiler owns Macro DAG dependencies, the legal parallel
batches, task count and order, model/tool routing, sources and output fields.
With `generation.freeze_dispatch: true`, the Main Author can write only
public `reasoning`, `task_instruction` and `context` for those frozen
assignments. The independent Critic and strict Guards reject unsupported
claims, wrong source attribution, leaked private-generation language, invented
identifiers and task-scope changes.

The completion answer is code-frozen; only the Main's new completion
reasoning is independently reviewed. Exhausted Delegate or Completion
revisions **reject the entire Scene**. There is no reviewed training
fallback to template text. In Template mode the deterministic history is a
diagnostic clean replay, not LLM-reviewed training supervision.

## Publication and Stage-4 handoff

The Stage-3 runner first selects **complete accepted Stage-2 Scenes** in
original Stage-1 order, up to `max_scenes`. Stage-2 missing or rejected
Scenes are skipped before Stage-3 selection; Stage-3 rejected Scenes are
recorded in its own `failed.jsonl` and contribute no partial public turns.

Published artifacts under `stage3/`:

- `episodes.jsonl`: completed, approved clean O1 trajectories
- `main_turns.jsonl`: per-round Main input/output training surfaces
- `accepted_scenes.jsonl`: **exact Stage-1 rows** for approved Episodes,
  in the same canonical order; rebuilt from journals after every run/restart
- `private/teacher_alignment.jsonl`: separate private alignment and reviews
- `failed.jsonl`, `summary.json`: reasons, acceptance and sources

Every Episode includes a `clean_reference_id` derived from the frozen
World, DAG and reviewed Stage-2 Finish Results. Stylistic Main prose does
not change that identity. Stage 4 must read `accepted_scenes.jsonl` plus
`episodes.jsonl` and the Stage-2 bank, and independently verify identical
batches and returned evidence before any fault injection.

An interrupted run preserves scene/round journals and signatures. Do not
point a changed code/config/model run to an incompatible old output folder;
use a new batch directory. Inspect `accepted_scenes`,
`fully_llm_reviewed_scenes` and `rejection_attempt_stages` before
starting Stage 4.
