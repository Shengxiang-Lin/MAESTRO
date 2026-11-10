# Trusted Main-Agent decision trajectories

Run from the repository root:

```bash
python -m envir generate-main-turns --config envir/configs/main_trajectories.yaml
```

All Scene/report inputs, output directories, model endpoints, token/attempt
constraints and worker settings are in the YAML. Optional CLI flags can
override paths, max-scenes, mode, worker count and progress.

The input is a **frozen accepted Scene** (Question + public source catalog
+ teacher Macro DAG) and its complete reviewed **Sub-Agent report bank**.
`trajectory_builder.py` schedules source-scoped Macro tasks in dependency
waves. A Main delegation may dispatch several independent tasks at once.
`action_author.py` lets an LLM author realistic O1-style reasoning,
instructions and upstream context; the deterministic Guard fixes model/tools
and the Critic reviews prose fidelity. Upon acceptance, Stage-2 reports are
replayed as successful Sub-Agent feedback, but are **not executed** as real
tools. The next Main decision sees the full cumulative *already returned*
history, including all parallel subtasks.

Main-format compatibility is implemented in `o1_compat.py` by calling
the real Orchestra-o1 Prompt Builder and history formatter. Public training
labels store only `input.subtask_history` and `output`; the full Prompt is
reconstructible from the Stage-1 Scene, `round`, the run manifest, and the
version-pinned O1 renderer.

```bash
python -m envir.main_trajectories.rebuild_prompts \
  --scenes envir/data/maestro_current_100/questions/scenes.jsonl \
  --turns envir/data/maestro_main_trajectories_100/main_turns.jsonl \
  --output envir/data/maestro_main_trajectories_100/full_prompts.jsonl
```

A Scene may need several **sequential Main decisions**; separate Scenes run
concurrently (`generation.workers` in YAML). Complete Episodes are
published to `episodes.jsonl` and their per-decision labels to
`main_turns.jsonl`. Private DAG/Oracle alignment lives only in
`private/teacher_alignment.jsonl`, rejected Scenes in `failed.jsonl`.
Run `summary.json` includes accepted/rejected Scene counts, committed
Main decisions, and detailed failure statistics. Resumable per-round/Scene
journals remain enabled and checked against source/code signatures.

This workflow intentionally contains **only trusted successful
trajectories**. Future adversarial/uncertain feedback generation will be
implemented separately in `envir/untrusted_orchestration`, without
silently changing this runner's clean-success semantics.
