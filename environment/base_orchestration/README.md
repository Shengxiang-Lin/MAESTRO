# Base orchestration: trusted Main-Agent decision trajectories

Run from the repository root:

```bash
python -m environment generate-main-turns --config environment/configs/base_orchestration.yaml
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

### Evidence-grounded terminal decisions and quality control

The Teacher DAG is still the default dependency-safe schedule. A new
**visible-answer shortcut** may omit a terminal Sub-Agent delegation only
when it is an evidence-free `take/project/unique` transformation and the
preceding *public* reviewed Sub-Agent Finish Result explicitly identifies
the **entire** required `Entry-XX` answer set. Arithmetic terminal
calculation uses the existing independently verified input-value rule.
New sources, ambiguous observations and joins are never bypassed.
`summary.json` records `visible_answer_shortcuts`.

The `complete` answer remains frozen to the validated Scene. Its rationale
now quotes the already-visible public Finish Results, rather than using
one shared sentence. In LLM mode, the Author may also add a concise
natural-language introduction, which an independent Critic reviews.
If this language review cannot pass, the output explicitly uses
`teacher_grounded_completion_fallback` (a deterministic, Guard-checked
evidence quotation; **not** reported as LLM-reviewed). Mode counts appear
in `summary.json` under `completion_generation_modes`.

A second independent Critic review is possible if the first critic
rejects prose *solely* by second-guessing tools already validated by the
frozen O1 tool contract. No such objection is automatically accepted.
The Guard also rejects text that leaks Author revision or Critic feedback.

The public `main_turns.jsonl` contains `input.subtask_history` and
`output` (dict) per round, **without duplicated `output_json`**;
serialize `output` to JSON at training/export time when needed.
Revised run signatures reject incompatible old checkpoints; use a
new output directory for regenerated data.

Main-format compatibility is implemented in `o1_compat.py` by calling
the real Orchestra-o1 Prompt Builder and history formatter. Public training
labels store only `input.subtask_history` and `output`; the full Prompt is
reconstructible from the Stage-1 Scene, `round`, the run manifest, and the
version-pinned O1 renderer.

```bash
python -m environment.base_orchestration.rebuild_prompts \
  --scenes environment/data/maestro_current_100/questions/scenes.jsonl \
  --turns environment/data/maestro_base_orchestration_100/main_turns.jsonl \
  --output environment/data/maestro_base_orchestration_100/full_prompts.jsonl
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
implemented separately in `environment/untrusted_orchestration`, without
silently changing this runner's clean-success semantics.
