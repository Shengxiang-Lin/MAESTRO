# Stage 3 — trusted Main-Agent orchestration

```bash
python -m environment generate-main-turns \
  --config environment/configs/main_orchestration.yaml \
  --reports environment/data/pilots/qwen_stage234_10/stage2_fixed/subtasks.jsonl \
  --max-scenes 50 \
  --output-dir environment/data/pilots/qwen_stage234_10/stage3_current
```

Stage 3 generates clean, per-decision Orchestra-o1 Main actions from
**already-reviewed Stage-2 Sub-Agent reports**. It does not solve a new
task graph, perform actual external tool calls or modify frozen source
evidence.

## Current pipeline and file ownership

- `config.py` loads the single maintained YAML and configures the Author
  and Critic backend.
- `generate_turns.py` selects complete published Stage-2 Scenes and owns
  concurrency, atomic Scene publication, resumable round journals, output
  files and summary statistics.
- `trajectory_builder.py` executes the frozen dependency-safe reference
  waves, attaches exactly the earlier Sub-Agent public reports and writes
  clean Main decisions. It keeps the stable `build_episode` interface
  used by the Stage-2/4 paired tests and clean-reference validation.
- `action_author.py` handles current-round natural-language authoring,
  deterministic structure/source/identifier Guards, independent Critic
  review and grounded Completion.
- `task_contract.py` compiles the current Macro's public semantic
  obligations (source and lookup keys, local filters, joins, calculation
  inputs, required output fields). Unknown operators fail closed.
- `o1_compat.py` reconstructs the original
  `OmniGAIAMainAgentPrompt` and public history.
  `reasoning_fidelity.py` checks claims against previously delivered
  evidence. `rebuild_prompts.py` exports full baseline O1 prompts
  for offline training or inspection.

## Freeze scheduling; author only natural language

With `generation.freeze_dispatch: true`, the deterministic compiler
selects the Macro batch and fixes task order, dependencies, models, tools,
permitted sources and return schema. The Qwen Author may write only
`reasoning`, `task_instruction` and `context`. Neither the Author nor
Critic can add a task, change a source, invent an earlier result or replace
required output fields.

The Author receives the user question and complete **already visible**
Sub-Agent history in a compact generation-only representation, the
current-round public source catalog, the fixed task descriptions and a
task-local semantic contract. Internal Macro IDs, future findings and
private Oracle rows do not enter the Main history.

The independent Critic checks the newly authored local prose. Guard
failures yield narrowly scoped repair hints for missing source IDs,
incorrect visible identifiers, task-boundary mistakes and unsupported
facts. A Critic request to redesign the frozen schedule or override
required fields never changes the authoritative contract.

In the real O1 examples, task instructions request concrete evidence
actions and a compact Context explains the immediate handoff. MAESTRO
keeps short, evidence-driven Main `reasoning` as an explicit training
target, even though some real O1 traces have empty reasoning fields.
The language Author does not need to invent extra actions for every
round.

## Strict success: no language fallback

There is **no** Delegate or Completion fallback, and no
`allow_delegate_fallback` setting or CLI flag.

A Delegate whose Author/Critic revisions all fail rejects the Scene.
Completion must also be independently LLM-reviewed; exhausted revision
attempts reject that Scene too. The deterministic Completion template
is used only in explicit template-mode replay (without an LLM Author),
not as an accepted substitute in LLM mode.

Thus, with `generation.mode: llm`, the reported
`accepted_scenes / requested_scenes` represents complete **LLM-reviewed**
trusted trajectories. A failed Scene retains the failure details in
`failed.jsonl` and round checkpoints but publishes **no** incomplete
episode or training turns.

The final Answer is still frozen by code. Completion's natural reasoning
must cite at least one actually returned Sub-Agent Result; a short
verbatim public Result citation is added only if needed. Prose is
preserved at sentence boundaries rather than cut at an arbitrary
character position. The Critic checks the authored reasoning, not the
deterministically frozen Answer.

## Output contract and Stage 4 reuse

Each public training row contains `input.subtask_history` and the
O1-compatible `output.{action,reasoning,params}`. All accepted Episodes
are in `episodes.jsonl`; the public per-turn samples are in
`main_turns.jsonl`. Private alignment/audit goes to
`private/teacher_alignment.jsonl`. A clean episode includes a stable
`clean_reference_id` derived from the frozen question and reviewed
Stage-2 evidence rather than stylistic Main narration.

Stage 4 may compile its own deterministic trusted prefix or read
`input.clean_episodes` from Stage 3. The runner verifies all returned
evidence and action batches against an independently compiled reference
before fault simulation. The Stage-3 public output schema and
`build_episode` interface are unchanged by this cleanup.

The run manifest binds Stage-1/2 input hashes, relevant code files,
model names and run options. After a code refactor, use a **new**
output directory rather than resuming a journal created by older code.

For implementation regressions:

```bash
python -m unittest environment.tests.test_main_turn_dataset \
  environment.tests.test_stage3_natural_task_prose \
  environment.tests.test_stage3_task_contract
```
