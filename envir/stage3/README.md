# Stage 3 — Per-Round Main-Agent Trajectory Dataset (primary workflow)

**Training unit = one Main-Agent decision round, not one Macro Task.**
Stage 1 supplies Question and a private executable Macro DAG; Stage 2
supplies independently reviewed source-grounded Sub-Agent finishes. Stage 3
uses the private DAG only to choose legal parallel batches and author the
next O1-style **Main Prompt → Main Action** supervised sample. After a
delegation, all of that batch's Stage-2 `finish_result` and public Trace
Summary become visible in the **following** Main prompt.

The prompt is rendered by the actual
`OmniGAIAMainAgentPrompt.build_prompt` and
`BaseMainAgent._format_subtask_history` used in
`baseline/orchestra-o1`. Its fields include user Question, simulated
workspace modality inputs, tool/model catalog, Attempt budget and the
history of *prior* Sub-Agent tasks, results, summaries and trace summaries.
The O1 `prior_context` argument is intentionally not used: the current
O1 prompt template does not render it. The Main output remains the O1
`delegate_task` JSON (possibly multiple independent tasks in one round)
or `complete` JSON. No `work_M*`, complete Gold DAG or Oracle is exposed
in the training sample.

## From-scratch experiment: three Web evidence acquisition routes

Use **new directories** to avoid manifest/signature mismatch with old
Stage-1 and Stage-3 checkpoints. The default YAML configs already point
to the local OpenAI-compatible model server on port 8807.

```bash
git pull origin main

# Stage 1: fresh 100 accepted scenes, including the new v2 web contracts
python -m envir generate-questions \
  --config envir/config.yaml \
  --target 100 \
  --output-dir envir/data/maestro_web_v2_100

# Stage 1 Replay verification (do not proceed if invalid)
python -m envir quality \
  --config envir/config.yaml \
  --scenes envir/data/maestro_web_v2_100/questions/scenes.jsonl \
  --output envir/data/maestro_web_v2_100/replay_check.json

# Stage 2: materialize and naturalize the full new Macro report bank
python -m envir naturalize-subtasks \
  --config envir/implement_subtask/config.yaml \
  --scenes envir/data/maestro_web_v2_100/questions/scenes.jsonl \
  --output-dir envir/data/maestro_web_v2_reports_100

# Stage 3: author one training sample per Main decision round
python -m envir.stage3.generate_turns \
  --scenes envir/data/maestro_web_v2_100/questions/scenes.jsonl \
  --reports envir/data/maestro_web_v2_reports_100/subtasks.jsonl \
  --output-dir envir/data/maestro_web_v2_main_turns_100 \
  --workers 4 --max-scenes 100 \
  --mode llm \
  --author-api-base http://127.0.0.1:8807/v1 \
  --author-model local-qwen3.5-9b \
  --critic-model local-qwen3.5-9b
```

For a cheaper pilot, run Stage 2 with `--max-scenes 20` into its own
new directory, then Stage 3 with `--max-scenes 20` and that pilot's
`subtasks.jsonl` into another new directory. Once accepted, run the
full generation with the fresh output paths above.

Every source acquiring from synthetic Web evidence independently freezes
`search_only`, `direct_extract`, or `search_then_extract`.
In the third route, Stage 2's `simulated_tool_calls` show a local
Search result URL feeding Extract's `url` parameter, with fact references
attached only to the evidence-producing call. Stage 3 keeps the same
tools in a Main decision's `delegate_task`; **there are no actual
HTTP requests**, and a multi-tool Sub-Agent trace is not a new Main turn.

## Frozen primitive evidence tools (Orchestra-o1 compatible)

Each fresh Stage-1 V2 Scene fixes an evidence-access policy on each
source and a corresponding **actual O1 Action** on each primitive
`scan`/`lookup_by_key` step. Macro `required_tools` are aggregated
from those primitives; Stage 2 trace steps preserve this contract as
`evidence_acquisition` plus `tool_executed: false`. Stage 3 only
inherits and verifies the frozen tools; the Author may write task-specific
reasoning/instructions/context, but **cannot select additional tools**.
A source-free evidence-composition Macro can legitimately have `tools: []`.

The O1 Main prompt lists actual
`GoogleSearchAction`, `ExtractUrlContentAction`,
`ImageAnalysisAction`, `VideoAnalysisAction`,
`ParseAudioAction`, `ExecuteCodeAction`, using their original parameter
schemas from `benchmark/omnigaia/tools`. No fake
`WorkspaceSearchAction` / `TableLookupAction` names remain.
New v2 synthetic Web sources support query-only, URL-first extraction,
and two-step search-then-extract workflows. All fixture URLs end in
`.example.invalid` and are never fetched; earlier v1 sources remain
query-addressed under their old frozen contracts.

Legacy Scene files without acquisition contracts remain supported
through an explicit compatibility adapter, but this cannot retroactively
certify that their original Stage-1 source/tool choices were frozen.
Regenerate Scene and Stage-2 files for the new audited tool contract.
The offline pipeline still never actually executes network, media or
code tools, even though it uses their real O1 interface names.

## Try 20 reviewed Main-turn trajectories using local Qwen

```bash
git pull origin main
python -m envir.stage3.generate_turns \
  --scenes envir/data/maestro_current_100/questions/scenes.jsonl \
  --reports envir/data/maestro_current_reports_100/subtasks.jsonl \
  --output-dir envir/data/maestro_main_turns_20 \
  --max-scenes 20 \
  --workers 4 \
  --mode llm \
  --author-api-base http://127.0.0.1:8807/v1 \
  --author-model local-qwen3.5-9b \
  --critic-model local-qwen3.5-9b
```

The model endpoint is the default in the Stage-2 sample config; change
it if your local server uses a different port/model. Critic is a separate
call and must pass the independent grounding/style review. There is NO
silent acceptance of a template when Qwen Author/Critic rejects an action.

To smoke-test the format without an API, use
`--mode template` instead of the three LLM flags. **Template-mode
actions are deterministic baselines, not independently reviewed
realistic training labels.**

A single output directory contains:

| File | Meaning |
| --- | --- |
| `main_turns.jsonl` | Main **input.prompt** and **output** action (plus `output_json`) per decision |
| `episodes.jsonl` | Whole O1-style timeline for inspection, including the simulated Sub-Agent batch responses |
| `private/teacher_alignment.jsonl` | Internal Macro IDs, visibility sets and Author/Critic audit; **never train on this** |
| `failed.jsonl` | Rejected/incomplete scenes and explicit causes |
| `journal/000000.rounds/round-0001.json` | Atomic Main-turn checkpoints, saved immediately after each decision |
| `journal/000000.json` | Atomic complete-scene checkpoint (published or rejected) |
| `summary.json` | Round counts, completed/rejected scenes, delegation versus completion counts |

By default Stage 3 uses **4 concurrent Scene workers** (`--workers 4`).
Each worker generates its assigned Question's Main rounds **serially**, so
prior Sub-Agent results still become visible only in later rounds. Author
and Critic requests for *different* Scenes may overlap on the same local
model server. Reduce to `--workers 1` if the serving endpoint cannot
handle concurrent requests, or increase it within `1..64` if capacity
permits. Worker count affects throughput only and can change on resume
without invalidating existing Scene/round checkpoints.

The CLI displays one global Scene `tqdm` bar plus one reusable Main-round
bar per worker (four with the default setting). Add `--no-progress` for
batch logs and scripts.

**Incremental persistence:** after each accepted Main decision (including
`complete`), Stage 3 atomically writes a round checkpoint to
`journal/<scene-index>.rounds/round-<round>.json`. If the process stops in
round 4, the first three round decisions will be replayed from checkpoints
and the LLM Author will only be called for unfinished rounds on restart.
A saved decision is checked against the newly reconstructed O1 prompt,
current DAG wave and assignment constraints before reuse.

The published `main_turns.jsonl`, `episodes.jsonl` and private
`teacher_alignment.jsonl` are **incrementally appended only when the
whole Scene succeeds**. An incomplete Episode is never part of the
training set. Workers write to separate round/Scene journal paths;
the coordinator alone appends the shared JSONL files, so workers never
interleave records. During generation, completed Scenes can be appended
in *completion order*. At successful shutdown or restart, the coordinator
rebuilds the canonical JSONL files in original Scene order from the atomic
`journal/<scene-index>.json` records. This also repairs interrupted
appends without duplicate samples.

To resume, rerun the **exact same command**. The generation manifest
locks source-file hashes, model parameters and relevant code. Use
`--overwrite` to discard all checkpoints and start a new run. Source files
remain unchanged. Rejected Scenes remain rejected unless regenerated in a
new output directory; partial rounds from a process interruption are
resumable.

**Important contracts**: a single Main `delegate_task` delegates all
currently independent Macro slots within the teacher's approved wave,
but each Sub-Agent instruction/context is natural O1 prose; Macro IDs
exist only in private alignment. The history is accumulated strictly in
delivery order. Current/future reports and Gold answers are never used
in the Main input or in action authoring. For pure comparison/percentage
goals, Main may `complete` after both earlier Sub-Agent reports actually
communicate the required counts and the teacher verifies the arithmetic;
otherwise the last Macro is delegated. All factual source observations
remain simulated, with no real web/video access and no independent
Agent-solving benchmark.

## Older Stage-3 trusted simulator (mechanics test only)

The following trusted `TrustedSimulator` implementation is preserved as
a state-machine smoke-test tool. It is **not** the primary dataset generation
entrypoint and its actions (`{"type":"delegate","macro_task_id":"..."}`)
are **not** Main-Agent supervised labels.

---

### Legacy simulator API

This is a **synthetic, cost-sensitive decision environment**, not an
independent external question-answering benchmark. Stage 1 owns the frozen
Source World and Gold Program; Stage 2 owns reviewed, source-grounded
Sub-Agent reports. Stage 3 controls *when* each task is delegated and *which*
facts become visible to the Main-Agent.

## Private / public contract

Create a **private** engine from a fully replay-valid Stage-1 scene and a
complete set of independently accepted Stage-2 reports:

```python
from envir.stage3 import TrustedSimulator

teacher = TrustedSimulator(
    scene, accepted_stage2_rows, mode="reviewed",
    budget=40.0,
)
main_agent = teacher.public_session()

observation = main_agent.reset()
# Only JSON serialized from reset(), observe() or step() goes to the model.
# The Main-Agent MUST NEVER receive the teacher object.
```

The teacher stores the World, full Gold Program, full Macro DAG, future
reports, true answers, authorized source fact rows and costs. It may call
`teacher.evaluate()` **after** the trajectory to obtain private correctness
and cost labels. The public Main-Agent interface exposes only:

- `reset()`: public question, non-sensitive source catalog, currently ready
  Macro objectives, budget, no future results
- `observe()`: only currently available tasks, reports already delivered,
  accepted task IDs, action history and remaining budget
- `step(action)`: validated action, incremental observation, public cost and
  completion status. **No success/correctness oracle is returned.**

All dependencies are enforced by the environment. A Macro is eligible for
`delegate` only when its upstream Macro results have been accepted. The
complete DAG is never disclosed, even though source descriptions and the
next *eligible* task objectives may be listed.

### Actions available in the trusted MVP

```json
{"type": "delegate", "macro_task_id": "work_M1"}
{"type": "request_evidence", "macro_task_id": "work_M1"}
{"type": "hold", "macro_task_id": "work_M1"}
{"type": "retry_same", "macro_task_id": "work_M1"}
{"type": "accept", "macro_task_id": "work_M1"}
{"type": "terminate"}
```

The default costs are respectively **1.0, 0.5, 0.1, 1.0, 0.1, 0.0** units,
configurable when constructing the teacher. Invalid dependencies, hidden
task access, repeated evidence requests, unreceived report acceptance and
insufficient budget raise distinct errors; they do not publish private
data. `request_evidence` reveals only the already executed Macro's
authorized evidence atoms, **after** its report arrives.

In trusted mode, `retry_same` returns the same report; it is intentionally
not an independent verification action. `verify_independent`,
`switch_backend`, adversarial observations and consequence critic rewards
are reserved for the next Stage-3 iteration, when they can be implemented
with real, distinct causal consequences rather than invented confidence.

The public completion state does **not** report whether a submitted final
answer is correct. The teacher privately evaluates the entire trajectory,
answer equality, dependencies, cost and accepted Macro completion.

### Deterministic mechanics smoke test

The following follows a deterministic reference policy **to validate the
environment only**; it does not use a model, measure generalization, or
require real image/video perception.

```bash
git pull origin main
python -m envir.stage3.demo \
  --scenes envir/data/maestro_current_100/questions/scenes.jsonl \
  --reports envir/data/maestro_current_reports_100/subtasks.jsonl \
  --budget 40
```

Only **complete, accepted** Stage-2 scenes can be used in reviewed mode.
The demo automatically selects a complete scene (or use `--task-id`).
To smoke-test frozen Stage-1 data without any Stage-2 reports, omit
`--reports`; this is explicitly labeled `oracle_baseline` and should not
be treated as realistic Sub-Agent generation.

```bash
python -m unittest envir.tests.test_stage3_simulator
python -m unittest discover -s envir/tests
```

This is a public **model interface** boundary, not a Python process
sandbox; never make the engine or its internal attributes a model tool.
