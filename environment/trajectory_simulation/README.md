# Gold Orchestration Trajectory Simulation

This stage converts the **existing frozen solution** into a clean, official
Main/Sub-Agent collaboration trajectory. It does **not** ask an LLM to design
a new full solution, execute real tools, or simulate unreliable agents.

## Provenance: no new gold solution generation

Given an accepted question plus its original frozen world, the program uses:

- `ProgramIR` and `SymbolicExecutor.execute_trace()` for the original
  dependency DAG, intermediate symbolic results and authoritative final answer.
- `minimal_source_proof()` for an actual sufficient evidence subset.
- `capability_world.resources[*].anchor_dependency_resource_ids` and
  resource visibility for required discovery prerequisites.
- The frozen world and its deterministic rendered sources for per-source facts
  and source observations.

The compiler selects one minimal proof witness and closes it over the
resource-acquisition ancestors. It builds one gold subtask per selected source
and groups currently independent subtasks into parallel waves. The compiler
never fabricates evidence or changes the frozen ProgramIR.

## Two distinct LLM roles, repeatedly invoked

**Role 1: Main Planner** (`planner_profile`)

For **every round** the Main receives the user question, overall **public**
gold task DAG (IDs, source types, capabilities and dependencies only), tasks
already completed, observed results and summaries, the currently ready tasks,
and pending tasks. It emits JSON for **only the current parallel assignment**.
The program rejects missing/extra subtasks and wrong agent/resource choices.

After all independent Sub-Agent requests from that round return, the same
Main role receives their results and produces `main_summary`. These past
results become the next round's completed history. When all official subtasks
have finished, Main produces the final answer, which must exactly match the
deterministic frozen answer.

**Role 2: Sub-Agent Executor** (`executor_profile`)

Each current-round subtask is a separate, concurrently scheduled LLM request,
receiving its assignment, already-completed prerequisite results, exactly one
frozen source observation and the corresponding authoritative fact claims.

Each response contains:

- `answer`: the **exact structured source facts** (public subject,
  predicate and object surfaces), checked against the frozen source.
  Private fact IDs and entity IDs are not returned to the Main.
- `answer_summary`: concise natural-language description of the subtask answer.
- `trajectory_summary`: compressed description of what the Sub-Agent did.
- `evidence_source_ids` and `status`: strictly validated.

This produces **one official trajectory increment at a time**, not a model
hallucination of all rounds in one call. Main never receives future source
contents, future answers or the oracle final answer in its actual LLM prompt.
The offline `--provider passthrough` test mode has programmatic access to
the oracle so end-to-end no-network tests can complete.

**Validation boundary:** all structural IDs, source claims, intermediate
dependency checks and final answer are deterministic. The natural-language
`answer_summary`, `trajectory_summary` and `main_summary` are constrained by
the provided evidence, but are **not** independently semantically proven by
the current validators. Review representative samples before publication.

## Commands

From the MAESTRO root after constructing frozen tasks and generating accepted
questions:

```bash
# Inspect the compiled official solution DAGs without any LLM calls.
# Writes a separate .gold_plan.jsonl by default, not the trajectory output.
python environment/simulate_trajectories.py plan

# Smoke-test without LLM requests (deterministic role implementations).
python environment/simulate_trajectories.py synthesize \
  --provider passthrough --max-records 3

# Generate the actual LLM-authored gold trajectories.
python environment/simulate_trajectories.py synthesize
```

The YAML model configuration defaults to **one local Qwen3.5-9B vLLM
service**, acting as two separately prompted and independently callable roles:

```yaml
trajectory_simulation:
  planner_profile: local_qwen35_9b
  executor_profile: local_qwen35_9b
  planner_temperature: 0.2
  executor_temperature: 0.2
  subagent_workers: 4
  max_attempts: 3
  flush_every: 5
```

For different role models, edit their profiles independently, or use
`--planner-model` and `--executor-model` to override the model names.
For source-specific backend endpoints, configure different named model
profiles in `environment/config.yaml`.

- `subagent_workers` controls parallel requests **within the current round**.
  Rounds of a task, and different tasks, are processed sequentially.
- `max_attempts` applies per Main allocation, Sub-Agent result, Main summary,
  or final answer. Invalid outputs are retried with targeted validation
  feedback, never silently accepted.
- `flush_every` periodically flushes completed **whole** trajectories to
  accepted/rejected JSONLs; it is **not a resumable checkpoint**.
  Re-running `synthesize` currently rebuilds and overwrites those output files.
  Use the `plan` command to review the DAG before expensive LLM calls.

## Outputs

Default inputs:
`environment/data/typed_worlds_2000.jsonl` and
`environment/data/questions_2000/realized_capability_worlds.jsonl`.

Default output `environment/data/trajectories_2000.jsonl` contains one JSONL
record **per accepted question trajectory**, with:

- `plan`: public official task DAG (without future evidence or answers);
- `steps[]`: each round's current `main_action.parallel_subtasks`,
  parallel `subagent_results`, `main_summary`, and before/after task state;
- `final`: verified complete action and final question answer;
- `validation`: structural/proof/final-answer checks;
- `gold_supervision`: private oracle/proof IDs/symbolic execution trace for
  training/inspection, **not** to be included in Main prompts.

Failures are recorded in `trajectories_2000.jsonl.rejected.jsonl`, with
reason/stage and validation issues. Accepted/rejected counts and reasons
are printed when synthesis completes.

## Tests

```bash
python -m unittest discover -s environment/tests -p "test_gold_trajectory.py" -v
python -m unittest discover -s environment/tests -p "test_config.py" -v
```

This version intentionally covers only the clean gold case. Fault injections,
recovery branches, tool execution and reward/counterfactual labels belong to
future stages and are not generated here.

## Compiler and validation fixes (2026-11-04)

The frozen worlds and accepted questions are unchanged. Acquisition is resolved
as AND over required anchors and OR over their providers. Only an already
reachable provider is selected; anchors from a wave are published after that
wave. Legacy records without anchor metadata retain explicit dependency handling.
Unknown/unreachable anchors remain errors rather than being silently ignored.

Resource selection preserves the full world's decision-required facts, the
symbolic answer, and decision-information completeness in the selected evidence.
This includes the losing candidates' values in comparisons/rankings. Shared facts
remain observable whenever any selected source provides them.

A discoverable resource has two tasks: `discover:<source>` searches from known
clues through Main's search action; `subtask:<source>` inspects its contents with
the registered specialist after discovery. Discovery returns an environment-owned
descriptor deterministically. The LLM Sub-Agent receives contents only for
inspection; it cannot invent a search result or read a future resource.

Final validation follows the terminal answer contract: unordered sets for names,
relations and selections; unordered multisets for attribute values; ordered lists
only when explicitly specified. Finite numeric formatting is normalized, while
boolean values remain distinct from numbers. Sub-Agent structured claims still
must match their authoritative facts exactly.

Generated summaries pass deterministic checks and a scoped role-model grounding
review. Public question values may describe search/filter objectives; they are
not automatically treated as observed evidence. Local subtask reviews must not
demand completion of the whole question. The offline provider uses deterministic
reference text and does not perform an LLM semantic review.

Rejected trajectories now retain attempt responses and validation issues. Final
answer failures also record the expected answer and answer contract in the
rejected audit, never in the real Main LLM prompt. Treat rejected audits and
`gold_supervision` as privileged metadata when exporting student examples.
