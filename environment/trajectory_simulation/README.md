# O1 Two-Role Trajectory Simulation

Stage three uses the existing frozen worlds and accepted questions without
modifying either. It generates O1-shaped collaboration records, not real tool
execution episodes, attack/recovery branches, costs or raw tool traces.

## Main controls each round

The Main LLM receives the question, known resource descriptors and previous
public replies. It chooses the next `delegate_task` or `complete` action.
For delegation it decides the instructions, context, tools, task grouping
and parallel width. It may dispatch only part of the feasible work or combine
several feasible source scopes into one task. No fixed DAG wave is imposed.

The constrained teacher prompt also includes private feasible scope handles,
an evidence-sufficiency flag and the terminal answer format. It does not include
future source contents, the oracle answer or the symbolic reasoning trace.
These teacher hints are not part of public `state_before` or history.
This remains constrained teacher synthesis, not an unconstrained O1 benchmark.

There is no separate Main summary stage. After the round's replies return,
Main decides its next action from the updated history.

## One simulator call per Main task

Each task in `delegate_task.params.tasks` produces exactly one independently
scheduled Sub-Agent LLM request and one reply, even if several sources are
assigned to that task. The simulator receives its assigned task, scoped frozen
observations, authoritative public fact surfaces and already observed context.

Public replies in `result.subtask_results` contain `subtask_index`,
`task_instruction`, `model`, `tools_assigned`, `done`, plus:

```json
{
  "finish_result": {
    "result": "Source-attributed answer to the delegated task",
    "status": "done",
    "summary": "Concise result summary"
  },
  "trace_summary": "Compressed description of simulated work"
}
```

Discovery also uses the simulator, but receives only a frozen descriptor.
It cannot access that resource's contents until a later inspection task.
The private `task_bindings` and exact `grounded_claims` audit fields are
validated and stripped from public actions, replies and history.

With JSON mode enabled, native role requests use strict JSON Schema. The
simulator's `finish_result.result` starts with a schema-constrained prefix
containing all assigned public facts (or only the discovery descriptor).
The LLM writes any subsequent task conclusion, `summary` and `trace_summary`.
This prevents source facts from disappearing behind an early no-match answer.
The backend must support JSON Schema, including string patterns and fixed
claim values; unsupported endpoints fail explicitly. `--no-json-mode` disables
guided decoding, but does not disable output validation.

## Evidence and validation

The full frozen resource environment is compiled, including alternative
providers and distractors. A minimum reachable proof is retained only as private
supervision; Main need not follow that particular proof or inspect every source.

Discovery dependencies are AND over required anchors and OR over providers of
the same anchor. A provider must have completed inspection in a previous round.
Legacy resources use explicit parent dependencies. Unreachable dependencies
remain errors. Resource titles are aliases of their exact IDs, not independent
sources; facts about the same entity may be joined across sources.

Completion requires a nonempty inspected-source cover whose symbolic result
matches the frozen result and whose decision-information check passes, including
the full world's required attribute facts. Attribute-only `required_fact_ids`
can be empty for relation tasks, so that field alone never authorizes completion.
This uses the frozen world's closed-world semantics, not open-web completeness.

`complete.params.answer` must be a string and satisfy the deterministic terminal
contract: sets for names/relations/selections, multisets for attribute lists,
and strict scalar typing. Numeric formatting is normalized; booleans are not
numbers. There is no separate final-summary critic that can veto a correct answer.

Feedback audit claims must exactly match their assigned evidence. Public text
must retain all assigned fact surfaces and source attribution. It also has
deterministic numeric, resource and visibility checks, but these do not prove
full natural-language semantic equivalence. Review samples before publication.

## Commands

Run from the MAESTRO root using the `orchestra-o1` Conda environment:

```bash
conda activate orchestra-o1

# Compile a private environment plan, with no LLM calls.
python environment/simulate_trajectories.py plan

# Deterministic smoke test; only this offline Main has oracle access.
python environment/simulate_trajectories.py synthesize \
  --provider passthrough --max-records 3

# Main and simulator use the configured LLM profiles.
python environment/simulate_trajectories.py synthesize

python -m unittest discover -s environment/tests -v
```

Defaults in `environment/config.yaml` use `local_qwen35_9b` for both separately
prompted roles, temperature 0.2, four simulator workers and three attempts.
Profiles and `--planner-model` / `--executor-model` can select different models.
`subagent_workers` caps concurrency within a round; questions run sequentially.
Invalid actions/replies retry with validation feedback, never silent acceptance.

## Outputs

Default inputs are `environment/data/typed_worlds_2000.jsonl` and
`environment/data/questions_2000/realized_capability_worlds.jsonl`.
Default output is `environment/data/trajectories_2000.jsonl`.

Each accepted record has `trajectory[]` entries with `action`, `reasoning`,
`params`, `result` and public `state_before`. The final entry is `complete`.
Private `gold_supervision` contains the answer, proof/source alternatives and
round bindings. It must not be exported as student observation data.
Metadata marks simulated feedback and the absence of raw tool traces.
The legacy training exporter expecting `o1_equivalent_prompt` needs adaptation.

Rejected records retain stage, validation issues and available attempt diagnostics
in a separate `.rejected.jsonl`. Treat these diagnostics as privileged metadata.
`flush_every` flushes whole accepted/rejected records, not resumable round
checkpoints. Re-running synthesis overwrites its output files.

The old fixed-plan implementation remains as `_synthesize_reference_one` for
reference regression tests; the public synthesis entry point uses `o1_pipeline`.
