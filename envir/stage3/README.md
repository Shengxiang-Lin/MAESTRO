# Stage 3 — Trusted Orchestration Simulator (MVP)

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
