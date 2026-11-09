# V2 pilot: Reasoning Program + Source–Entity–Observation

This is an isolated, opt-in, source-scoped execution pilot. It does **not**
modify the existing commands: build, realize-world, naturalize or replay.

## Goal

The former Macro TaskGraph can have many nodes even when a one-step function
over the entire Teacher World yields the answer. This pilot introduces a
separate, executable Reasoning Program in which downstream transformations
consume the actual typed outputs of upstream operations, not raw facts
silently forwarded through all nodes.

Current files:

- envir/macro/program_ir.py — bounded relational IR with typed tables and
  provenance propagation: scan, filter_eq, filter_gte, join, project,
  unique and count.
- envir/macro/source_entity_world.py — seeded fictional entities and
  source-scoped observations in table, text, and simulated image roles.
- envir/macro/difficulty_validation.py — weak single-source/no-join baseline,
  source-removal access failures, and answer-changing counterfactual mutations.
- envir/tests/test_reasoning_ir.py — unit tests for joins, provenance,
  visibility, topology, missing sources and counterfactual sensitivity.

## Small deterministic smoke test

~~~python
from envir.macro.source_entity_world import build_linked_case, public_case
from envir.macro.program_ir import evaluate_program, answer_values
from envir.macro.difficulty_validation import audit_linked_case

case = build_linked_case(seed=17, difficulty="hard")
print(public_case(case))
result = evaluate_program(case["gold_program"], sources=case["world"]["sources"])
assert answer_values(result) == case["gold_answer"]
print(audit_linked_case(case))
~~~

Or run the repository test suite:

~~~bash
python -m unittest discover -s envir/tests -p "test_*.py" -v
~~~

The pilot's roster ledger is a simulated table; its position-code glossary
is simulated text; its review board is an **annotation-only image-source
simulation**, not real pixel perception. The hard case includes a separate
clearance notice. Both require dependent joins to produce the complete answer.

## What this does NOT prove

- The generated examples are a **test fixture**, not a general-purpose
  100/859/1000-case dataset generator or measured OmniGAIA parity.
- Passing a missing-source test proves that this particular Program needs
  that source to execute, not that no alternative Program could answer.
- Counterfactual sensitivity is stronger evidence that a fact matters but
  still does not prove globally minimal reasoning complexity.
- The "active-only" baseline is deliberately weak: it does not enumerate
  all one-operator or full-World alternative solutions.
- There is no real image/video/audio decoding, no sub-agent failures, and
  no new orchestration policy or training data produced by this pilot.

## Next development milestone

1. Generalize the seeded fixture into multiple scenario/reasoning grammars,
   with a frozen ProblemSpec and difficulty target.
2. Generate source-typed fictional observations *after* the spec is reviewed.
3. Compile the validated Reasoning Program to Macro Tasks with narrowly
   typed WorkProduct handoffs and source authorization.
4. Add stronger minimized-program and source-subset baselines, answer
   uniqueness and accidental-shortcut testing.
5. Naturalize a Question from the approved ProblemSpec only after strict
   gold execution and difficulty validation.

Keep legacy TaskGraph banks as baselines; use a new schema and output directory
for any V2 training data. Do not merge partial pilots into the production
generation path before the end-to-end tests are meaningful.
