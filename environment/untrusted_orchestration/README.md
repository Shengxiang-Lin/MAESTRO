# MAESTRO Stage 4: Main-observable unreliability

**Goal:** From a frozen Stage-1/2/3 scene and the latest returned Sub-Agent
feedback, produce exactly ONE Main-Agent next-decision example.

## Scenario-first, not fault-first

| Target | Main-observable issue | Construction prerequisite |
| --- | --- | --- |
| A: Evidence Insufficiency | Missing, partial, false completion | Required evidence was not delivered |
| B1: Internal Inconsistency | Two incompatible claims in a single report | Explicit, same-object internal claims |
| B2: Constraint Inconsistency | Violation of public task or execution requirement | Public keys, arithmetic operands, scope or relation witnesses |
| B3: Inter-Report Inconsistency | Latest report conflicts with another delivered report | Same subject, attribute, scope and actual public reports |

The private generator first identifies feasible scenario targets, then probes
the existing executable fault operators. Stage 4A subsequently validates the
**realized** observable scenario independently. A candidate with no public
witness, or a realized class different from its target, is not released as a
positive Main supervision label.

Historical Stage 3 dispatches and reports remain byte-for-byte unchanged.
There is no inserted recheck, invented independent source, fabricated evidence
citation, hidden Gold in the Main prompt, or post-decision recovery rollout.

## Code ownership

- scenario/observable_scenarios.py: canonical two-family taxonomy and
  scenario feasibility checks
- scenario/candidate_exploration.py: active single-turn scenario-first
  enumeration; older multi-round recovery is isolated behind an explicit
  non-single-turn path
- runtime/faults.py, runtime/atomic_faults.py: existing atomic mutation and
  actual executable operator replay
- inspection/ and state_machine.py: Main-visible evidence comparison
- runtime/evidence_planning.py: legal Main next-action plan
- reporting/ and offline_pipeline.py: LLM feedback/decision author and critic
- scenario/single_turn_selection.py: independent realization check and
  scenario-first balanced selection
- publication.py: private audits, published Main decisions and scenario counts

Source independence, modality, same-round vs cross-round and downstream
dependency status are **attributes** of evidence relations, not extra
fault kinds. B3 is left empty if no comparable public claims exist.

## v3 baseline and migration

Completed v3: 3,963 candidate trials, 1,634 actionable candidates,
1,000 published turns, no published inter-report conflicts, and all
1,000 published trusted prefixes preserved. The new scenario-first
candidate order intentionally changes candidate identity and sampling;
do not claim paired numerical equivalence without keyed replay analysis.

Use a new output directory because code signatures differ. Confirm
published A/B1/B2/B3 counts, review acceptance, single-turn output
contract, no leakage of private labels and frozen Stage 3 history.

Legacy recovery-episode support is retained strictly for older callers;
it is not part of the current single-turn dataset.

## Selection diagnostics (single-turn)

`private/candidate_selection.jsonl` has two separate decisions:

- `selection_reason`: immutable Stage-4A visibility, legality, novelty and
  scenario-budget classification (`target_scenario_not_publicly_realized`,
  `duplicate_public_issue_and_action`, `single_turn_budget_reserve`,
  `balanced_public_scenario_and_action`).
- `language_review_selection_reason`: whether the candidate entered 4B/4C
  as a primary or backfill attempt. An unselected candidate is **not** a
  failed recovery: no future recovery episode is executed.

`summary.json.candidate_screening_outcomes` partitions all trials into
`selected`, `no_public_witness`, `duplicate`, and `budget_reserve`.
Use this instead of the old `not_execution_recovered` label.
