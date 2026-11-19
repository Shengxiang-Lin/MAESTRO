# Untrusted evidence recovery (Stage 4)

Run with the canonical config, which reads Stage-1 scenes and Stage-2
reviewed Sub-Agent messages:

```bash
python -m environment generate-untrusted-turns \
  --config environment/configs/untrusted_orchestration.yaml
```

Default output: `environment/data/stage4/`.

## Short single-fault replay (default pilot)

The maintained pilot stratifies up to three *actual returned feedback rounds*
from each accepted Stage-3 episode. It uses a seeded selection across
early/middle/late positions instead of always taking the first round.
Within each selected feedback round, Stage 4A preflights one-atom faults
against existing executable Macro results and authorized source facts;
no new World, long causal chain, or arbitrary fabricated fact is created.

Each fork produces a paired private *no-repair continuation* by executing
the original trusted schedule on the communicated (possibly poisoned)
WorkProducts. The recovery continuation starts from the same visible prefix,
uses the existing field/dependency-aware state machine, and is capped by
`generation.max_post_fork_rounds`. An expired horizon is
`short_replay_horizon`, **not** a successful recovery.

`selection_strategy: stratified` balances different causally replayed recovery
behaviors across rounds. `max_diagnostics_per_scene` additionally chooses a
small number of genuinely failed executions for 4B/4C language review;
those are **never** published as positive Main SFT turns. This sampling
changes only the fork and review distribution, not the trusted dispatch
semantics or the legality of recovery decisions.

Stage 4 normally creates a deterministic clean O1 prefix. A separate
Stage-3 LLM dataset is **not** required, but `input.clean_episodes` can point
to a reviewed Stage-3 `episodes.jsonl` for exact trajectory reuse. The
supplied episodes are checked against a fresh, executable Stage-1/2 baseline;
invalid/missing references are rejected rather than silently substituted.

`private/clean_references.jsonl` stores validated clean reference episodes;
`private/paired_trajectories.jsonl` links fault branches to the stable
`clean_reference_id` and records the no-repair continuation outcome. Neither
is included in Main prompts. Fault episodes carry the same reference ID for
later pairing across Stage 3 and Stage 4.

## Current observable-first recovery semantics

The *sampling unit* is an accepted Stage-3 `(question, Main feedback round)`.
The injected single-atom fault changes the previous Sub-Agent attempt or
its delivered report, not the next Main action. Stage 4A then simulates the
continuation using the frozen Stage-1 Macro operators and sources, including
actual retries, version replacement, downstream stale results and fresh
Sub-Agent returns. Accepted complete episodes are split into per-round Main
training records.

The default `round_sampling_strategy: evidence_aware` still covers
early/middle/late positions. Within each bucket it gives priority to
already-existing verification Macro roles, multi-source acquisition and
Join operations. It does not add evidence, change a DAG edge, or expose
future reports in a Main prompt. `stratified` remains available as a
random-position ablation.

Every 4A decision now records an `observable_issue_snapshots` entry in
its private audit. Missing, partial, internal report conflicts, observable
key mismatches, late independent same-claim disagreements and stale
descendants are classified **only from information already delivered**.
A changed private WorkProduct or injected fault name never suffices.
`private/candidate_selection.jsonl` consequently reports
`immediately_observable`, `eventually_observable` or
`not_observed_in_replay` for each candidate, plus the first observed
decision round and visible anomaly kinds.

A cross-report disagreement can be **simultaneous or delayed**. Both
claims must literally appear in the delivered Result, name the same
entity/attribute/scope, cite different **single known independent**
observation origins AND distinct source IDs, and come from non-dependent
tasks. The earlier report need not have been consumed: consumption is an
independently audited condition for subsequent downstream invalidation.
A composite Join result citing multiple origins is not a second witness.
Different *candidate sets* from complementary joins are NOT conflicts.
No hidden clean source value is used to pick a side. If the observed
evidence is insufficient, the fault remains diagnostic rather than a
fabricated positive recovery. The 4C Critic also distinguishes sufficient
external sources from a still-pending source-free derivation.

Stage-4B exposes one canonical reporting API:
`report_contract.build_report_contract` and
`report_validation.validate_report_draft`. Old redundant aliases are
removed; Stage 1–3 authoring and data contracts remain untouched.

## Maintained two-family taxonomy

**I. Intra-Sub-Agent report issues** are observable from the assigned task,
its returned evidence and any Main-visible prerequisite values:

- **Delivery:** `no_result`, `report_missing`,
  `incomplete_retrieval`, `done_but_incomplete`. The latter genuinely
  withholds a typed field but reports `finish_result.status=done`.
  Missing fields trigger repair only when the public task instruction
  identifies the field and an outstanding consumer needs it.
- **Acquisition/association:** `wrong_retrieval`,
  `wrong_entity_binding`. `wrong_scope` is an observable subtype
  of `wrong_retrieval`, **not** a second injector: an explicitly assigned
  equality/threshold constraint must be present in the Main-visible task
  instruction and contradicted by a delivered output field.
- **Reporting:** `fabricated_claim`, `result_summary_conflict`.
  The former keeps the true executed WorkProduct immutable but delivers
  a grounded distractor value with its actual FactRef. The latter
  preserves the real Result and contradicts it in Summary. Neither is
  silently laundered through a trusted LLM report reviewer.
- **Computation:** `computation_error` perturbs one actual numeric
  operator output (`count`, `group_count`, `derive_arithmetic`,
  `derive_percentage`, `derive_date_gap`, or numeric
  `goal_compute`), then reruns real downstream primitives. Only the
  simple source-free, public-input `derive_arithmetic` form currently
  receives automatic Main-visible inconsistency detection; other
  perturbations may remain diagnostic, not forced recovery examples.

**II. Inter-Sub-Agent conflicts** are not separate fault injectors:
an actual wrong/fabricated report plus a separately sourced, already
returned same-claim report can form a disagreement. Same-round and
later-visible disagreements both request rechecking, with actual
downstream invalidation only if an earlier value was used. If reports
still disagree after rechecking, the branch is unresolved, even if a
private Gold answer happened to match.

For CURRENT simulated benchmark data, set
`generation.latest_cross_report_pairs: true`. The Stage-4-only
`latest_witness.py` first attempts to identify an explicit, earlier
Stage-3 source report containing an already-returned entity/attribute
pair. For the current clean Stage-2 natural report form, the literal
`entity (value)` statement must match an actual authorized Stage-1
source observation and its evidence origin. **Only ONE supplemental
Sub-Agent** is then added to the latest received batch, with a distinct
virtual source, executable Macro and `fabricated_claim` fault that
contradicts the previously returned clean report. The old report and its
entire earlier delegation history stay unchanged.

If no such earlier literal source-backed assertion is available, the
fallback creates TWO supplemental **simulated** audit observations
of a currently grounded Stage-1 fact. One added report is corrupted,
the other remains clean, and both are attached to the latest batch only.
Both forms require matching actual subtask assignments in the immediately
preceding `delegate_task` action, source-scoped operator replay and
full FactRefs for newly synthesized reports. The original Stage-1–3
reference is strictly validated before local augmentation; the
augmented Scene never overwrites frozen files. Added clean reports
are reviewed in 4B, and 4C sees the expanded source catalog.

**Important evidential limitation:** the supplemental sources are
*benchmark-synthesized* observations, with the same underlying fact
copied from a validated original source into two constructed audit
registers. They are NOT evidence of genuine independent real-world
verification. The private `witness_extension.private_generation_basis`
records this fact. The benchmark tests the Main's behavior on multiple
apparently independent reports, not actual outside corroboration.
No invented value from this private generation contract is directly
given to Main. If a relevant anchored fact, a grounded distractor or enough
parallel capacity is unavailable, the generator skips the supplemental
scenario. Historical single-witness scenarios produce **delayed**
conflict at the target Main round, while two-witness fallback scenarios
produce same-batch conflict. Neither requires retroactively altering
earlier Sub-Agent handoffs.

In `private/candidate_selection.jsonl`, compare the **private fault
mechanism** with `observable_issue_family` (`intra_report_issue`,
`inter_report_conflict`, `not_observed`), `observability` (immediate,
later, or not witnessed), optional `fault_subtype_private=wrong_scope`,
and `recovery_effect` (execution unblocked, inconsistency resolved,
incorrect answer genuinely corrected, downstream recomputed, or not
recovered). These aggregates are also exported to `summary.json`.

Stage 1–3 remain frozen. Natural comparable cross-report witnesses may
still be absent. Stage 4 never mislabels a virtual audit register as an
original Stage-1 source, and never claims a private synthetic witness is
an independent real-world finding.

## Functional pipeline

1. `trusted_prefix.py`: validate clean, executable historical batches.
2. `candidate_exploration.py`, `latest_witness.py`,
   `candidate_selection.py`: explore
   **one atomic fault per candidate**, replay recovery and select
   causally distinct branches before expensive language review.
3. `state_machine.py`, `evidence_planning.py`: version WorkProducts,
   track invalidations/retries, generate permitted evidence requests.
4. `report_contract.py`, `report_author.py`: render actual attempt
   deliveries and review Sub-Agent language, without restoring hidden
   clean values.
5. `visible_claim_extraction.py`, `visible_claim_state.py`: compute
   **Claim–Source–Dependency** snapshots from only explicitly returned
   public report text. Source origins come from structural provenance,
   never from guessed agreement. Retries replace old claims. Unparseable
   text and unknown origins remain unverified.
6. `main_decision_contract.py`, `offline_pipeline.py`: naturalize
   the selected legal Main next action against only the current
   public evidence; Oracle/Gold remains private.
7. `publication_guard.py`: fail closed on private/internal wording
   leaks and publish only globally recovered reviewed positives.

## Stage-2 evidence contract reuse (Stage-4 only)

For frozen `task_plan` scenes with `result_encoding: typed_evidence_json_v1`,
Stage 4B now serializes the **actual current Attempt** to Stage-2-compatible
`[{"values": ..., "source_fact_refs": [...]}]` JSON. The immutable Result
is not LLM paraphrase, and it is not truncated or reconstructed from the
clean Oracle. On partial delivery, missing fields remain `null` and missing
rows stay missing; Summary/Trace describe incomplete delivery. Report-only
faults still bypass trusted naturalization and are never automatically fixed.
Long code-owned typed Results are checked by exact equality rather than the
short natural-prose result length guard; Author-written claims remain guarded.

`private/scene_evidence_affordances.jsonl` separately reports Macro-local
multi-source operations, frozen verification gates, cited source coverage,
and declared observation origins. A multi-source JOIN is **not** automatically
classified as independent corroboration. Visible Claim labels additionally
retain all explicitly cited origin/source IDs for a composite WorkProduct,
without counting a derived observation as several independent confirmations.
These are read-only private diagnostics: they do not change 4A scheduling,
Main prompts, the Stage-1 World, Stage-2 reports, or Stage-3 trusted episodes.

## Files and training semantics

- `episodes.jsonl` / `main_turns.jsonl`: only recovered, fully
  reviewed O1-compatible positive trajectories.
- `private/claim_state_supervision.jsonl`: one row per observed
  Sub-Agent batch used as next-round Main-state supervision, **only**
  for fully accepted recovery branches.
- `private/reviewed_observable_states.jsonl`: observable Claim–Source–Dependency
  states from language-reviewed recovered **and** non-recovered branches.
  `positive_main_action_label=false` rows are risk/evidence supervision
  only, not approved SFT decisions or labels of hidden factual truth.
- `private/claim_state_diagnostics.jsonl`: states for reviewed but
  unsuccessful branches; diagnostic only, **never** silently promoted
  to positive SFT decisions.
- `private/fault_alignment.jsonl`, `private/main_turns_local_review.jsonl`:
  causal/reviewer supervision and local qualification diagnostics.
- `private/decision_reward_states.jsonl`: exact-candidate, **pre-action**
  state for deterministic per-round Main dispatch reward: all currently
  executable evidence goals, source/tool contracts, task dependencies,
  priority, parallel capacity and completion eligibility. This is PRIVATE
  scorer metadata, never part of the Main prompt or O1 public action.
  Hidden-only repair requirements make a row unscorable.
- `summary.json`: generation counts plus round-wise claim state
  labels and status distribution.

Each state snapshot includes explicit Claim nodes, compatible claims'
relations, source provenance, task/Claim dependency edges, changes from
the previous state and the round at which Main can act on the new
evidence. `corroborated` means a declared independent-source warrant,
not absolute ground truth; `disputed` calls for reconciliation.
Partial evidence and missing/invalid reports never silently inherit
a clean Oracle value.

## Deterministic per-round orchestration reward

The Stage-2 scorer in
`environment/untrusted_orchestration/orchestration_reward.py`
uses action binding, evidence coverage, priority, legal parallel utilization
and focused field requests. It does **not** compare with Teacher natural
language or run a future-outcome simulator. Broken dependency, duplicate,
ambiguous, unauthorized or premature actions receive a hard negative reward.
The evaluator may inspect the global current-round execution structure,
but does not require the Main to predict hidden fault sources.

For historical Stage-4 journals that predate this field, use the
no-LLM `python -m train_maestro_8b.backfill_reward_states` command
documented in `train_maestro_8b/README.md`. This replays audited attempts
and verifies each old Teacher batch against reconstructed admissible choices
before exporting a private score state.

Completion reward now checks the candidate answer against a **private**
completion reference; a nonempty but wrong answer receives a negative reward.
Old exported score-state rows lacking this reference should be backfilled
before GRPO, rather than blindly rewarding any nonempty completion.

Only synchronous Sub-Agent result batches and single atomic injected
faults are supported. Delayed network responses, partial asynchronous
batches and multiple injected faults per branch are not generated.

For fast deterministic diagnostics use `--mode template --max-scenes 2
--output-dir environment/data/pilots/stage4`. For reviewed SFT, use the
default `llm` mode.
