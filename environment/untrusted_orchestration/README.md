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

For a late cross-report disagreement to request a targeted recheck,
both claims must literally appear in the delivered Result, name the
same entity/attribute/scope, cite different **single known independent**
evidence origins, come from non-dependent tasks, and the earlier report
must already have been consumed by downstream work. A composite Join
result citing multiple origins is not a second independent witness.
No hidden clean source value is used to pick a side. If the observed
evidence is insufficient, the fault remains diagnostic rather than a
fabricated positive recovery. The 4C Critic also distinguishes sufficient
external sources from a still-pending source-free derivation.

Stage-4B exposes one canonical reporting API:
`report_contract.build_report_contract` and
`report_validation.validate_report_draft`. Old redundant aliases are
removed; Stage 1–3 authoring and data contracts remain untouched.

## Functional pipeline

1. `trusted_prefix.py`: validate clean, executable historical batches.
2. `candidate_exploration.py`, `candidate_selection.py`: explore
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
