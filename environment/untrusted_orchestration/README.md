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
behaviors across rounds, but successful cross-feedback conflict recovery is
prioritized ahead of routine missing/partial repairs. Production defaults now cover **every accepted feedback round** and
all executable one-atom candidate faults. The default `null` values for
`branches_per_scene`, `max_candidates_per_round`,
`max_selected_per_round`, and `max_language_attempts_per_round` remove
sampling and review caps; explicit integer limits remain available for pilots.
Behaviorally equivalent recoveries are still deduplicated. The selection signature distinguishes conflicts with
different *actually observed* historical report rounds. It never picks a
non-recovered candidate as positive SFT. `max_diagnostics_per_scene` additionally chooses a
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

**Latest feedback versus prior history.** A newly received report in wave
N is compared with **every earlier public wave 1..N-1**, not just N-1.
A single newest report may disagree with several different earlier reports;
all distinct grounded claim pairs are recorded, while the newest task requires
only one actual recheck per task version. An old report is never changed just
to manufacture a conflict. Stage 4 now generates two
Main-observable issue families only:

1. **Latest report itself is abnormal** (missing, partial, incorrect scope,
   or explicit Result/Summary/available-input mismatch). The fault is
   applied to the latest returned Sub-Agent batch; Main acts on that
   feedback and its past observations.
2. **Latest report conflicts with a prior report.** Earlier and newest
   public Result assertions are compared ONLY when the entity, field and
   record scope match. The disagreement is not proof that either side
   is true, and same-source reports are never counted as independent
   corroboration. Main can recheck, repair or continue independent work.

**History-first keyed handoffs** are now planned before selecting the
fault. An earlier Sub-Agent must have actually delivered a cited, unique
key; a later Sub-Agent must execute a genuine authorized `lookup_by_key`
whose output preserves that very field. Stage 4A then preferentially
chooses a *real executable* fault that makes the latest delivered value
disagree with the already-public input. The old report is reformatted
only when it literally contained that key, so a Main-visible comparison
requires no future lookup, full source scan, or hidden Gold.

When that candidate is unavailable, Stage 4 still checks for a row
in an earlier task's already executed WorkProduct. If that is unsupported,
it can instead schedule one **bounded
source survey in an existing earlier batch**, provided that batch has a
legal spare slot **and the source permits browsing**. Keyed-only
stores may never be scanned in a prior wave without an already available
authorized upstream key. The survey returns all existing key/value pairs (up to
24 source rows), with original FactRefs; it does **not** select a future
answer or lookup key in the earlier Main dispatch. When the newest report
arrives, its actual executed lookup key is stated explicitly in public
feedback, and only a matching survey row may contradict its delivered
field. Project `rename_fields` is respected for source/output schema
matching, and multiple result rows require an unambiguous public row key.
The clean and faulty branch share the same earlier survey.
No new World sources, guessed FactRefs, or unsupported private-only
comparisons are used.
For the first feedback batch (no earlier wave), or without comparable source
evidence, the candidate remains a diagnostic negative.

These are branch-local changes to Stage-4 scenario presentation; frozen
Stage-1/2/3 source artifacts are never rewritten.

The deprecated same-batch auxiliary verifier and its synthetic-source
interfaces remain deleted. Private audit records
`history_conflict_anchor`, optional `prior_observation_patch`, and
`paired_clean_history` to preserve comparable clean/fault contexts.
Summary reports `historical_anchor_status_distribution`,
`cross_round_report_conflicts_detected`,
`historical_conflict_older_round_distribution`,
`historical_conflict_round_gap_distribution`,
`selected_recovered_historical_conflicts`, and
`selected_historical_conflict_round_gap_distribution`. Individual 4A
candidate rows contain `history_conflict_older_rounds` and
`history_conflict_round_gaps`. These fields are **private selection/audit
metadata**, not additional facts exposed to Main.

4C now checks `done_but_incomplete` against the structured, flattened
O1 reports, not against `json.loads()` of the formatted history prose;
the Result/Summary conflict contract recognizes Dossier/ENT/PRJ IDs.
The 4C Critic is explicitly instructed that different attributes of the
same entity do not constitute rival same-claim observations, and that a
source-free comparison still needs a delegated derivation step.

`summary.json` also reports `published_result_format_by_phase` for
`clean_prefix`, `faulted_latest_batch` and `recovery_continuation`.
It distinguishes `legacy_prose`, `cited_record_prose_v1`,
`typed_evidence_json` and `missing`, to expose surface-format shortcuts,
**not** a claim that the surfaces are already normalized. Rewriting frozen
Stage-3 clean reports or private fact provenance to hide that drift would
invalidate clean/fault pairing; any future training-time normalization must
preserve Main-visible source evidence and be evaluated separately.

**II. Cross-report disagreements** are not separate fault injectors:
an actual wrong/fabricated latest report plus a previously delivered
same-claim record can create an observable disagreement. Independent
sources, shared sources and Result/Summary discrepancies remain explicitly
distinguished. The Main does not know the private fault origin and must
recheck rather than treating the older report as automatically correct.
If the comparison remains unresolved after the bounded retry budget, the
branch is not published as positive supervision.

In `private/candidate_selection.jsonl`, compare the **private fault
mechanism** with `observable_issue_family` (`intra_report_issue`,
`inter_report_conflict`, `not_observed`), `observability` (immediate,
later, or not witnessed), optional `fault_subtype_private=wrong_scope`,
and `recovery_effect` (execution unblocked, inconsistency resolved,
incorrect answer genuinely corrected, downstream recomputed, or not
recovered). These aggregates are also exported to `summary.json`.

Stage 1–3 remain frozen. Some scenes genuinely have no legal comparable
prior records; Stage 4 records that limitation instead of inventing
a source or claiming unsupported verification.

## Functional pipeline

1. `trusted_prefix.py`: validate clean, executable historical batches.
2. `scenario/candidate_exploration.py`,
   `scenario/candidate_selection.py`: explore
   **one atomic fault per candidate**, replay recovery and select
   causally distinct branches before expensive language review.
3. `scenario/handoff_conflicts.py`, `scenario/history_conflicts.py`,
   `scenario/prior_observation.py`, `state_machine.py`,
   `runtime/evidence_planning.py`: compare historical claims,
   create supported prior-source observations, version WorkProducts,
   track invalidations/retries, generate permitted evidence requests.
4. `reporting/report_contract.py`, `reporting/report_author.py`:
   render actual attempt
   deliveries and review Sub-Agent language, without restoring hidden
   clean values.
5. `inspection/visible_claim_extraction.py`,
   `inspection/visible_claim_state.py`: compute
   **Claim–Source–Dependency** snapshots from only explicitly returned
   public report text. Source origins come from structural provenance,
   never from guessed agreement. Retries replace old claims. Unparseable
   text and unknown origins remain unverified.
6. `orchestration/main_decision_contract.py`,
   `offline_pipeline.py`: naturalize
   the selected legal Main next action against only the current
   public evidence; Oracle/Gold remains private.
7. `inspection/publication_guard.py`: fail closed on private/internal wording
   leaks and publish only globally recovered reviewed positives.

## Stage-2 evidence contract reuse (Stage-4 only)

For frozen `task_plan` scenes with `result_encoding: typed_evidence_json_v1`,
new Stage-4 reports render delivered records as cited text using
`cited_record_prose_v1`. Every value, null and FactRef is preserved, including
nested final answers. Result is code-owned; the LLM writes Summary and Trace.
Historical clean Stage-3 reports are reused verbatim.

The same public-text parser supplies claim extraction, missing-field checks,
conflict detection and 4C visibility checks. Legacy JSON deliveries remain readable.
Validation compares the complete rendered delivery with the executed Attempt;
it never fills missing fields from the clean answer. Report-only faults mutate
their communicated values before rendering and remain privately identified.

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
`environment/untrusted_orchestration/rewards/orchestration_reward.py`
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

## Slightly expanded local Qwen pilot

The sample configuration contains reviewed Stage-2/3 inputs for ten scenes.
To expand beyond a three-scene smoke test **without exceeding the available
trusted inputs**, run:

```bash
python -m environment generate-untrusted-turns \\
  --config environment/configs/untrusted_orchestration.yaml \\
  --max-scenes 10 --branches-per-scene 3 --workers 4 \\
  --mode llm \\
  --output-dir environment/data/stage4_history_v6_10
```

`candidates_simulated` counts inexpensive preflight trials, while
`main_turns.jsonl` counts individual Main-Agent training samples. Inspect
`historical_anchor_status_distribution`,
`cross_round_report_conflicts_detected`,
`outcomes`, and `published_result_format_by_phase` in `summary.json`.
The local Author/Critic APIs must be running as configured; CI only provides
offline and small paired smoke checks.

### 4B FactRef-count guard

A prose claim such as `14 fact references` is permitted only when the
delivered Result really cites 14 **unique** nonempty FactRefs. The guard
normalizes only that exact, explicitly labeled count phrase; an unsupported
count or unrelated number remains invalid. Date wording remains anchored
to actual observed ISO dates. Neither rule loosens general factual checks.

## Source-tree layout (current)

The root package keeps only primary CLI/pipeline entrypoints
(`generate_turns.py`, `config.py`, `trajectory_builder.py`,
`offline_pipeline.py`, `state_machine.py`, `trusted_prefix.py`,
`decision_reward_replay.py`). Implementation modules live in:
- `scenario/`: source-history-first candidate generation and selection
- `runtime/`: source-constrained fault execution and evidence planning
- `reporting/`: Sub-Agent report content, fidelity and report-level faults
- `orchestration/`: Main action binding, contracts and language fidelity
- `inspection/`: only Main-visible claim extraction, history and safeguards
- `rewards/`: deterministic per-round reward contracts

Old top-level file paths are deleted, not maintained as compatibility wrappers.
Calls from the training pipeline and tests use the new canonical module paths.

### Later same-source repeated observations

In addition to faulting an existing latest-batch Sub-Agent (method A), a
scenario can now add a **real Main-delegated recheck** in a later feedback
batch (method B). The recheck executes the same earlier authorized source
and upstream inputs, with the additional task explicitly recorded in the
Main assignment as well as the returned feedback. It is only generated
when the earlier public report explicitly supplies the complete query result.
The same frozen query and upstream input versions can be compared as returned
record sets, including mistaken entity bindings and multi-source provenance.
Only the Main-visible old and new rows enter this comparison. Scalar independent
source checks retain their provenance rules.

A repeat may consume a value originally returned several waves earlier. When
the repeated result has a still-pending consumer, the scenario redirects that
consumer's typed input to the new attempt. Earlier executed tasks and same-batch
tasks keep their existing dependencies. The private unrepaired continuation
records whether accepting the conflicting new value would change the answer;
a clean repeated value must preserve the original computation. A shared source
is recorded as a same-query disagreement.

The behavior signature includes comparison kind, report pair, public conflict
object and values, and actual history rounds. It merges equivalent atomic fault
variants while retaining different Main-visible conflicts. Finite debugging
candidate caps reserve space for existing-task faults and historical repeats,
with round-robin coverage of available history gaps. Production enumeration
continues to honor null candidate limits.

With unlimited per-round candidate selection, Stage 4 can review more
trajectories; on large runs, use `--mode template` to audit coverage before
running the LLM mode. The replay budget (`max_post_fork_rounds`) and
parallel execution safety limit (`max_parallel_tasks`) are separate from
candidate sampling and remain explicit.
