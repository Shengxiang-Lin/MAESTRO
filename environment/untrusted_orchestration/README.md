# Stage 4 — Untrusted orchestration and causal recovery

Stage 4 produces **per-round Main-Agent decisions** from executable,
Main-visible Sub-Agent histories. Stage 1 supplies the frozen task graph
and source contracts; Stage 2 supplies reviewed reports; Stage 3 supplies
the clean reference. Stage 4 runs separate private counterfactual and
recovery executions and publishes only validated recovery decisions.

## Entry point

```bash
python -m environment generate-untrusted-turns \
  --config environment/configs/untrusted_orchestration.yaml \
  --max-scenes 10 --workers 4 --mode template \
  --output-dir environment/data/stage4_pilot_10
```

The canonical config enumerates **all eligible feedback rounds** and
all executable single-fault source variants. Set explicit numeric
`branches_per_scene`, `max_candidates_per_round`,
`max_distractors_per_fault`, `max_selected_per_round` or
`max_language_attempts_per_round` only for bounded smoke runs.
`max_post_fork_rounds` is a **causal replay horizon**, not a sampling
cap; expiry is not successful recovery. Run `--mode llm` only after
checking template-mode source/recovery diagnostics.

## Execution performance

4A is deterministic **CPU-heavy** replay. It now uses a process pool
bounded by `--workers` and the actual available CPU quota, whereas 4B/4C
continue using threads for Author/Critic calls. Frozen Stage-1/2 report
validation is reused once per accepted scene; per-candidate atomic
journals are unchanged. Runtime gains depend on real allocated CPU cores;
20 workers do not imply 20 available cores.

## Global training budget and fine-grained recovery

Stage 4A enumerates all executable grounded faults independently of the
training budget. After 4A completes, `selection_strategy: global_balanced`
and `target_main_turns: 10000` choose **entire recovered Episodes**
whose estimated Main decisions fit that global budget. The budget is
measured in post-fork Main **turns**, not in Episode count or trial count.

- Fault-kind allocation is availability-aware: kinds that have no
  demonstrably recoverable candidates receive **no fabricated quota**.
  Within each kind, rare recovery effects and strategy signatures are
  prioritized. Selection is deterministic for a fixed seed.
- Candidate reserves remain in 4A for review failures; 4C can replace
  failed chosen cases without reopening all reserves or exceeding the
  global target. Genuine non-recovered cases remain private diagnostics.
- A target is a **maximum**, not an assertion that enough eligible
  Episodes exist. Any unmet target is explicitly reported; individual
  Episodes are never truncated to force an exact count.
- CLI override: `--target-main-turns 10000`. Existing debug limits
  such as `--max-scenes` affect supply, not the budget itself.

Stage 4B commits each completed authored Sub-Agent report to
`stage4b/journal/<candidate>.rounds/round-<ordinal>.json`; Stage 4C
commits each reviewed Main decision to
`journal/<candidate>.rounds/round-<ordinal>.json`. After all units in a
candidate finish, the original atomic candidate journal is committed
as before. On restart the signed manifest is checked, and per-item
checkpoints replay exact report contents/authoring statuses or Main
actions with matching task identities, assigned instructions, and
prompt hash. A mismatch is a **hard validation failure**, not permission
to regenerate a different continuation silently. Only globally accepted
recovered Episodes are published as positive SFT.

For audit, `summary.json` exposes
`target_main_turns`, `estimated_selected_main_turns`,
`selected_recovered_by_fault_kind`,
`fault_kind_execution_outcomes`, `fault_kind_observability` and
`computation_fault_operators`. These distinguish structural fault
availability, visible evidence, execution recovery and actual published
coverage.

## Reporting modules and stable metrics

Stage 4's report package is organized by **responsibility** rather than
repeating `report_` in every filename:

| Module | Responsibility |
| --- | --- |
| `reporting/author.py` | 4B LLM Author/Critic and approved narrative |
| `reporting/contract.py` | Source-grounded delivery/step contract, including complete-step handoff witnesses |
| `reporting/validation.py` | Pure draft fidelity checks and rejection reasons |
| `reporting/faults.py` | Explicitly controlled report-only perturbations (not normal authoring) |
| `reporting/surface.py` | Lossless Result encoding/decoding and public prose sanitization |
| `reporting/work_evidence.py` | Public calculation/association statements and text-only checks |

The former tiny `report_evidence.py` and `public_language.py` modules
are merged into their owning modules. The superseded `report_*.py`
files and their unused compatibility imports are removed. No
LLM prompts, report schema, fault executor, or reward scoring rules are
intentionally changed by this package refactor.

`summary.json` uses **canonical** quantities:
`candidate_trials` (all 4A trials),
`rounds_extracted_from_base` (feedback rounds),
`candidates_selected_for_language_review` and
`candidates_filtered_before_llm` (language-review selection),
`recovered` / `published_episodes` (accepted Episodes),
`main_training_turns` (published post-fork Main decisions),
`decision_reward_states` / `decision_reward_unscorable`
(reward coverage), and
`publication_generation_process_leaks` (one leak audit).

Removed aliases: `requested_branches`, `requested_feedback_rounds`,
`reviewed_branches`, `reviewed_turns`,
`screened_out_before_llm`, and `publication_surface_leaks`.
The constant `candidate_backfill_diversity` string and hard-coded
`limitations` list are also removed from each summary: explanation
belongs in this README, not in a numeric run report. These changes do
**not** remove source-backed detailed counters (fault distribution,
cross-round conflicts, report review failures, replay failures) or
private per-case audit files. Compare old and new summaries using the
canonical quantities when reviewing previous pilot results.

## Ten-question official LLM language pilot (1,000 Main turns)

Use the isolated `environment/configs/untrusted_orchestration_llm_pilot_10.yaml`
after confirming the local `local-qwen3.5-9b` OpenAI-compatible API
is reachable at the configured `127.0.0.1:8807/v1`. This intentionally
selects at most **1,000 published post-fork Main decisions**, not 1,000
Episodes. Both Stage 4B report narratives and Stage 4C Main reasoning
are generated with the real Author/Critic, not template teachers.
The malicious surface of a report-only injected fault remains controlled
to preserve the actual delivered contradiction, never silently corrected.

```bash
python -m environment generate-untrusted-turns \\
  --config environment/configs/untrusted_orchestration_llm_pilot_10.yaml \\
  --max-scenes 10 --workers 8 --mode llm \\
  --output-dir environment/data/stage4_llm_pilot_10_1000
```

Stage 4A remains exhaustive and its results are source-backed; the
global Main-turn budget limits expensive language review after 4A.
Stage 4B/4C have separate per-report and per-Main-decision atomic
checkpoints. Rerun the same command without `--overwrite` for a
signature-checked continuation. Missing or rejected LLM reviews do
**not** become accepted template-based SFT labels. Inspect the final
`summary.json` for actual published turns and rejected/fallback counts.

The report contract now includes concise *observed work evidence*:
actual arithmetic inputs/results and attempted Join key pairs, with
the same rendering rule for clean and corrupted executions. Main may
recompute/check only these literally reported facts, never private
Gold or injection labels. For `done_but_incomplete`, a finished attempt
may truthfully say a required extraction remained incomplete: natural
4B language is LLM-authored, while the completion-status overlay stays
controlled. Generic absence of an internal field name in the user
instruction is no longer a prerequisite for verification.

## Safety and causal invariants

1. The initial fault is **one** real executor or delivery fault at an
   explicit, actual Sub-Agent execution wave. No arbitrary wrong values,
   new data sources or invented FactRefs are permitted.
2. Main receives only delivered reports and its preceding actions.
   Fault identity, attempts, World/Gold, source provenance and
   deterministic teacher state are **private**.
3. A report cannot be inserted without an associated legal Main
   delegation and executable task. Same-source repeated queries are
   **report discrepancies**, not independent corroboration.
4. Every output or claim whose input changes must be replayed on the
   **actual communicated WorkProduct**. Clean Stage-3 downstream
   values cannot be reused after an earlier fault.
5. Retry, verification and rollback use visible claims, admitted source
   permissions, versioned dependencies and legal scheduling.
   Final correctness alone is insufficient to label a recovered episode.

The optional reviewed Stage-3 `input.clean_episodes` must match the
validated Stage-1/2 reference fingerprint. Without it, Stage 4 builds a
deterministic clean prefix. Each candidate retains the same immutable
clean reference and a scenario-local private prefix augmentation.

## Scenario construction

**A — Existing latest-batch fault.** Mutate an actual Sub-Agent in the
latest returned batch. Missing, incomplete, acquisition, entity binding,
arithmetic and report-only faults all require executable preflight.
Main may retry when the returned evidence itself is anomalous.

**B — Later same-source recheck.** Find a complete *publicly returned*
source-query result, then explicitly add a later authorized Main
assignment to repeat that exact query with the same upstream versions.
A legitimate fault in the *new* report may cause a discrepancy with an
old report. B may update future, not-yet-executed consumers' input
binding when that binding remains legal. Old completed results are not
rewritten.

**C — Early fault with delayed discovery and actual propagation.**
Choose an old report with at least one **already executed intermediate
descendant**. Poison that old execution, replay the intermediate
descendant(s) using the wrong communicated result, and delegate a
**clean later repeated query**. Keep original downstream dependency
edges. After Main sees the conflict, the bounded teacher verifies the
new report first, then the older side if disagreement remains.
An older corrected WorkProduct must invalidate genuinely consumed
descendant versions, which must then be recomputed. A case is
publishable as recovered **only** when the private audit proves:

- A real intermediate WorkProduct differed from its clean reference
- An old/new contradiction was visible from public reports
- A completed intermediate consumer was actually invalidated
- The invalidated consumer was actually re-executed

If any condition fails, keep it as a private unsuccessful diagnostic;
do not label a speculative rollback as positive SFT.

**History-first keyed handoff.** An earlier public Sub-Agent can provide
a genuine lookup key for a later authorized lookup. The old report is
never fabricated to force a match. Optional earlier-source surveys
require legally browseable sources and an explicit historical Main
assignment; keyed-only sources cannot be scanned without a public key.
When no comparable public evidence exists, the case is ineligible.

An explicit independent-source claim dispute requires distinct
source/observation origins. A same-source repeated-query discrepancy
instead compares the complete public result sets for the same frozen
query and parent-version binding; it never uses Gold to infer a
missing public claim.

## Stage 4 pipeline and package responsibilities

- `generate_turns.py` / `config.py`: production orchestration,
  manifest, resumable private journals, accepted publication.
- `trusted_prefix.py`: executable clean-reference validation.
- `scenario/candidate_exploration.py`: fault and recheck preflight;
  `scenario/candidate_selection.py`: causal 4A selection and dedup.
- `scenario/same_source_recheck.py`: shared query replay and explicit
  scenario-local Main dispatch; `scenario/history_conflicts.py`,
  `handoff_conflicts.py` and `prior_observation.py`: public
  historical evidence alignment.
- `trajectory_builder.py`: prefix replay, Main's recovery decisions,
  counterfactual continuation and causal acceptance checks.
- `state_machine.py`: versioned Attempts, observable issues,
  two-sided dispute handling, invalidation and recomputation.
- `runtime/`: executable faults, legal evidence scheduling;
  `reporting/`: actual Sub-Agent reports and immutable result
  contract; `orchestration/`: O1 Main decision contracts and
  reasoning review; `inspection/`: public claim extraction and
  generation/privacy guard; `rewards/`: private per-turn reward.

There are no shadow report-only agents, no alternate state machine,
and no compatibility wrappers for deleted top-level module paths.

## Data and audit interpretation

`episodes.jsonl` and `main_turns.jsonl` contain only globally
accepted, fully reviewed O1-compatible positives. A private
`stage4a/journal` retains every attempted execution (including
incorrect and unresolved); `stage4b/journal` contains re-authored
actually executed reports; `journal` records final Main decisions.

Useful private outputs:

- `private/candidate_selection.jsonl`: one row per candidate,
  `scenario_blueprint`, actual replay outcome, historical gaps,
  `delayed_causal_audit`, `behavior_signature` (case-rich) and
  `policy_signature` (abstract operations).
- `private/paired_trajectories.jsonl`: clean-reference identity and
  private unrepaired continuation; it is **not** shown to Main.
- `private/claim_state_supervision.jsonl` and
  `private/reviewed_observable_states.jsonl`: next-round public
  Claim–Source–Dependency belief/risk labels. Negative or unresolved
  cases are **not** silently exported as positive action labels.
- `private/decision_reward_states.jsonl`: pre-action executable
  tasks, evidence needs, dependency/parallel legality and completion
  eligibility for deterministic reward; never a Main prompt field.

Key `summary.json` checks:
`candidate_delayed_propagation_trials`,
`execution_recovered_delayed_propagation`,
`selected_recovered_delayed_propagation`,
`published_delayed_propagation_recoveries`,
`actual_downstream_invalidations`,
`published_with_downstream_invalidation`,
`published_historical_conflict_round_gap_distribution`,
`distinct_selected_case_signatures` and
`distinct_selected_policy_signatures`.

The `template` teacher is a deterministic causal reference, **not**
an online trained Main-Agent policy result. For language-based SFT,
4B/4C Author+Critic must accept report and decision prose. Failure
and fallback remain private diagnostics; they cannot be promoted
through a successful deterministic outcome.

## Implementation checks

Run the environment unit tests and small paired Stage 2–4 smoke
pilot before any full-scale run. In particular, exercise: no phantom
dispatch; no early clean-answer reuse; same-source discrepancies not
counted as independent votes; bounded two-sided retries; actual
invalidated versions; non-recomputed consumers failing publication;
unchanged Stage 1–3 contracts; and no future/private data in O1 inputs.

## Public conflict dependency and reward consistency

Stage 4 pauses downstream consumers of both publicly conflicting reports,
including consumers of an original query when its recheck is a DAG leaf.
Independent evidence tasks remain eligible for parallel dispatch. A repair is
scorable only when current delivered reports contain the discrepancy;
stale private diagnostic flags cannot justify it.

Strict reward replay restores an earlier keyed-handoff contract from its
recorded task relationship and checks the actually delivered claims again.
Replay removes later history-conflict flags from logged attempts.
Two-sided rechecks retain the original pair of conflicting tasks.

Current assignments name their calculation inputs and operations. Stage 4C
checks high-confidence mismatches between reasoning and those instructions,
including date-pair versus observation-date extrema and source-free work
described as new external acquisition. Report prose normalization preserves
all executor-owned Result values and citations.

Candidate audits distinguish answer_changing, completion_blocked and
consistency_only using the private unrepaired continuation. Under a global
language budget, consistency-only repairs follow answer-changing or blocking
cases within the same fault kind; they remain valid consistency exercises.
Computation variants also include swapped ratio operands, ratio/percent scale
confusion and sum/difference confusion when an executable node supports them.

Use a new output directory for regeneration; older accepted trajectories are
not silently reclassified under the updated scheduling policy. This change
is contained in Stage 4 and consumes frozen Stage 1-3 artifacts.

The report Critic receives actual execution input bindings and arithmetic
operands separately from the projected Result. A projected single identifier
does not erase a witnessed wrong association. A bounded second review may
reconsider that specific objection; rejection still produces fallback and
cannot publish an unreviewed positive.

Stage-4C retains rejected reasoning and guard reasons in private
reasoning_guard_drafts. Revision feedback explains the applicable field or
action constraint. Relationship identifiers are intermediate lookup keys;
an observation-ID/year request does not request coordinates or full dates.

For Main review, confirmed_repeated_query_conflicts compares the current
delivered Results of an identical delegated query with identical publicly
tracked prerequisite input versions. Different query bodies, changed parent
versions, unknown inputs or an unparseable Result cannot create this witness.
Only current received values and report round numbers enter the review
contract. Neither a reporter's done/corrected wording nor a completed
downstream calculation settles a still-disputed input.
