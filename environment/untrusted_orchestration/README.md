# Stage 4: evidence-centric offline orchestration (4A → 4B → 4C)

Run from the MAESTRO repository root:

```bash
python -m environment generate-untrusted-turns \
  --config environment/configs/untrusted_orchestration.yaml \
  --mode llm \
  --output-dir environment/data/maestro_untrusted_rounds_v9
```

The runner uses the frozen Scene/World/Macro DAG, reviewed Stage-2 reports,
and accepted Stage-3 base episodes as inputs. **The primary sample unit is
one existing (Question, Main decision round)**, not a random number of
branches per Question. For each accepted Stage-3 `delegate_task` batch that
returned Sub-Agent feedback in round `t-1`, Stage 4A extracts its next Main
decision point `t`; the clean feedback history before `t-1` is preserved.
In `scenario_mode: round_fork` (the exact V8 compatibility case), one
preflight-executable missing/partial/wrong fault is injected into the
just-returned batch, and the continuation is re-executed. In the default
`scenario_mode: event_driven`, that initial injection is represented as
the earliest AtomicFault of a causal event scenario. Optionally another
atomic fault can be scheduled when a future task actually executes, or
an independently executable evidence task can be held until a specified
already-scheduled task returns. **No historical Main action is rewritten.**
The Main only sees actual returned feedback in chronological order.

**All available rounds are indexed.** If no configured fault can execute
at a given round, it remains visible as a skipped round; a future round
does not silently replace it. The default YAML has `max_scenes: null`
and `branches_per_scene: null` to cover all accepted Questions and
feedback rounds. `branches_per_scene` is retained only as an optional
chronological debugging cap. A `max_rounds` recovery budget below a
particular Main decision round is recorded as a skipped round rather
than quietly discarding it.

## Unified event engine (V9)

`round_fork` is now a **single-atom event scenario**, not a separate
simulation implementation. The default config selects `event_driven`;
set `scenario_mode: round_fork` to replay V8 sampling semantics. The
original public O1 Action and compact `main_turns.jsonl` interface are
unchanged. Stage 1/2/3 inputs are reused without regenerating Questions.

**Implemented atomic operators (real execution or delivered feedback):**

| Kind | Surface | Effect |
|---|---|---|
| `no_result` | acquisition | Actual evidence operation supplies no usable WorkProduct |
| `incomplete_retrieval` | acquisition/delivery | Field omitted or returned row set truncated |
| `wrong_retrieval` | processing | A real alternate source row replaces one executed primitive output |
| `wrong_entity_binding` | processing | Actual Join uses a real but nonmatching candidate |
| `report_missing` | report | Tool completed but Sub-Agent delivered no Finish Result |
| `result_summary_conflict` | report | Report Result and Summary differ on an actual field |
| `fabricated_claim` | report | Report claims an alternate real-source value that the executed step did not select |

For the last three, the *actual executor WorkProduct* and the *Main-visible
report* are stored separately. In particular, descendants consume the
**communicated** value of a fabricated report, never a hidden correct
executor value. Such a claim without a visible resolution must not be
labelled as successful recovery. All distractors come from existing
authorized observations; unsupported combinations are rejected during
preflight. Controlled faulty reports are separately marked in the
PRIVATE review audit, rather than mislabeled as LLM-authored.

An event `hold_independent_evidence` can delay a task that is **not
yet dispatched**, when it is causally independent of the anchor task
and there is a shared downstream consumer. The event must be released
by actual executor progress. A delayed-recovery branch may be published
as positive reviewed turns only when the replay actually produces a
Main-visible late conflict AND an actual downstream invalidation; a scheduled
delay alone is insufficient. This is *not* the same as a completed
Sub-Agent message arriving late over an asynchronous transport. The
latter (`message_delayed`) remains a **declared but unsupported**
transport capability pending a faithful O1 late-result adapter; the
generator refuses to simulate it rather than forging an extra Main
turn or retroactively modifying history.

The optional `max_fault_atoms: 2` schedules a second preflight-eligible
fault on a future task's **actual first attempt**. If the continuation
never executes that task, the event is logged as unrealized and cannot
be called a fully recovered composite case. Atomic kinds and delays
are selected reproducibly with a seeded, constraint-guided generator.

**Late conflict:** When a newly delivered independent report disagrees
on a typed join key with a previously delivered report *already consumed
by a real downstream attempt*, the event-driven state machine can
request a targeted recheck of the earlier **used** binding. It does
not consult hidden Gold to decide which report is wrong. On an actual
upstream replacement, `used_parent_versions` and `invalidations`
record stale descendants and selective recomputation.

Output diagnostics include `private/event_timeline.jsonl` and the
private event plans in `fault_alignment.jsonl`, along with summary
counters `actual_downstream_invalidations`,
`visible_late_conflicts_detected`, `atomic_fault_outcomes`,
`verified_without_downstream_invalidation`, and
`scenario_blueprints`. When a recheck reconfirms the same communicated
relation, previously executed downstream tasks are preserved and only
a private revalidation is recorded. These metrics describe **actual**
observed execution,
not guaranteed success: not every existing Scene admits a delayed
causal repair.

**Publication safety:** `private/publication_guard.jsonl` records
branches excluded from reviewed turns for generation-process leakage
such as `The critic correctly noted ...`. Failed and undetected
counterfactuals remain in private audit/diagnostic Episodes, not
positive reviewed turns. All published Main actions still require
correct executable outcome and Stage 4B/4C language review.

## Three real sequential stages

**4A — Evidence-Centric Global Simulation (no LLM).** Construct a complete
counterfactual execution using actual source-/operator-level results, versioned
WorkProducts, and the global dependency graph. At *each round* inspect
Main-visible missing/partial/contradictory evidence, determine required fields,
and emit one **evidence request per needed field**. Requests contain the
requested field, relevant previously visible identifiers, authorized source
scope, a Main-visible reason, and acceptance criteria. Requests are bundled
into executable Sub-Agent assignments when they share the same valid primitive
execution contract. Task instructions are dynamically narrowed for missing
retrieval, partial field completion, stale dependency recomputation, or
observable conflicts. Internally each executable dispatch is bound to a
frozen Macro operator chain; this is an **execution adapter**, not the
Main-Agent's scheduling vocabulary. No new ungrounded independent source,
operator or alternative route is fabricated. Completion is evaluated
privately against frozen Gold but **Gold cannot trigger a VERIFY label**.

**4B — Sub-Agent Feedback Generation (LLM only for Reports).** After *all*
4A round samples have been simulated, naturalize the faulty or newly
executed Sub-Agent Attempt traces
from the private ledger. A report sees only its own observed rows, source
atoms, assigned instruction and execution outcome, not a future attempt or
clean Gold. Previously accepted clean prefix reports are preserved. Report
Author/Guard/Critic can revise a failed report without calling Main; fallback
reports remain explicitly labeled and excluded from positive reviewed-turn.

**4C — Main Action Generation (LLM only for Main).** After *all* 4B
round samples have been processed, rebuild cumulative O1 `subtask_history` strictly from
the FINAL reviewed/fallback Sub-Agent reports. Assignments retain 4A's legal
action count, source permissions, primitive tool routing and evidence goals,
but refresh parent context from the newly finalized public feedback. A
factual visibility guard rejects labels if required identifiers or
contradictory observations were not actually shown to Main. Main Author/Critic
generate each round's natural O1 delegate/complete action.

## V8: evidence-plan validation, reasoning review and progress semantics

- **4A:** Produce specific requests for the exact missing field or remaining
  record set. For a partial row-set return, the assignment asks for a
  complete applicable set rather than guessing the identity of an unseen
  record. Identifiers used for targeted repair come only from Main-visible
  previous reports. Each private dispatch records `focus_fields` separately
  from the entire underlying Macro output schema.
- **4C contract validator:** Before any Main Author call, check that every
  evidence request is covered exactly once, all requested fields are named,
  source/model/tool/typed output schema and target focus remain unchanged,
  and the resulting O1 action is structurally legal. This is independent
  of LLM judgment; invalid plans are rejected, not polished into
  something else.
- **4C reasoning-only critic:** The LLM Author writes natural Main reasoning
  but cannot rewrite executable task instructions or contexts. Critic now
  receives the Main-visible history, the requested fields and the
  candidate **reasoning only**. It does not judge tool ordering or demand
  a different final answer/task from the frozen plan.
- **Progress accounting:** `Main orchestration` keeps the entire
  `planned` number as its denominator. `processed` counts actually
  reviewed Main Actions; `skipped_downstream` counts planned future
  actions not generated after a branch stops; `written` counts Action
  reviews durably stored in the per-round journals. Accounted-for
  progress reaches 100% without pretending each action was authored.
- **Wrong-fault diagnostics:** `private/wrong_visibility.jsonl` categorizes
  whether an *explicit* visible conflict verification was planned at the
  injection decision, later, or never. This is a measure of planner
  observability, **not proof** that an unverified error is intrinsically
  undetectable.
- **Reviewed output terminology:** Console output uses
  `reviewed branches` and `reviewed turns`. The summary includes these
  fields, retaining the historical `recovered` and `main_training_turns`
  keys for compatibility. Public `main_turns.jsonl` retains its O1
  input/output format and strict acceptance policy.

## V6 evidence-quality hardening

**4C evidence-constrained Main generation.** The Main Author can now
naturalize the reasoning of a delegate action, but the O1
`task_instruction`, `context`, model and tool routing are compiled from
the immutable 4A evidence assignment (with prerequisite context refreshed
from FINAL 4B reports). A fluent draft cannot change an executable request
for `marker_code` into a demand for a final winner/score that the source
program does not produce. The result is marked
`llm_reviewed_contract_frozen` with `assignment_contract_frozen=true`
to distinguish authored reasoning from program-owned task text. Final
Action structure is checked again before publication.

**4B observed versus delivered evidence.** Numeric identifiers from
*actually executed* intermediate filter/lookup steps are legal provenance
context in a report's `result` prose. They must not be substituted as the
final delivered value of an output field. An explicit field-value Guard
compares final field assertions with the complete actual delivered relation;
the intermediate trace and erroneous returned WorkProduct remain immutable.
A wrong result must be described as an actual *return*, not silently corrected
to a plausible clean value. Critic instructions reinforce this distinction.
When the Critic specifically confuses an intermediate observation with
final delivery, the same independent Critic is given one narrow second
review with the unchanged actual WorkProduct and trace. This does not
automatically override failure: an unsupported or inconsistent narrative
remains rejected if the second review also fails.
Author and Critic must still independently pass: there is no blanket
acceptance of inconsistent or unsupported results.

**Local round quality (private diagnostics only).** The strict public
`main_turns.jsonl` remains restricted to globally recovered branches with
accepted language. Independently, the publisher checks whether each earlier
Main decision has accepted visible prior feedback, reviewed Main language and
a legal dependency batch. The resulting private reports at
`private/main_turns_local_review.jsonl` and
`private/main_turns_local_qualified.jsonl` allow analysis of otherwise
discarded valid *local* decisions. These are NOT automatically positive reviewed-turn:
a locally sensible decision does not prove the terminal answer correct.
`summary.json` now separates execution incorrect/unresolved from
language/action publication rejection and records the local diagnostic count.
Splits should be grouped by Question ID to prevent round-level leakage.

Stages are intentionally separated rather than interleaved. The CLI prints
three sequential progress bars: **Evidence simulation** (rounds),
**Sub-Agent feedback** (reports), and **Main orchestration** (actions). Each stage is journaled with an atomic file per round sample:

- `round_index.jsonl`: all extracted (Question, decision-round) identities,
  received Sub-Agent batch and eligibility/skip reason
- `stage4a/journal/`: PRIVATE simulated evidence requirements, assignments,
  and complete attempt ledger
- `stage4b/journal/`: evidence-grounded reviewed Sub-Agent reports
- `journal/`: reviewed Main actions and final per-branch outcome

With unchanged input/code signatures, rerunning the command resumes
incomplete stages; changing any relevant code requires a new output folder.

**Important current limitation:** evidence *planning* is field-level, and a
single assignment can satisfy multiple requests, but execution still uses
frozen Macro operator programs to guarantee correct provenance. The system
does not yet dynamically compile completely new primitive programs or create
new authorized sources. Independent Verify/Switch actions require a
supported evidence route in the frozen World; otherwise the branch remains
unresolved. This is an offline dataset simulator, not an online autonomous
Main-Agent rollout.

## Current functionality

- `faults.py`: Deterministic `missing`, `partial` or `wrong` outcomes, with a **per-attempt observed operator trace** built from the actual post-injection execution. The missing mode stops at the failed operation; partial is currently a final-delivery truncation (not a mid-chain partial source read); wrong changes a primitive Relation then recomputes downstream primitives.
  The `wrong` mode uses actual alternative source observations or
  mismatched **real** parent candidates at a join; unsupported cases are
  skipped instead of inventing values. Fault target selection weighs
  downstream dependency impact, with optional multi-target batches.
- `state_machine.py`: Versions individual WorkProducts, derives necessary
  parent fields for readiness, identifies conflicts **only if visible in
  previous Sub-Agent reports**, and marks dependent results stale after
  re-execution. Original reports remain visible in the cumulative O1
  history and are never silently overwritten.
- `trajectory_builder.py`: Constructs new Main `delegate_task` actions
  for targeted rechecks, unfinished evidence and independent ready tasks
  **in the same decision batch**. All tasks are actually re-executed on
  the resulting WorkProducts. Unresolved or incorrect outcomes are
  diagnostic only and never published as positive reviewed-turn labels.
- `report_author.py`: Stage-2-style Sub-Agent Author/Guard/Critic
  consumes the current attempt's actual `observed_operator_steps`, output
  rows and evidence atoms. It shares Stage 2's observed-step representation,
  result normalization, surface normalization and high-precision stage-scope
  checks; it generates natural-language `finish_result.result`, `summary`
  and `trace_summary` **for the actual attempt**. Executor-owned `status`
  is never requested from the model and a legacy model `status` is ignored. The Guard verifies values,
  identifiers, report status, evidence scope and partial delivery against
  the deterministic WorkProduct, not the private clean Gold. For a missing
  deliverable, `finish_result` remains absent. Review failures are
  explicitly labeled and excluded from positive LLM-mode reviewed-turn.
- Main Action Author/Critic reuses the Stage-3 O1 Guard, but with
  recovery-specific prose prompts. The deterministic planner still chooses
  every legal target, tool/model routing and parallel batch; LLM never
  changes scheduling decisions or makes a hidden attack label visible.
- `generate_turns.py`: Exhaustively enumerates accepted Main feedback
  rounds per Question, samples a grounded fault within each just-returned
  batch, then runs the fully sequential 4A/4B/4C workflow with configurable
  worker pools, per-round atomic journals, input/code signatures for exact
  resume and positive-label publication.

The public report Author never receives the private fault label, clean
reference WorkProduct or overall Gold answer. In a wrong-result attempt,
actual observed (possibly wrong) records are authoritative, while the
complete frozen Gold remains private for episode outcome evaluation.

The `summary.json` explicitly separates `requested_feedback_rounds`,
`rounds_with_executable_fault`, `rounds_without_executable_fault`,
`rounds_skipped_by_recovery_budget` and `rounds_per_question`. Its `status`
reports whether round generation completed, while `publication_status`
tracks whether those rounds passed recovery and language review.
The `summary.json` also separates `execution_recovered_branches` and
`execution_outcomes` from `recovered`/positive reviewed-turn publication. It also
records `subagent_executor_result_normalized`,
`subagent_guard_failure_causes` and `subagent_review_failure_stages`.
A branch
can be deterministically recovered but fail LLM language review; it remains
diagnostic and is never silently treated as reviewed training data.
For changed trace/Author code, generate into a new output directory:
existing signed journals cannot be resumed across these changes.

The runner also executes a private **no-repair counterfactual** along the
old Macro DAG using the poisoned WorkProduct, to distinguish an incorrect
upstream key from a correct arithmetic calculation on that key.

## Pilot vs production

Default YAML uses `generation.mode: template` for a repeatable first
pass without local API calls. To naturalize **both** actions and reports,
set `generation.mode: llm`. Two independent settings,
`generation.report_mode` and `generation.main_mode`, accept
`inherit` (use `generation.mode`), `template`, or `llm`. The
`subagent_model` YAML section configures the Stage-2-style report
Author/Critic; `model` configures the Stage-3-style Main Action
Author/Critic. They can use different Qwen models, temperatures and
revision budgets via `report_max_revisions` / `main_max_revisions`.

Example: run only the Sub-Agent Author/Critic first:

```yaml
generation:
  mode: template
  report_mode: llm
  main_mode: template
```

The deterministic fault and decision planner never invoke an LLM.

### Progress display

The CLI displays **two independent live progress bars**:
`Sub-Agent feedback` (each new report Author/Guard/Critic call) and
`Main orchestration` (each new Main Action Author/Guard/Critic call).
Unlike trusted Stage 2 and Stage 3, recovery execution is causally
interleaved: Main decisions can require another report and vice versa.
It would be incorrect to force the two bars into sequential full passes.
Each bar estimates its total work from the original DAG, expands as
dynamic recovery schedules extra calls, and displays the exact total
on completion. Postfixes show accepted/fallback report counts and
accepted/failed Main action counts. A final branch-level line reports
**execution recovered** separately from **fully reviewed reviewed-turn**.

Sub-Agent reports use the **complete delivered relation** for factual
grounding. Source evidence atoms and intermediate operator rows are
explicitly *previews*, not indications that the scan was incomplete.
A `partial` handoff may omit values observed before delivery; it remains
partial even if the internal earlier step found those values. An LLM
`result=null` for a partially delivered WorkProduct can be replaced with
a readable, deterministic statement of exactly that delivered Relation
before undergoing the full Guard and independent Critic; no arbitrary
wrong or fabricated value is silently corrected.

`summary.json` reports independent
`subagent_report_generation_modes`,
`subagent_report_review_statuses`,
`main_action_generation_modes` and
`main_action_review_statuses`. Their per-attempt details remain only in
`private/fault_alignment.jsonl`. An LLM report that fails deterministic
Guard or independent Critic may be retained as a clearly labeled
diagnostic template, but its branch is **not** published as positive LLM
reviewed-turn. This is **not** an online learned Main-Agent rollout. Persistent
`same_path` faults without an authorized recovery route remain unresolved.

The `input` fields in the YAML point at the *existing* accepted trusted
outputs. Output paths must differ from the existing inputs.

Published files:
- `episodes.jsonl`: visible O1-style branched episodes, including
  unresolved outcomes but **no hidden injected-fault labels**.
- `main_turns.jsonl`: only post-fork Main decisions from fully recovered
  Episodes, one record per decision, with compact cumulative
  `input.subtask_history`.
- `private/fault_alignment.jsonl`: target primitive, actual per-attempt
  relations, uncorrected propagation/counterfactual, invalidations,
  reason for recovery or failure, and measured simulation cost.
- `failed.jsonl`, `summary.json`, `manifest.json`, and `journal/`
  for per-branch diagnosis and resumable runs.

The supported model deliberately starts with one injected last-batch event
per independent branch. Later follow-up rounds can remain affected by a
persistent same-path fault. This is **not** a general-purpose adversarial
agent attack library or independent-source retrieval engine yet. The
frozen World is synthetic, and media observations remain simulated.
