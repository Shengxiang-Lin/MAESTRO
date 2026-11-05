# Skeleton-bound O1 trajectory synthesis

This stage realizes the frozen Stage-1 solution as Main/Sub-Agent interactions.
It preserves composition fragment boundaries, adds cuts for independent branches
and shared intermediate results, and dispatches every ready fragment in parallel.
The accepted Stage-2 question is preserved.

## Strict semantic boundary

Gold ProgramIR, public source evidence, dependencies, tool lists, and exact
answers remain authoritative. **Natural language is not the answer oracle.**
The default LLM synthesis mode (`--style-provider natural`) makes two
additional, independently reviewed language realizations:

- Main rewrites each immutable, source-scoped gold delegation into a fluent
  information goal and expected deliverable. One writer request is made per
  assigned subtask, so a missing JSON array entry cannot silently delete a
  parallel sibling. Each task is independently guarded/reviewed; rejected
  rewrites retain the verified canonical task without discarding valid sibling
  rewrites. Main `reasoning` is empty by default, matching the real O1
  delegation traces. Completion uses the exact gold answer without an extra
  language-model request. Tools, dependencies, results and answer remain fixed.
- Each Sub-Agent writes a natural concise finding and evidence explanation
  from its already verified public observations and exact executed steps.
  Fabricated search/tool actions and unsupported facts are forbidden.
- A deterministic guard and a separate semantic/evidence LLM review must
  both pass. If either rejects the rewrite, the original verified wording is
  emitted instead and the cause is saved in `generation_audit`; the sample
  is never silently marked as a successful natural rewrite.
- Natural `task_instruction` is **not** used to compute an answer. A v3
  `[Public task packet]` contains the original publicly executable numbered
  checks, alongside the same evidence handles and completed public bindings.
  `replay_task` and `export_episode` execute these checks solely against
  published evidence, not against hidden Gold values. The public task
  transport and source access remain reproducible. **After** the full episode
  passes replay, the Main SFT exporter removes the v3 machine packet from
  the training target's `params.tasks[].context`. The original episode
  keeps the replayable contract for auditing; canonical fallback (v2) tasks
  remain unchanged and should be analyzed separately for natural coverage.
  `generation_audit` and `metadata.natural_main_tasks_{rewritten,total}`
  quantify naturalization.

The legacy `--style-provider llm` selects one of the four deterministic
`concise`, `stepwise`, `goal`, and `evidence` variants. The
`--style-provider template` option chooses deterministic variants without
LLM style calls. `--provider passthrough` remains the offline structural mode.

These are **gold-guided language realizations**, not autonomous planning
evaluations. Stage 3 does not simulate actual tool traces; public evidence
claims derive from frozen source witnesses.

- Main chooses a language variant for each ordered ready task and its reasoning.
  The public output is `delegate_task` with native `params.tasks`, or `complete`.
- Each current-round Sub-Agent receives a separate LLM request to choose its
  report style. Independent requests run through `subagent_workers` concurrently.
- `summary` states the subtask's result. `trace_summary` narrates inspected/reused
  references and the performed operations in prose or progress bullets.
- Exact intermediate answers populate `finish_result.result`. The final Main
  answer is copied from completed public results and checked against the oracle.
- Projected tasks and reports must exactly match one allowed realization. A
  same-model semantic judge cannot waive the contract or modify the gold logic.

For example a follow-up instruction names the completed `Attempt 1, subtask 1`
result and its exact value. It cannot replace that set with all records. Filters
report their true input and retained output; they do not invent reasons for
discarding particular entities. Joins identify both inputs explicitly.

Latent FIND targets are expressed through their public question clues, not their
hidden entity names. A name becomes visible when a Sub-Agent identifies it.
Private graph IDs, rejected-generation commentary and teacher instructions never
appear in projected public language.

## Evidence and parallelism

Every fragment is evaluated on a reachable, decision-complete evidence witness.
Acquisition is AND over required anchors and OR over their providers. Read order
respects those anchors. Previously read evidence is marked as reused.

**Public entity universe:** Stage-1's private `world.entities` can contain
synthetic distractors with no Source evidence. Public Replay has never seen
those entities; its FIND_ALL and type enumeration operate on entities actually
named in published source claims. Witness compilation now projects the same
source-backed entity domain and evaluates the **entire ProgramIR** against
all frozen public Sources before compiling any fragment. If the final
public-source answer differs from the unchanged private oracle, compilation
fails closed with `public_full_world_answer_disagrees_with_gold`.
When only intermediate candidate sets differ (e.g. invisible distractors
removed before a later relation/filter), fragment witnesses and public
intermediate supervision use the publicly observable set, not a private
candidate that no Sub-Agent could enumerate. Witness coverage,
decision-information checks, and exact final-answer agreement remain strict.
Private audit `public_intermediate_projections` records affected node IDs and
set cardinalities, not student-visible answers.

Uncomposed independent operands of a join are separate tasks rather than being
hidden inside one terminal task. Only the actual join and final calculation wait
for their operands. Non-root shared intermediates have explicit result bindings.
No source-by-source decomposition replaces the original reasoning skeleton.

Main's public state contains the question, provided reference descriptors,
available native tools, an explicit public evidence store, and cumulative O1-formatted subtask history. Simulated
records have public reference handles, not invented image paths or URLs.
Canonical O1 task contexts use a version-2 **public evidence reference packet**:
`known_resource_ids` and completed input bindings, **not full copied source
records**. `Public Evidence Store` in the visible Main state still contains
the actual observed records. The synthesizer and SFT exporter resolve the
references from the chronological public store, then replay the exact same
operation steps. Natural-mode O1 tasks instead carry a version-3 packet with
`execution_checks` so human-facing `task_instruction` may be fluent prose
while replay remains strict. These public checks are NOT a private solution
or evidence oracle. An unknown or future resource reference fails validation.
This greatly reduces JSON repetition in Main's SFT target.

The teacher language-choice prompt retains completed result bindings; narrative
history above 9000 characters uses a marked recent-history view. Saved public
states always preserve the full cumulative history.

## Publicly executable Gold plans (before LLM calls)

`plan` now checks **two different boundaries**:

1. Gold compilation: source-reachable evidence can reproduce the original
   frozen intermediate and final answers.
2. Public executability: each proposed O1 delegation, including its completed
   input references and source observations, can be replayed *without* any
   private oracle/entity injection. Independent same-round subtasks do not
   observe sibling outputs until that parallel wave finishes.

A plan can therefore be `compiled` but have `status: quarantined` with
`public_executability.reason`, `fragment_id`, `round`, `operations` and
`new_source_ids`. These are **diagnostics**; do not promote them to prompts.
Accepted plans include `public_executability.publicly_executable: true`.
Compilation and public executability do **not** certify that the Stage-2
English question has the same meaning as Stage-1 ProgramIR. That requires
separate question eligibility review.

### Concealed intermediate identities

The first-stage public question renderer may intentionally conceal a direct
FIND when a separate, evidenced branch of an INTERSECT uniquely determines
the same singleton. A direct `Identify the hidden entity` delegation would
be impossible for an O1 Sub-Agent. The `public_projection` pass removes
**only this certified redundant intersection** when all of the following
hold: first-stage metadata explicitly says `unique_intersection_clue`,
the hidden FIND is standalone and has no evidence, the independent
identifying branch has public evidence and the same exact answer, the
INTERSECT has no extra evidence, and neither removed fragment has other
consumers. The private symbolic trace and final oracle remain unchanged.
`public_projections` records the transformation for private auditing.

Stage 3 may also certify the same algebraic redundancy when Stage-1
metadata did not tag it: the concealed singleton FIND, independent
source-backed branch, and their INTERSECT must all have the same unique
result in the **entire public-source execution**. The adapter then requires
actual witness support, a standalone evidence-free hidden FIND, matching
join inputs, no unsafe shared consumers, and unaltered downstream bindings.
Uncertified candidates are not rewritten. Private
`public_projection_diagnostics` records whether candidates were projected;
`public_projections` records successful, auditable rewrites.

Other concealed entities are never supplied from the oracle. If a
necessary identifying clue cannot be formulated or replayed, the task is
quarantined with `latent_find_missing_public_clue`, rather than producing
a misleadingly successful synthetic trajectory.

## Scene tool mapping

`tools.py` owns `ToolProfile` and the scene registry. `--scene omnigaia` selects
the native tools used by the actual OmniGAIA implementation:

| Frozen modality | Native inspection tool |
| --- | --- |
| text | ExtractUrlContentAction |
| search | GoogleSearchAction |
| image / image_report | ImageAnalysisAction |
| audio / audio_transcript | ParseAudioAction |
| video / video_report | VideoAnalysisAction |
| table / database | ExecuteCodeAction |

Discoverable evidence additionally uses the scene's search tool. Evidence reuse
without a new read uses the scene's calculation tool. Unknown modalities/profiles
raise an error. Add another `ToolProfile` to the registry to support another
scene; compiler, prompt, public tool registry and feedback share this mapping.

## Commands

Run from the repository root, with the normal configured conda environment:

```bash
# The plan output now reports compiled / ready / quarantined separately.
# Default compact schedule merges linear steps while preserving true forks.
python environment/simulate_trajectories.py plan --scene omnigaia \
  --schedule compact --max-records 10 --output work/trajectory_plans_compact.jsonl

# Diagnose the original unfused fragment graph separately.
python environment/simulate_trajectories.py plan --scene omnigaia \
  --schedule baseline --max-records 10 --output work/trajectory_plans_baseline.jsonl

python environment/simulate_trajectories.py synthesize --scene omnigaia \
  --provider passthrough --schedule compact --max-records 10 --output work/trajectory_offline.jsonl

python environment/simulate_trajectories.py synthesize --scene omnigaia \
  --schedule compact --max-records 10 --output work/trajectory_llm.jsonl

python -m unittest discover -s environment/tests
```

For a paired 100-question check against an unchanged frozen Stage-1 batch,
run `plan` and `synthesize --provider passthrough` with the same
`--worlds`, `--realized`, `--seed`, `--max-records 100`,
`--schedule compact`, and `--scene omnigaia` options. Compare `ready`,
`quarantined`, `rejected`, and the per-task reasons. A structurally ready
passthrough trajectory is **not** yet training eligible; Stage-2 question
semantics must also pass an independent actual review.

Main/executor model profiles, temperatures, retries and current-round worker
concurrency continue to come from `environment/config.yaml`. Different questions
remain sequential. The passthrough provider is a structural reference, not
LLM-generated training data. Synthesis overwrites its selected output paths;
use distinct paths for smoke tests. There is no cross-run resume checkpoint.

**When Gold compilation fails**, the plan JSONL or synthesized rejected JSONL
now contains a `witness_diagnostics` object, with the output node,
source/fact/entity coverage, acquisition reachability, exhaustive-search gate
counts, and full-source executor/information checks. These diagnostics tell
whether failures come from missing public entity coverage, unreachable
resources, missing facts, or the actual proof computation. **No proof gates
are relaxed** and Stage 1 frozen data is never repaired in this stage.
`question_reviews.json` is optional; its absence does not satisfy the
separate semantic review required to export training examples.

## Output and scope

Each episode has native O1 `trajectory[]` actions and `subtask_results` with
`finish_result` and `trace_summary`, followed by `complete`. `gold_supervision`
contains private graph bindings, source witnesses and exact answers: exclude it
from student prompts. Rejected audits are private too.

Metadata explicitly records LLM style selection with verified semantic projection.
The system does not execute or fabricate raw tool calls, costs or step counts.
`language_contract_verified` proves conformance to the frozen skeleton, not that
the Stage-2 natural-language question is equivalent to that skeleton.
`upstream_question_semantics_revalidated` records whether semantic eligibility
was actually checked. Unreviewed offline trajectories cannot be exported as
training data. Incorrectly rewritten questions must be repaired at Stage 2,
not silently replaced at Stage 3.
