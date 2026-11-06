# Stage 3 — Clean Sub-Agent Reports → Main Scheduling SFT

Stage 3 consists of **two explicitly separate, sequential phases**. It does
not require or train on an entire autonomously executed task episode.

The frozen Stage-1 ProgramIR / Fragment DAG and the deterministic
Controlled Renderer question supply the immutable semantics; the Stage-2
natural-language rewrite is a distinct *surface* requiring independent
certification before it may be shown to Main. Sub-Agent answers come from public
source-replayed symbolic execution; the LLM writes only language.

### Answer-blind simulated Web Search (newly constructed tasks)

New `web_search` and `hybrid` tasks opt into
`task_local_ranked_public_index_v1`. Stage 3 builds a **teacher-private
search catalog** from ALL publicly rendered task sources, including
distractor pages. Only the public Main question, its assigned task instruction,
and snippets returned by earlier simulated searches may form queries.
The ranker never accepts a Gold answer or Gold source list.

`GoogleSearchAction` is simulated as query → deterministically ranked hits
(title, resource handle, snippet) with pagination and bounded query expansion.
The matching resources can then be inspected with the assigned O1 tool
(`ExtractUrlContentAction`, `VideoAnalysisAction`, etc). Every Gold-required
discoverable source must actually appear among independently retrieved hits
before public Source replay can certify the subtask. Otherwise the world is
quarantined with `public_search_witness_unreachable` rather than silently
inserting the answer-bearing source. A Source from an actual initial attachment
is inspected without a fabricated search.

The Sub-Agent bank retains *executed simulated* queries/hits in
`simulated_tool_provenance.search_calls` **only when Main assigned
`GoogleSearchAction` and the Sub-Agent actually needs newly discoverable
evidence**. The independent preflight retrieval check is **teacher-only**,
not a Sub-Agent call. With no assigned search tool, the bank now has no
search execution record; a task that nevertheless requires an unknown,
discoverable Source fails rather than receiving an unauthorized search.
Main prompt/history never receives the index or future search hits. The
inspection set remains teacher-guided by the frozen proof AFTER retrieval;
this is not an autonomous search-policy benchmark. It tests whether a
public-only search *could locate* the necessary evidence, not whether an
untrained agent would choose the same queries or click decisions. The corpus
is local to each synthetic world and uses synthetic Wiki/YAGO aliases, not
real online pages.

Old frozen tasks lack the new retrieval policy and remain on the legacy
replay pathway for reproducibility. To exercise web-first simulation, rebuild
Task Construction/freeze and Stage 2 questions in a **new output workspace**,
then rerun Stage 3. Merely rerunning a Stage-3 command on an existing v6 Bank
does not retrofit historical search actions.

### Candidate discovery is not an initially supplied roster

An unbound `FIND_ALL` or `FIND`-by-type operation starts with **unknown
candidate names**. The canonical Main delegation asks the Sub-Agent to find
entities in the task-related evidence first, then apply the frozen filters.
The exact root result is reconstructed from **published source claims and
entity types**, not injected into the Main prompt as an oracle list.
Sub-Agent `summary` / `trace_summary` may name the candidates AFTER
describing how the evidence identified them; they must not falsely claim
that a complete candidate list was supplied at the outset when other
sources had to be discovered. In contrast, a later-round Main delegation
may explicitly name entities in earlier, already published Sub-Agent results.

This is source-backed *simulation*. Public aliases (e.g. synthetic city
names) deliberately differ from original YAGO entity labels, and the
`simulated_tool_provenance` log records `live_tool_calls=false`.
A real public web search may not resolve these aliases. Do not claim this
pipeline measures successful real-web search, or that the task corpus is
the whole Internet. A truly live-search benchmark would require an
independently retrievable evidence index or resolvable original names.

The new canonical instruction text is supported by the public replay
parser as the same `FIND_ALL` / `FIND` operation.
Stage-3 Bank v2 intentionally does **not** load old v1 files. Rebuild the bank
from the frozen Stage-1 and Stage-2 inputs before running new Main rounds.
This protects against mixing a Stage-2 surface question with a canonical-only
compilation from an older bank.

## Phase 1 — Complete and verify every Sub-Agent subtask

Run **once** after constructing the frozen Worlds and accepted questions:

```bash
python environment/simulate_trajectories.py subtasks \
  --scene omnigaia \
  --provider openai-compatible \
  --schedule compact \
  --max-records 20 \
  --feedback-workers 4 \
  --output work/subtask_bank_20.jsonl
```

Inputs: frozen Stage-1 Worlds and Stage-2 accepted questions. The defaults
come from `environment/config.yaml`; override with `--worlds` and `--realized`.

**Dual-question boundary (v2).** Stage 3.1 compiles and replays **only**
the Controlled Renderer question reconstructed from the Stage-1 World.
Stage 2's `realized.question` remains the separate Surface Question. Canonical
and Surface have independent eligibility hashes. Canonical subtask language,
hidden-identity clue handling, and simulated search queries never derive from
the Surface rewrite. Stage 3.2 may use Surface as the Main prompt only if its
semantic eligibility and independent public-source reachability are certified.
If Surface fails, the *completed canonical bank* is retained only inside the
teacher-private `.quarantine.jsonl` diagnostic under `audit.canonical_bank`;
it is NOT emitted as an accepted bank or training example. A surface cannot
silently fall back to canonical inside Main SFT. The summary reports
`canonical_banks_generated` separately from `accepted_banks`.

For each question, this phase:

1. Compiles and preflights the complete Gold Fragment DAG against **public**
   Source evidence. Hidden private entities cannot satisfy a public lookup.
2. Certifies the Canonical Renderer directly from the frozen World and
   ProgramIR and validates the **Stage-2-issued Surface certificate** against
   the exact question, canonical renderer and ProgramIR hashes. All expensive
   full-graph LLM semantic review happens upstream in Stage 2, not here.
   A changed/uncertified Surface is quarantined; no Stage-3.1 question-review
   model is instantiated. An exact Canonical question is independently safe
   through its local renderer certificate, even without an older Stage-2
   publication record. This is a fail-closed data-integrity check, not a
   mathematical proof that every approved paraphrase is equivalent.
3. Visits **all** fragments in topological waves. For each subtask, records
   the canonical task, source-replay-verified exact answer, `summary`,
   `trace_summary`, observations, dependencies, and report-language audit.
4. Reconstructs an auditable **simulated O1 tool-observation sequence**
   from the public Source acquisition waves and assigned native tool profile.
   A discoverable Source can yield a `GoogleSearchAction` discovery event
   followed by an `ExtractUrlContentAction`, `ImageAnalysisAction`,
   `VideoAnalysisAction` or `ParseAudioAction` inspection event, as
   supported by its modality; previously known Sources are reused without
   creating a new tool call. No fabricated search queries, timestamps, URLs,
   failures or unobserved facts are added.
5. Pairs each permitted tool/source event with that source's EXACT
   observed claims in a grouped `source_evidence_guide`. This reduces
   cross-source attribution errors when the LLM writes its report.
6. Attempts fluent, evidence-grounded Sub-Agent report wording: concise
   `summary` plus real-O1-style `trace_summary` that naturally describes
   what was examined, observed, verified and returned. The writer receives
   the canonical report and the same `execution_reference` as the reviewer:
   explicit step meanings, exact inputs/outputs, completed dependency
   references, and relevant facts bound to their subjects and source titles.
   Relation steps distinguish starting entities from discovered entities;
   ranking steps explicitly state minimum/maximum and ties. A safe result
   statement and facts about the final entities prevent transferring earlier
   entities' attributes into the summary. Multiple sources supporting the
   exact same relevant fact are grouped as `corroborated_facts`; the writer
   must describe their agreement when present. Format-repair prompts retain
   this execution reference. Full class names
   such as `VideoAnalysisAction` are **optional**, never mandatory. A trace
   may omit tool mechanics if it accurately explains the supported result.
   Both literal and natural-language claims of *unpermitted* tool actions
   remain forbidden. Actual completed dependency labels (`Attempt N,
   subtask M`) are provenance metadata rather than observed numeric facts.
   Both naturalization and publication use the same dependency references;
   unknown labels, crossed attempt/subtask pairs, and unsupported factual
   numbers are still rejected. The output contract has **exactly two nonempty JSON
   strings**, `summary` and `trace_summary`; trace bullets belong inside
   the latter string, separated by escaped newlines. For local vLLM, Stage 3.1
   requests a JSON Schema response for those two fields and falls back to
   `json_object` only if the endpoint explicitly rejects schema support.
   A valid nonempty `trace_summary: list[str]` is losslessly joined into
   one string before the SAME deterministic and semantic checks, avoiding
   unnecessary re-generation. `natural_feedback_normalized` counts
   successfully checked writer responses that required such conversion.
   The reviewer distinguishes verified FIND/FILTER/INTERSECT/SELECT_BETWEEN
   and backward relation results from unsupported direct Source claims;
   facts and relation directions must still be correctly attributed.
   Internal observation keys and invented quoted search queries are rejected.
   Numeric guards exempt `Attempt N` / `subtask N` labels only when they
   occur in the assigned subtask's completed dependency references. Other
   unsupported numeric claims still fail the same checks.
   If the local LLM returns missing/blank
   `summary` or `trace_summary`, Stage 3.1 retries the writer **once** using
   a shorter prompt with predicate-relevant source facts and the same fixed
   answer. The retry is not a bypass: both deterministic fact checks and the
   separate semantic reviewer still must pass. Persistent missing fields
   are recorded as `summary=blank/null/...;trace_summary=...` in the
   private bank's `feedback_fallback_reason` instead of losing the cause.
   If the grounded-feedback reviewer returns concrete `issues`, Stage 3.1
   performs **at most one critic-guided content repair**. The Writer sees its
   previous report, reviewer objections, and the original complete fixed
   execution/source/tool evidence. Objections are fallible diagnostics, NOT
   additional facts: the Writer must check them against the frozen evidence
   rather than blindly adopting them. The revised report must independently
   pass the identical deterministic evidence/tool guards and a fresh full
   semantic review. Repair requests/successes/failures are counted separately
   as `natural_feedback_content_repair{,_success,_failed}`. A rejected repair
   falls back to the correct canonical report; the answer is never changed.
   Deterministic hallucinations and unauthorized actions are still hard
   failures and do not trigger an unreviewed rewrite.
   Stage 3.1 also checks explicit public fact assertions in natural
   `summary`/`trace_summary`: unambiguous `X is a R Y`, `X as a R Y`,
   and `X --R--> Y` statements must match a scoped public source's exact
   subject/predicate/object orientation. Likewise, direct `X has attribute
   of number/date` and `X's attribute is number/date` claims must match
   the owning entity's public evidence (including recorded units).
   Ambiguous paraphrases are left to semantic review rather than being
   guessed by regex; this guard proves **some incorrect claims**, not the
   correctness of all other prose. Both initial and repaired model reports
   pass these deterministic checks.
   A parseable draft failing a precise relation/attribute check, omitting a
   required final entity, or leaking an internal evidence field can receive
   the SAME at-most-once evidence-grounded repair, followed by deterministic
   checks and a fresh Feedback Reviewer. The private counter
   `natural_feedback_deterministic_repair` distinguishes this path from
   `feedback_reviewer`-triggered repair; unrepaired reports still fall back
   to the verified canonical wording. Reviewer objections and internal
   field names are never additional source facts.

   v7 hardening: exact English count words (e.g. `three` for result 3) are
   accepted as equivalent numeric answers, while intermediate collection
   sizes such as "11 entities" are permitted ONLY when the verified public
   execution has a collection of exactly that cardinality. This does not
   authorize arbitrary new numerical facts. Unknown Reference titles,
   out-of-scope source IDs, and unsupported numerical/source assertions
   receive at most one full-evidence repair; both validators run again.
   The repair Writer sees the complete list of authorized source titles.
   Writer/Reviewer also receive explicit per-step `entity_scope` labels
   distinguishing the owners of filtered attributes from entities discovered
   by a subsequent RELATE.
   Production Feedback Reviewer JSON now uses a bounded `grounded` boolean
   plus at most two `issues` of at most 180 characters; malformed long
   reviews are retried once or fail closed.

   Full graph-aware Question Eligibility (including contradictory verdict
   repair and focused hop checks) now belongs to the **Stage-2 final
   publication gate**. Stage 3.1 consumes its versioned certificate and
   does not repeat these model requests. Rebuild Stage-2 question files
   against the same frozen inputs before running a fresh v2 Subtask Bank;
   old, uncertified Surface rewrites will be quarantined at the certificate
   boundary.

   A local vLLM HTTP 400 for overlong context is **not** a repairable content
   error: the server's actual `--max-model-len` controls total prompt plus
   reserved output tokens. The separate shared registry may describe a larger
   context than an already-running reused service; inspect/restart the vLLM
   process rather than increasing client `--max-tokens` (which reserves
   MORE output budget).

   Private bank entry `simulated_tool_provenance` explicitly marks
   `live_tool_calls=false` for audit.
7. Only after all parallel siblings pass does the next wave see their public
   results. Siblings never inspect each other's unreleased outputs.

The output `subtask_bank_20.jsonl` is a **complete private teacher bank**:
one independently validated record per accepted question. It includes future
Gold answers for the controller and must **never** be passed to the Main
model as prompt text. It has schema ID `maestro.clean_subtasks.v2` and a
content-integrity hash. Inspect it and the accompanying rejected/quarantine
JSONL files before starting Phase 2.

**Phase 1 does not generate Main natural-language delegations, scheduling SFT
targets, or any simulated Sub-Agent errors.** It never calls the Main writer;
the planner role is used only for question-equivalence review.

## Phase 2 — Compose causal Main states and gold-correct delegations

After Phase 1 has completed, run Phase 2 **using the bank only**:

```bash
python environment/simulate_trajectories.py rounds \
  --provider openai-compatible \
  --bank work/subtask_bank_20.jsonl \
  --round-indices 2,3 \
  --output work/main_rounds_20.jsonl
```

Phase 2 does **not** read the original World / question JSONL, recompile the
ProgramIR, or call the Sub-Agent writer. For every bank:

1. Rechecks the v2 bank's hash, both canonical and surface semantic
   certificates, surface search reachability, completed canonical public replay
   and frozen Gold answer. Old v1, corrupt, or unverified banks are rejected.
2. Traverses DAG waves. A Main state contains **only the Stage-2-certified
   Surface Question**, not the Canonical compilation text,
   and only previously completed, verified Sub-Agent results and evidence;
   the current and future waves remain invisible.
3. Uses the stored canonical Gold delegation as immutable teacher guidance.
   The Main LLM may rewrite each task into a natural O1 instruction, subject
   to strict semantic review and deterministic public-source replay.
4. Emits one **Main** SFT sample per selected decision round: the real
   `OmniGAIAMainAgentPrompt.build_prompt` input and the correct O1
   `delegate_task` JSON action, or a fixed-answer `complete` target.
5. Writes each accepted round immediately. A failed Round 1 natural rewrite
   does not prevent a correct Round 2 state, because the previously verified
   teacher action and Sub-Agent report still reconstruct a valid history.

A Main rewrite that does not pass review falls back to the public-executable
canonical action for **history construction**, but strict natural Main SFT
omits that current action by default. `--include-fallback-rounds` includes these
template-style targets only when explicitly requested.

`--round-indices 2,3` selects the target decision rounds; unselected earlier
waves still contribute their correct feedback. Omit it to include every
delegation wave and the final `complete` decision. `--exclude-complete` excludes the
final action.

The output `main_rounds_20.jsonl` is **already in the
`train_maestro_8b` Stage-1 SFT schema**. Each row has:
`input`, `target_json`, `macro_label` (`advance` or `complete`),
`round_index`, validity fields, and `history_origin`. Do **not** export
it again from a full-episode trajectory. Internal public replay checks are
removed from natural Main training targets only **after verification**.

## Correctness boundaries and current scope

- **Gold Answer**: exact symbolic value, supported by public Source evidence.
- **Subtask feedback**: always completed correctly. The natural-language
  summary is attempted, reviewed and may fall back to an accurate template.
- **Main scheduling**: exactly the Gold-compatible ready subtask set per wave;
  the LLM writes phrasing, never changes the task graph, tools, operands or
  return values.
- **Causality**: completed predecessor waves are Main-visible; parallel
  siblings and future answers are not.
- **No errors/attacks yet**: do not generate fabricated retries, switches,
  rollbacks or unsupported `partial` reports. Clean-data macro labels
  are only `advance` / `complete`.
- **No actual live tool traces**: the private simulated tool-observation
  sequence represents only what is supported by frozen Source witnesses and
  native tool assignments. Report prose may use realistic O1 action/observation
  style, but do not interpret its action counts as real browser, media-model
  or Python tool executions.
- **Scheduling policy**: the current implementation follows deterministic
  ready-wave topological ordering, not all possible linear extensions of
  the DAG. Alternative feasible schedules can be added independently.
- **Validation**: unit tests verify source replay, O1 prompt/tool contracts,
  per-wave visibility, independent phase execution, bank tampering
  rejection and persistence of successful earlier Main rounds.

For a local smoke check without real LLM calls, use `--provider passthrough`
only for low-level fixture tests. It cannot supply independent question
semantic review, so it should not be treated as training-ready production
data.

## Training

```bash
accelerate launch -m train_maestro_8b.train_stage1 \
  --model /path/to/Qwen3-8B \
  --train_file work/main_rounds_20.jsonl \
  --output_dir workspace/checkpoints/maestro-clean-sft \
  --gradient_checkpointing
```

Split train and test by task/World **before** sampling rounds, to prevent
sibling decision states from the same source World leaking across splits.

## Tests

```bash
python -m unittest discover -s environment/tests -v
```

GitHub Actions runs this suite and compiles the environment on every relevant PR.
