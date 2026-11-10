# MAESTRO Environment — Stage 1: Generate Question

This directory contains the **single maintained Question-generation pipeline**
for MAESTRO. The obsolete Task-First, V2 experiment and separate naturalize
commands have been removed; no old command compatibility is maintained.

## Usage

Edit [config.yaml](config.yaml) for input, output, difficulty mix, Author,
Critic, concurrency and revision settings. Then:

~~~bash
python -m envir generate-questions --config envir/config.yaml
~~~

For a read-only audit:

~~~bash
python -m envir quality --config envir/config.yaml
~~~

The implementation and detailed output contract are documented in
[generate_question/README.md](generate_question/README.md).

Stage 2, **Implement Subtask**, now materializes the frozen Macro
execution into per-Subtask `answer` + evidence-grounded `trace_summary`.
It reuses accepted Stage-1 private scenes and does not call a model:

~~~bash
python -m envir implement-subtasks \
  --config envir/implement_subtask/config.yaml
~~~

See [implement_subtask/README.md](implement_subtask/README.md).

A separate Stage-2 language pass rewrites the frozen Oracle responses into
realistic Sub-Agent `finish_result.result/status/summary` messages with
an LLM Author and independent Critic. The typed answer, primitive evidence
steps, and fact refs are never regenerated or edited:

~~~bash
python -m envir naturalize-subtasks \
  --config envir/implement_subtask/config.yaml
~~~

Stage 3, Subtask scheduling and composition, remains separate and is not
implemented here.

Stage 1 remains Generate Question.
Stage 2 (implement Subtask) and Stage 3 (compose/schedule Subtasks) have
distinct future interfaces. They do not belong in this generator.

The accepted public Question bank never contains a Scenario/Question
fallback. Every successful Question is audited against the compiled
source-scoped Macro TaskGraph and terminal answer fields. Failed candidates
are private diagnostics. All media evidence is fictional structured
simulation, not perception of original images/audio/videos.


### Streaming and checkpoint recovery

Scenario Author/Critic and Question Author/Critic now append reviewed
decisions to dedicated accepted/rejected/audit JSONL journals AS THEY
FINISH, in stable input order. The live progress bar shows accepted,
rejected, and written (the committed ordered prefix).

Accepted Questions also appear immediately in the ordinary files
questions/scenes.jsonl and questions/questions.jsonl; rejected
Questions appear in questions/failed.jsonl.

Each stage has a signed, SHA256-verified checkpoint and a writer lock.
Interrupting and rerunning the SAME command and output directory
(with unchanged frozen inputs/model/prompt code) skips completed LLM
decisions. Uncommitted tails are discarded and published files are
reconstructed from the durable journal.

The config key generation.flush_every defaults to 1 (fsync every
decision); the CLI may override this:

~~~bash
python -m envir generate-questions \
  --config envir/config.yaml --flush-every 5
~~~

A different model, input scene bank, generator contract or code must
use a new output directory or explicitly overwrite the old run.
Completed legacy outputs from the old all-at-once implementation have
no checkpoint and cannot be resumed retrospectively.


### Single-command Stage 2 (Oracle + naturalized Sub-Agent reports)

The recommended new-run entrypoint can now execute both Stage-2 passes
without an intermediate manual command:

~~~bash
python -m envir naturalize-subtasks \
  --config envir/implement_subtask/config.yaml \
  --scenes envir/data/maestro_question_fresh_100/questions/scenes.jsonl \
  --output-dir envir/data/maestro_subagent_reports_all_in_one_100
~~~

The CLI creates a PRIVATE deterministic Oracle bank in
<report-output>/oracle/{subtasks.jsonl,scene_index.jsonl,...}, then
generates normal Sub-Agent finish reports in <report-output>/
with the existing Author/Critic and streaming journal. An Oracle
manifest locks the scene input hash, execution implementation, and
teacher output hashes. On interruption, the SAME command/directory
reuses this valid Oracle; the report checkpoint skips already
committed Macro reviews. Incorrect or changed inputs fail closed.

The separate implement-subtasks command is still supported for
Oracle-only inspection. The old naturalize-subtasks command without
--scenes continues to consume pre-materialized Oracle files configured
under reporting.source_subtasks and reporting.source_scene_index.


## Opt-in clue-driven evidence acquisition (clean pilot)

The normal generator defaults to the original static computation patterns.
Use --clue-chain-share to replace a controlled percentage of MEDIUM
scenes with a DIFFERENT information-acquisition dependency:

- Browse a fictional simulated illustration. Two exhibit features
  (emblem color + motif) are needed to distinguish the intended item;
  decoys deliberately share one feature each
- Read that item's marker code; use it as a typed, evidence-backed query
  into a KEYED (non-browseable) catalog index
- Use the accession identifier returned by the index to query a
  second KEYED (non-browseable) archive
- Answer the target's accession year; retain exact fact provenance from
  illustration -> index -> archive

The index and archive IDs/titles are advertised to Main-Agent, but
their records are NOT returned by full read_source or scan. The
lookup_by_key primitive requires an upstream typed WorkProduct and
emits only matching records. The compiler preserves lookup operations
as separate dispatchable Macros, each with a real dependency. The
source observations remain synthetic; there is NO real media decoding.

Pilot with 20 Stage-1 tasks, approx. 10 clue-chain tasks with the
default 20/50/30 difficulty mix (all medium tasks become clue-chain):

~~~bash
python -m envir generate-questions \
  --config envir/config.yaml \
  --target 20 --clue-chain-share 100 \
  --output-dir envir/data/maestro_clue_pilot_20
~~~

Then Stage 2A + Stage 2B via the existing unified one-command entry:

~~~bash
python -m envir naturalize-subtasks \
  --config envir/implement_subtask/config.yaml \
  --scenes envir/data/maestro_clue_pilot_20/questions/scenes.jsonl \
  --output-dir envir/data/maestro_clue_reports_20
~~~

Quality reports include evidence_acquisition keyed_lookup_count,
mean_evidence_acquisition_depth, clue_counterfactual_pass_rate and
unauthorized_full_read_success_rate. The last should be zero for
synthetic keyed stores. Check the private source/answer/evidence data,
and inspect several public questions for scenario variety before
raising the clue share or introducing new retrieval motifs.

The clue-chain pilot intentionally uses only MEDIUM tasks; Easy and Hard
keep their existing programs for now. 100 means 100 percent of MEDIUM,
not all difficulty levels. Defaults remain unchanged when the option is
omitted (generation.clue_chain_share: 0).


## Opt-in typed Evidence Acquisition Graph grammar

The new --evidence-graph-share percentage applies to MEDIUM and HARD
scenes, independently from the earlier clue-chain pilot. It defaults to
zero, so historical generation remains unchanged. When both shares are
enabled, evidence_graph takes precedence on selected medium tasks.

The seeded grammar currently samples four semantically distinct plans:

- chain: 2–3 dependent keyed lookups after an illustrated two-feature clue
- parallel: independent illustration and caption sources join before the
  keyed retrieval
- disambiguation: multiple visually matching candidates require separate
  motif notes to eliminate a false identity before the final archive lookup
- conflict: two inventories are reconciled to find the single disputed
  record before an authoritative keyed retrieval

The sampler varies archetype, identity labels, relation values, decoys
and key-chain depth while respecting typed source contracts. Worlds are
constructed witness-first, and each required source is subjected to a
grounded answer-changing counterfactual. The public question supplies the
initial clue and target result, not the hardcoded sequence of retrievals.

A small dry run (Easy tasks still use their baseline family):

~~~bash
python -m envir generate-questions \
  --config envir/config.yaml \
  --target 100 --evidence-graph-share 100 \
  --output-dir envir/data/maestro_evidence_graph_pilot_100
~~~

The normal naturalize-subtasks --scenes entry can be applied to accepted
scenes from this run. Inspect quality_report.json -> evidence_graph for the
archetype distribution, keyed lookup count, source-counterfactual pass rate,
and independent-wave width. Test your actual local LLM Author/Critic rates
before expanding to large batches.

This is a constrained graph grammar pilot, not unconstrained arbitrary-DAG
search or true multimodal/web perception. Stage 2 reports are deterministic
Oracle-derived results with LLM naturalization; Stage 3 autonomous
Main-Agent action planning remains separate.


## Opt-in Macro-first Stage 1 generation (frozen task boundaries)

The recommended experimental path is now `--macro-first-share` (0..100).
It samples a **Macro Task DAG before** any primitive Gold Program or World:
Goal -> frozen Macro nodes/edges/typed handoff contracts -> independent
local executable Gold Solver for each Macro -> global witness-first World ->
strict answer/edge replay -> Question Author/Critic -> Stage 2 Oracle reports.

Macro-first currently offers three bounded graph families:
- serial_discovery: visual identification -> 1–2 separate keyed resolution
  Macro tasks -> authoritative result
- parallel_resolution: two independent visual/caption Macro tasks -> identity
  fusion Macro -> optional keyed resolution Macro -> authoritative result
- conflict_reconciliation: two independent inventories -> dispute resolution
  Macro -> optional keyed resolution Macro -> authoritative result

Each Macro has `macro_id`, `depends_on`, `input_contract`,
`output_schema`, `source_scope`, and a `solver_template`. The
`compile_frozen_macro_plan` path preserves these boundaries and verifies
that the local solver only consumes declared upstream WorkProducts and
permitted sources. It does not split a Macro due to its internal scans,
filters, joins, or keyed searches. Global Gold Program nodes are retained
only as a compatible deterministic execution/replay representation.

The World is consistent across all Macro handoffs. Validation checks the
exact final answer, frozen graph, execution and provenance, source-fact
counterfactuals and **every Macro handoff** under an empty-input
intervention. A reference DAG is an execution proof and dependency
contract, not the unique possible optimal Main-Agent trajectory.

To try the new path without changing older experiments:

~~~bash
python -m envir generate-questions \
  --config envir/config.yaml --target 100 \
  --macro-first-share 100 \
  --output-dir envir/data/maestro_macro_first_100

python -m envir naturalize-subtasks \
  --config envir/implement_subtask/config.yaml \
  --scenes envir/data/maestro_macro_first_100/questions/scenes.jsonl \
  --output-dir envir/data/maestro_macro_first_reports_100
~~~

`macro_first_share` applies only to Medium and Hard scenes; Easy stays on
the historical baseline. With zero (default), generation behavior remains
unchanged. If macro-first and evidence_graph shares overlap, macro-first
takes precedence on the selected scenes. Observe
`quality_report.json["macro_first"]` for graph pattern counts, necessary
handoff edges, and local operator complexity.

Limitations: current grammar uses three reference/record investigations and
a small set of year-valued goals; this is a bounded Macro-first pilot, not
arbitrary DAG composition. Local sources are synthetic structured
observations, not real media or real tool searches. The Stage 2 Oracle
provides reliable workproducts; it does not prove a real LLM Sub-Agent can
perform arbitrary searches. Stage 3 adaptive orchestration, retries, and
untrusted outcomes remain separate work.


## Macro-first V2: varied goals and local evidence (opt-in)

The newer, isolated pilot `--macro-first-v2-share` changes task **content**
while retaining the frozen Macro-first task-first architecture. It is
independent of `--macro-first-share` (V1); existing V1 ProblemSpecs and
scenes continue to replay unchanged. V2 takes precedence only on its
selected Medium/Hard scenes. Easy remains the older baseline.

```bash
python -m envir generate-questions \
  --config envir/config.yaml --target 100 --macro-first-v2-share 100 \
  --output-dir envir/data/maestro_macro_v2_100

python -m envir naturalize-subtasks \
  --config envir/implement_subtask/config.yaml \
  --scenes envir/data/maestro_macro_v2_100/questions/scenes.jsonl \
  --output-dir envir/data/maestro_macro_v2_reports_100
```

The sampled `macro_plan` freezes each Macro's role, typed handoff,
source scope, and independently sampled `evidence_plan`. Four bounded
DAG families include serial identification, parallel cue resolution,
disputed-report reconciliation and joint **video/audio temporal alignment
inside one Macro**. Optional keyed index hops vary 0–2; these do not
change the Macro-local evidence budgets.

Goals depend on the topic's inferred domain (e.g. match result,
performance venue, publication year, review status, case title). Source
observations have 4–17 candidate rows, depending on independently
sampled evidence volume, with controlled ambiguity. Video observations
include action and timestamps, audio includes spoken cues/timestamps,
and the temporal Macro checks footage time <= commentary time.
Media are **typed synthetic observations**, not real decoded files.

V2 questions use ordinary investigation wording and should not disclose
fictional/synthetic benchmark framing. This presentation requirement is
different from the private metadata: every source remains explicitly
marked synthetic for the researcher. Question/Scenario Author and Critic
use separate V2 prompts to prevent goal substitution and unsupported
historical claims about real named entities.

`quality_report.json["macro_first_v2"]` records goal families, graph
families, source observation volume, direct sources per Macro, and
video/audio joint tasks. Internally exact replay checks frozen goals,
the global answer, provenance, controlled source interventions and
every explicit cross-Macro handoff.

**Boundaries:** This is still a bounded, self-contained synthetic
investigation workspace, not full web browsing or original media
perception. Candidate observation count is an evidence-search volume,
not automatically the number of independently supporting facts. Neither
the synthetic hard label nor a longer local evidence list proves
difficulty for a live LLM Sub-Agent. A local Author/Critic pilot and
independent blind-agent evaluation are needed.


## Source-grounded Semantic Evidence Contract (Macro-first V2)

Newly generated V2 ProblemSpecs now include
`semantic_evidence_contract_version: maestro.semantic_evidence.v1`.
The generator compiles a deterministic `semantic_evidence_contract`
from the **actual Gold Solver's primitive predicates and source schemas**,
including exact typed Goal semantics, image/video/audio fields, relevant
filter constants, compare operations and time relations.

This freezes semantic obligations *before* Scenario/Question Author. Their
prompts receive only the source features that the program actually checks,
not all randomly sampled unused media features. Deterministic gates
reject clear high-confidence violations: fictitious clearance/approval,
audio absent from the workspace, visual color/emblem predicates absent from
video, audio timestamps not used by the Solver, exact seconds claimed for
relative temporal order, and reversed video/audio ordering. Source titles
must also match their local modality. The same checks run at Scenario
draft review, Question draft review and strict Scene replay even when a
question was accepted by the LLM Critic.

The older Hard-intent heuristic incorrectly added an entry-specific
clearance rule to V2; it is now excluded. That invented criterion was a
real source of false Question conditions in prior experiments.

New contracts are **version-gated**: previously frozen V2 scenes without
the marker keep the same compilation/replay behavior. The read-only
`python -m envir quality` report now includes
`semantic_evidence_diagnostics` and can flag potentially unsupported
claims in older V2 questions without modifying or automatically rejecting
those historically published results.

```bash
# Re-audit older V2 scenes (read-only)
python -m envir quality --config envir/config.yaml \
  --scenes envir/data/maestro_macro_v2_100/questions/scenes.jsonl \
  --output envir/data/maestro_macro_v2_100/semantic_recheck.json

# New pilot: choose a fresh destination
python -m envir generate-questions --config envir/config.yaml \
  --target 100 --macro-first-v2-share 100 \
  --output-dir envir/data/maestro_macro_v2_semantic_100

python -m envir naturalize-subtasks \
  --config envir/implement_subtask/config.yaml \
  --scenes envir/data/maestro_macro_v2_semantic_100/questions/scenes.jsonl \
  --output-dir envir/data/maestro_macro_v2_semantic_reports_100
```

The new static check is **not** a proof of unrestricted natural-language
entailment, open-world search, real image/video/audio perception or
absence of all alternative solution paths. It is intentionally
conservative and should be evaluated against an independent blind
question reviewer. Do not compare revised Scenario acceptance rates
without reporting the stricter validation criteria.
