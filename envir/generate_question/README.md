# MAESTRO — Stage 1: Generate Question

**One supported generation path.** This module ends once a source-scoped
synthetic World, executable Macro TaskGraph, Gold output, and independently
reviewed natural-language Question are frozen. It does not implement the
future Subtask Execution or Subtask Scheduling phases.

## Run

From the MAESTRO repository root, with the existing vLLM endpoint running:

~~~bash
python -m envir generate-questions --config envir/config.yaml
~~~

All routine settings live in **envir/config.yaml**:

- input: MultiModalQA topic/source hints and source scan limit
- generation: output directory, target attempts, seed, Easy/Medium/Hard mix
- scenario: OpenAI-compatible Author/Critic endpoints and concurrency
- question: OpenAI-compatible Author/Critic endpoints and concurrency

Override a couple of experiment values without editing YAML:

~~~bash
python -m envir generate-questions --config envir/config.yaml \
  --output-dir envir/data/question_run_500 --target 500
~~~

For this command to work, input.max_records must be >= the overridden target.
Changing an existing run's output in place is intentionally not the default.

## Frozen contract and QA gates

MultiModalQA topic metadata -> frozen ProblemSpec -> optional Scene Author +
Critic -> synthetic source World and executable Reasoning Program -> compiled
Macro Tasks and reference executor -> independent Question Author + Critic ->
strict replay and Question-Macro contract gate.

The Question Author works from user intent, readable titles and purposes of
available materials, and a style hint. The Critic also sees the compiled
Macro contract and reference question. Raw output labels such as
`entry_label` are rendered in ordinary language before review.

The final Question is checked against the ACTUAL MacroGraph:

1. It may not invent eligibility predicates missing from the executable
   operator chain, including "no archived", independent approval in a
   threshold-only task, or entry clearance when not required.
2. The requested OUTPUT field must match the terminal schema. In particular,
   entry_label means an entry's display name, never label_code or route_code;
   requesting abbreviated codes where the Graph only projects entry_label
   fails review.
3. It must preserve active, code-translation, exact comparison direction,
   source-specific constraints and the required final answer kind.

The deterministic screens intentionally target explicit, high-confidence
mistakes. The independent model Critic still reviews unrecognized semantic
paraphrases; neither provides a formal proof of complete equivalence.

## Output directory

~~~text
question_run_500/
  constructed/
    scenes.jsonl          # PRIVATE frozen World, Program, MacroGraph and Gold
    questions.jsonl       # provisional reference Questions (NOT final)
    failed.jsonl          # rejected Scenario/World candidates
    difficulty_audit.jsonl
    summary.json
  questions/
    scenes.jsonl          # PRIVATE ACCEPTED reviewed Questions
    questions.jsonl       # FINAL PUBLIC accepted Questions only
    failed.jsonl          # PRIVATE rejected Question attempts + reasons
    difficulty_audit.jsonl
    summary.json
  quality_report.json
  summary.json
~~~

**Fallback is failure.** A target of 100 with 9 Scenario rejections and
3 further Question rejections reports attempted=100, generated=88,
failed=12, status=partial. It never fills the public accepted set with
reference Questions or counts Critic failures as successes.

One rerun to validate the accepted Question bank (no model calls):

~~~bash
python -m envir quality --config envir/config.yaml
~~~

Or explicitly point the read-only audit at a previous output:

~~~bash
python -m envir quality \
  --scenes envir/data/question_run_500/questions/scenes.jsonl \
  --output envir/data/question_run_500/quality_recheck.json
~~~

The quality report includes replay, Question semantic warnings, answer
diversity, Macro DAG signature diversity, sources and operation depth.
Image/video/audio roles are fictional structured observations, not original
media perception. Do not publish PRIVATE World, Program or Gold material.

## Phase boundaries

- Stage 1 — Generate Question: **this package**.
- Stage 2 — Implement Subtask: **future**, consumes frozen public task and
  authorized source contracts.
- Stage 3 — Combine/Schedule Subtasks: **future**, consumes executable subtask
  interfaces and the frozen Macro DAG.

No premature cost optimization or source-trust adversarial state is modeled
during question generation.


## Executable clue acquisition (opt-in)

The new clue_chain / accession_year semantic family intentionally has
a different goal than the old approval, time-window, status disagreement
and threshold templates. One illustrated exhibit must be identified
by TWO observed motif features; its marker opens an index record, and
the resulting accession identifier opens exactly one archive record.
The desired answer is an accession year, not a qualifying entry list.

The Program IR now includes lookup_by_key:
- one typed upstream relation input with a key field and provenance
- one explicitly authorized keyed store with an exact key-field match
- output containing ONLY matching source records and merged fact refs
- no unrestricted scan or read_source fallback for keyed stores

The two lookups are separate Macro tasks so Main-Agent can see and
dispatch the acquisition dependencies. No private World or Gold values
are exposed in the initial public Question view.

The frozen Spec declares investigation.visual_features and an ordered
acquisition_chain, independent of the LLM's prose. Question Author
can make a natural clue-led request, but is not permitted to invent a
different marker, feature, source path or final output. Strict replay
checks both source access modes, missing-source sensitivity, misleading
single-source shortcuts, visual decoys, and changing the pictured
marker to redirect the archive answer.

The Stage-1 CLI accepts --clue-chain-share 0..100 for MEDIUM difficulty.
This is an isolated clean pilot with simulated image annotations, not
yet open-web source discovery, video decoding or adversarial feedback.


## Seeded typed graph grammar (optional)

For evidence-acquisition investigations, use --evidence-graph-share 0..100
(or generation.evidence_graph_share in envir/config.yaml). Medium and Hard
tasks are eligible. This supplements, rather than replaces, the legacy
static computation families and the older fixed clue_chain pilot.

GraphSpec is frozen before any synthetic World is sampled:
typed source schemas and policies + legal topological Program IR +
answer-field contract. The sampler chooses one of four information
acquisition motifs (variable-depth chain, parallel evidence fusion,
candidate disambiguation, conflicting-inventory reconciliation). The
World synthesizer first creates an answer-supporting witness, then
overlapping distractors with random public identity labels. Downstream
keyed stores cannot be browsed or scanned in the QuestionSession.

Strict replay checks World regeneration, compiled Program, exact answers,
source provenance, Source Scope, mandatory-source counterfactuals,
and Macro schedule. Graph validation rejects type-mismatched or
non-grammar operator mutations. Read-only quality reports include
evidence_graph.archetype_distribution and
evidence_graph.source_counterfactual_full_pass_rate.

Example: python -m envir generate-questions --config envir/config.yaml
--target 100 --evidence-graph-share 100 --output-dir
envir/data/evidence_graph_pilot_100

Limitations: the grammar is intentionally bounded; media are simulated
structured observations; structurally valid DAGs are not necessarily
hard for a blind Main-Agent. No unsupported original MMQA facts are used.


## Macro-first generator

Use `--macro-first-share 100` or `generation.macro_first_share: 100`
to select a Macro-first generator for eligible medium/hard cases. The
frozen `problem_spec.macro_plan` is sampled before any local solver:
each Macro declares its goal role, dependencies, typed upstream input
contract, typed output, and private source scope.

`macro_first.compile_macro_solvers` builds a local primitive operator
chain for each frozen Macro. Primitive nodes are topologically flattened
into `gold_program` only to reuse existing Oracle, trace and replay
infrastructure. `compiler.compile_frozen_macro_plan` preserves every
original Macro ID and edge and rejects cross-Macro primitive access that
is not explicitly declared. It is deliberately distinct from the old
`compile_macro_tasks`, which fuses primitive nodes after the fact.

`macro_first_world` synthesizes globally consistent source facts,
dependent keys and overlapping decoys. Strict replay re-samples the
World and local solvers from the frozen Spec and checks all source and
Macro handoff counterfactuals. `difficulty_audit.macro_first_audit`
contains the preserved-boundary and per-edge dependency diagnostics.

Question Author and Critic see the semantic requirements but must not
reveal the internal Macro execution plan. Source data for images,
video and audio are synthetic annotations only. Old `clue_chain` and
`evidence_graph` settings remain supported and unchanged by default.


### V2 Macro-first goal and evidence composition

Use `--macro-first-v2-share 100` to enable the isolated
`macro_first_v2.py` sampler and `macro_first_v2_world.py` synthesizer
for Medium/Hard scenes. V1 remains available with
`--macro-first-share 100`; both default to zero, and old frozen V1
scenes still replay.

Each V2 ProblemSpec freezes the domain and **typed final Goal** before
any World values are sampled, as well as the Macro topology and each
Macro's `evidence_plan` (source observation targets and evidence volume
grade). Witness-first synthesis creates a consistent source-scoped World,
then `compile_frozen_macro_plan` verifies local Gold Solvers against
those explicit Macro boundaries. Macros with multiple internal source
scans are legal: video/audio event alignment compares timestamp order
in one Macro and passes a typed WorkProduct to a downstream fusion.

The question is phrased as a natural working investigation without
revealing the synthetic nature of the World, but the private scene
retains `simulated_only=True` and does not assert actual historical
facts. Question Guard rejects meta-fiction terms and missing Goal,
modality or temporal clues. Scenario/Question LLM prompts are
version-specific so V1 and legacy reference questions do not change.

V2 is bounded: 4 topologies; variable 0–2 keyed index hops; typed
image/video/audio/text/table observations; topic-conditioned year,
string, status and venue/title goals. This improves variety but does
not provide real video decoding, real-world search or formal optimal
Macro decomposition.


### V2 semantic answer/source fidelity

`semantic_evidence.py` generates an internal, frozen semantic contract
directly from the true local Solver operators (scan/filter/join/compare/
lookup) and the source schemas. A field's presence alone is not evidence
that the Solver uses that field as a filter. In particular, videos do not
acquire image-only emblem properties and audio utterance timestamps do
not become exact footage-to-audio intervals.

`scenario._narrative_issues` validates proposed source titles and
narrative goals against the contract before LLM Critic. `question.guard_question`
rejects explicit unsupported predicates and modality/time assertions,
while authoring receives only used features. `quality.audit_scene`
rechecks all LLM-authored Questions and ensures the stored semantic
contract exactly matches regenerated Gold Program, so a passing LLM
Critic does not bypass it.

The additional version marker is opt-in for newly generated V2 specs.
Old V2 snapshots remain replayable; the read-only quality report
`semantic_evidence_diagnostics` can assess them retrospectively.
Conservative regex diagnostics are high-confidence guards, not complete
logical entailment and not a replacement for independent Agent tests.


### Macro-first information necessity (new versioned scenes)

New specs freeze `macro_information_necessity_version` on the
`macro_plan`. For `multimodal_event`, video+audio yields at least
two valid candidate IDs and the independent context log yields at
least two; each contains a different false positive, and the overlap
is exactly one. Competitors' authoritative final answer values are
distinct. The private `information_necessity.audit_candidate_information`
walks Macro WorkProducts and the keyed index/archive chain to ensure
that dropping a branch loses needed identity/answer discrimination,
and stores its finding in the Scenario difficulty audit.

New scenes' `semantic_evidence_contract` also freezes a single
typed output field and applies conservative checks against explicit
additional response targets (e.g. asking for a title AND a year).
Legacy V2 plans, Worlds and semantic contracts without this marker
retain their older deterministic replay behavior. This check is a
bounded output-slot guard, not a full natural-language entailment
proof.


### Schema-constrained Scenario + one-output Goal (new V2 marker)

`source_title_contract.py` compiles each public source title from the
frozen `source_id`, `modality` and domain; LLM output cannot silently
relabel text captions as audio or image observations as video. The
domain-context Goal and evidence roles are provided to the Scenario
Author, which only drafts narrative fields.

`semantic_evidence.py` has a versioned single-answer-slot guard for
explicit conjunctive output requests (e.g. asking *which case* AND
*what status* when the Gold Answer has only status). The deterministic
reference Question for newly marked V2 scenes also asks for exactly
one output field, with case identity only as a qualifying clue.
Supported morphological paraphrases such as triangle/triangular are
accepted.

`macro_first_v2.py` samples a frozen `candidate_profile` for the
multimodal family; `macro_first_v2_world.py` realizes variable branch
ambiguity and ensures alternatives can lead to other final answers.
`information_necessity.py` audits these candidate sets from actual
WorkProducts. `difficulty_proxy.py` reports interpretable structural
workload features by target label. No claims of measured Sub-Agent
accuracy, open-web retrieval or video/audio perception are implied.
Older frozen V2 ProblemSpecs are not retroactively changed.
