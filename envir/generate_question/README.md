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
