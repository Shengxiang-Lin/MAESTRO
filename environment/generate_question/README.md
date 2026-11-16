# Stage 1 — V1.4 evidence-aware Macro programs

```bash
python -m environment generate-questions --config environment/configs/questions.yaml
```

Stage 1 constructs frozen synthetic sources, a fully executable Macro DAG,
Gold Program, Gold Answer, FactRefs, and (for report-dispute patterns) a
private evidence relation ledger. The existing Scenario/Question Author and
Critic LLMs only naturalize the user-facing task; **no LLM judges Claim
relations, provenance or trust-state reference updates**.

## New in V1.4

- `surface_quality.py` and `question.py` accept grounded `locate` requests
  and compositional two-source higher-count paraphrases without relaxing
  mandatory output and numerical constraints.
- New eligible disputed-report Specs sample `claim_verification_mode` as
  `parallel` or `keyed_followup` using a separate deterministic hash stream.
  Keyed follow-up can only retrieve records from independent report B
  **after** report A has supplied candidate case IDs. It is a real typed
  `lookup_by_key` with Source A provenance, not a manufactured delay.
- `dispute_resolution` now executes `claim_relation_filter`, a typed
  provenance-checked primitive. Rows must carry FactRefs from both report
  sources; compared year fields must be comparable. Historical V1.3
  scenes without `evidence_trust_version` retain the old `compare_fields`
  program and exact Replay behavior.
- `evidence_relations.py` restricts the Claim Ledger to facts actually
  accessible to their source Macro. New `reference_wave_trust`
  annotations track when evidence becomes **eligible** following each
  frozen Macro wave and which target-related Claim pairs gain independent
  support or become disputed.
- `quality_report.json` separates raw Claim-pair counts from
  answer-relevant, genuinely cross-wave pairs and state transitions.
  All ledgers and reference schedule annotations are **private**, never
  added to the user Question or Orchestra-o1 Main prompt.

`reference_wave_trust` means *source evidence became available to a
Macro*, **not** that every item was necessarily stated in a natural-language
Sub-Agent finish report. Main-visible state supervision must be computed
from actual delivered reports, with no hidden World-to-prompt shortcut.
Evidence source independence means independent authorship/lineage, not the
absence of task data dependencies: report B can depend on case IDs from
report A without copying its year assertion.

Stage 2 materializes frozen Macro WorkProducts and naturalizes reviewed
Sub-Agent reports. For V1.4 source-report Macros, a compact list of actual
returned case/year rows is added deterministically to the reviewed summary.
Stage 4 separately parses literal returned text and records private
`visible_claim_state` transitions (including faulty/missing reports and
retries) without reconstructing unseen facts from the World. Stage 4 provides adversarial feedback and runtime
Main-Agent recovery behavior; truthful source disagreements are not
themselves Sub-Agent faults.

Previous V1.3 data are retained in distinct directories. Strict Replay
must continue to work on old frozen scenes; use a fresh output directory
for any new generation run.


## Typed Macro upgrade (incremental, versioned)

New work on the typed Macro DAG now lives **in this canonical Stage-1 package**
rather than in an independent generator tree:

- `typed_role_grammar.py`: Discovery, Identification, Investigation,
  Verification, Comparison and Synthesis port contracts.
- `typed_graph_generator.py`: deterministic typed partial-order graph
  sampling with concurrency and wave budgets.
- `typed_graph_validator.py`: check edge types, no invented dependencies,
  reachability, topological order, and legitimate cross-wave source evidence.
- `typed_macro_quality.py`: enforce requested final WorkProduct coverage
  and audit every port with controlled answer-sensitivity / execution-blocking
  counterfactuals. Operational blocking is explicitly **not** mislabeled a
  proof of semantic necessity.

The standalone `environment.macro_v3 --variant v3.2` pilot now imports these
modules using backward-compatible shims. Its final Synthesis retains each
requested Investigation, Verification and Comparison work product, even
when those products are also inputs to other Macros. This prevents a
VerifiedClaim from disappearing merely because another investigation
consumed it.

**Migration boundary:** The canonical production `generate-questions`
command, frozen V1.3/V1.4 ProblemSpecs, `quality.audit_scene()`, the
legacy deterministic macro executor, and Stage 2–4 interfaces are
intentionally unchanged in this first migration. The experimental typed
World/solver is **not** yet compiled into the original
`problem_spec → gold_program → macro_graph` contract. It is incorrect
to claim full old Stage-1 quality/Replays or Stage-2 compatibility for
typed v3.2 outputs. This work consolidates the shared generator and
quality foundations for the next adapter, without weakening old audits.

## Canonical YAGO-backed Macro-first pilot (typed_knowledge_v1)

The previous v3.x independent scene format is **not** the production entrypoint.
This opt-in family uses the ORIGINAL `generate-questions` pipeline and its
existing relational Gold Program, frozen Macro Compiler, FactRefs, strict
Replay, source/edge counterfactual audits, and Stage-2 materializer.

```bash
pip install "pyoxigraph>=0.5,<0.6"

python -m environment generate-questions \
  --config environment/configs/questions_knowledge.yaml

python -m environment quality \
  --config environment/configs/questions_knowledge.yaml \
  --scenes environment/data/stage1_knowledge_pilot/constructed/scenes.jsonl
```

For a dry run without LLM Author/Critic, the supplied YAML has null
`scenario.model/api_base` and `question.model/api_base`. Add both
endpoints when ready to run naturalization; a model-free question is still
a deterministic **draft**, even if its executable quality and Replay pass.
Use a fresh output directory when changing configuration or models.

### What is implemented

- YAGO 4.6 entity-to-entity relations are sampled from the supplied local
  `data/world-source/YAGO-4.6` facts archive; no fabricated RDF assertion is
  labeled as YAGO. A per-run adjacency and predicate index is built once.
- A new versioned, frozen `typed_knowledge_v1` ProblemSpec records the actual
  RDF subject/predicate/object witnesses and origin file. Historical
  Replay is self-contained: it rederives the same World/Program from the
  frozen spec, not from a potentially changed multi-GB dump.
- Each task has 2–3 independent coarse Discovery Macros, each investigating
  two separately scoped simulated source records. Identification reconciles
  the candidates and obtains the YAGO-entity reference using a keyed index.
- A 2–3-hop genuinely connected YAGO relationship path is investigated
  within one or two coarse Investigation Macros (rather than always making
  a single relation lookup into a Macro). Relation evidence uses keyed
  access, not unrestricted full-source scans.
- The original `program_ir` operators are unchanged:
  `scan/filter_eq/join/unique/lookup_by_key/project`. The original
  `compile_frozen_macro_plan`, `run_macro_tasks`, `macro_contract`,
  `audit_scene`, `validate_replay` and Stage-2 `realize_scene` execute
  the new family with the same interface as old Stage-1 records.
- Deterministic question contracts require every necessary clue and the
  single terminal linked-entity answer field. The Question Author/Critic
  path can be enabled through the same configuration.

### What is NOT implemented yet

- This is **one reliably compilable, variable-sized subgrammar**, not
  arbitrary six-role typed graphs. Verification, Comparison and more
  sophisticated cross-wave dependencies still need typed IR/World compiler
  support before switching the production mix.
- The facts are real YAGO RDF, but multimodal archive cues and case
  crosswalks are explicitly **fictional simulated evidence**; two simulated
  cards representing one RDF assertion do not imply independent historical
  sources. YAGO provenance is checked during sampling, whereas offline
  replay checks the frozen witness and its checksums.
- Current KnowledgeGraph parsing focuses on entity-to-entity facts;
  entity types and multi-language label indexes are future work.
- The currently supported family is graph-grounded entity retrieval,
  not general OmniGAIA/MM-BrowseComp-style open-web multimodal reasoning.

Do not change the existing `questions.yaml` or frozen V1.3/V1.4 data to
enable this pilot. Production Stage 2–4 uses accepted old-format scene
banks as before.

## Knowledge V2: candidate necessity and shortcut resistance

The first canonical YAGO pilot (`typed_knowledge_v1`) passed existing frozen
handoff/source audits, but its Identification workproducts could contain
multiple candidates. Later YAGO lookups accidentally removed unrealistically
unlinked decoys. **V1 remains byte-for-byte replayable**; it is not silently
re-labeled as validated for stronger information necessity.

Use the stricter **opt-in** V2 mode, with a new output directory:

```bash
python -m environment generate-questions \
  --config environment/configs/questions_knowledge_v2.yaml

python -m environment quality \
  --config environment/configs/questions_knowledge_v2.yaml \
  --scenes environment/data/stage1_knowledge_v2_pilot/constructed/scenes.jsonl
```

V2 constructs a real, *complete* YAGO relation trail for **every** candidate
dossier (not only the answer), using exactly the same predicate sequence.
All candidate starts, intermediate keys and terminal answer labels must be
distinct. Motifs with too few actual alternative paths are rejected;
no synthetic fake RDF links or absent records count as valid distractors.
YAGO is still a single underlying factual provenance, even when exposed
as multiple synthetic evidence excerpts.

For each independent simulated clue, exactly one competing dossier matches
all the *other* clues but fails that clue. Other background distractors
fail all clues. Consequently every Discovery clue is necessary for a
unique Identification *before* following the YAGO path.

The original `information_necessity.py` now contains an extended
`knowledge_candidate_replanning_v2` audit:

1. Verify the Identification Macro's actual WorkProduct contains **one**
   target entity, independently of any downstream YAGO lookup.
2. Execute the frozen Gold Program with **one clue filter relaxed**;
   multiple distinct valid final answers (not an empty or blocked program)
   must remain.
3. Repeat with both filters belonging to **each Discovery Macro** relaxed.
4. Relax **all** Discovery filters; the program must still return
   multiple real alternatives. Thus a short-circuit that simply uses the
   known YAGO relation table no longer yields a unique answer.

The results appear at
`quality_report.json → knowledge_world_grounding.necessity_v2` with
counts for each gate; detailed per-scene records live at
`difficulty_audit.macro_first_audit.candidate_information_necessity`.

These are **bounded re-planning counterfactuals** within the frozen
materialized World, not a proof that every possible adversarial shortcut
or external knowledge strategy is impossible. V2 remains the
Discovery/Identification/Investigation family, not a full six-role typed
DAG compiler. The Question is a deterministic draft until naturalized
and independently reviewed.

## Hybrid World v3 — real YAGO motifs + fictional entities (opt-in)

This is an incremental extension of the **canonical**
`environment generate-questions` workflow, not an independent scene
writer. It freezes `typed_knowledge_v3` ProblemSpecs and reuses the old
Program IR, Macro Compiler, WorkProducts, Gold/Trace, source and handoff
counterfactuals, exact replay, and Stage 2 compatibility.

```bash
python -m environment generate-questions \
  --config environment/configs/questions_hybrid_world_v3.yaml

python -m environment quality \
  --config environment/configs/questions_hybrid_world_v3.yaml \
  --scenes environment/data/hybrid_world_v3_pilot/constructed/scenes.jsonl
```

Versioned `v1`/`v2` task banks and their replay semantics are unmodified.
Use a **new output directory** for v3, never overwrite frozen older banks.

### Real + fictional layers

- **`world_motifs.py`** mines only actual YAGO RDF triples as
  connected chains, forks or (when available) diamonds. The target
  multi-hop route remains a frozen, truly witnessed YAGO relationship
  chain with real competing candidates. Additional mined motif edges
  are explicitly **context only** in this first pilot: they do not earn
  a claimed causal dependency or second-publisher credit.
- **`hybrid_world.py`** expands every YAGO-grounded case into a
  **fictional** research project / exhibition / field survey / heritage
  project, with a synthetic project→real-reference ground-truth link.
  The private ledger separates these synthetic truths from YAGO facts.
- Two separately *simulated* project-reference source directories
  each list two real-entity candidates per fictional project: one
  correct, one different, unverified alternative. Their intersection
  uniquely determines the project's reference. Claims in synthetic
  source rows are not automatically true synthetic facts and are
  **not real historical documents**.
- Real candidate YAGO paths are complete. A missing/discarded Discovery
  still leaves multiple *different valid YAGO answers*, instead of
  filtering away broken decoys.

### Executable Macro patterns

The same types of local information goals can be composed into either:

```text
parallel_fork_join:
  Discovery x 2–3 ─→ Identification ─┬→ Project Reference A ─┐
                                    └→ Project Reference B ─┤
                                             Verification JOIN
                                                      ↓
                                           YAGO Investigation x 1–2

serial_refinement:
  Discovery x 2–3 → Identification → Reference A
    → Evidence-dependent Verification B → YAGO Investigation x 1–2
```

Each step compiles to the ORIGINAL relational Gold Program, with actual
source scopes and fact references; no separate typed v3 scene executor.
A total of up to 9 Macro tasks retains the original Stage-1 budget.

The old `information_necessity.py` additionally runs the new
`hybrid_world_replanning_v3` controlled executable tests: unique
Identification, omit each clue, omit each Discovery, omit all Discovery
filters, and make each synthetic mapping directory uninformative in
turn. Each bypass must leave multiple feasible answers, never merely
an empty or blocked execution. See the `hybrid_world_v3` section of
`knowledge_world_grounding` in `quality_report.json`.

### Current research boundary

The mixed World and two Macro compositions are validated as an executable
hybrid **subgrammar**; arbitrary six-role DAGs and systematic use of the
extra real fork/diamond edges as independent verification subgoals are
future expansions. These synthetic media records are programmatically
realized annotations, **not actual decoded images/audio/video**.
Deterministic Question drafts should still be naturalized and reviewed
before training distribution-scale models.

## Hybrid Motif v4 — causally required YAGO forks + fictional event networks

This version specifically addresses the observed v3 pilot gap: genuine YAGO
forks/diamonds were previously stored in World metadata, while fictional
projects had few visible attributes. v4 still uses the **original**
`generate-questions`, canonical World, primitive Program IR, frozen Macro
compiler, evidence and source permissions, information necessity, strict
Replay and Stage-2 task-feedback interface.

```bash
git pull origin main
python -m environment generate-questions \
  --config environment/configs/questions_hybrid_motif_v4.yaml

python -m environment quality \
  --config environment/configs/questions_hybrid_motif_v4.yaml \
  --scenes environment/data/stage1_hybrid_motif_v4_pilot/constructed/scenes.jsonl
```

Existing v1/v2/v3 frozen data and configuration semantics are unchanged.
The new variant is `typed_knowledge_v4` and writes to a **new output
directory**, rather than silently replacing previous scenes.

### Two mutually necessary knowledge layers

- `real_motif_grounding.py` mines a *genuine secondary RDF fork*
  from each competing YAGO anchor, using a common predicate and
  distinct real RDF objects. If the input YAGO sample does not have
  enough matched secondary facts, the candidate World is **rejected**;
  no fictional edge is labeled real. The complete YAGO answer routes
  are still drawn from genuine connected facts as in knowledge v2.
- `hybrid_world_v4.py` creates fully fictional projects **and** typed
  fictional events, organizers and venues. It freezes the synthetic
  ground-truth links, event eligibility relationships and source claims,
  with entity names visible in `entity_index` and `event_ledger`.
  These names and relations are fictional; they must not be
  misrepresented as YAGO or historical facts.
- Two simulated project-reference directories, the fictional event's
  permitted real-fork objects and the actual YAGO fork lookup are
  all required by the final solver. For each candidate project, let
  `T` be the true YAGO anchor and `U,V,W` other genuinely queryable
  anchors: A provides `{T,U,V}`, B provides `{T,U,W}`, the event
  permits `{fork(T),fork(V),fork(W)}`. The actual RDF fork lookup
  disambiguates the intersection to `T`. Unlike v3, the second
  genuine YAGO relation is part of the **Gold Program and FactRefs**.
- The Macro graph dynamically uses either parallel reference investigations
  or serial refinement, an independent event Investigation, a required
  real-motif Verification (joining the two information layers) and the
  main witnessed YAGO multi-hop Investigation. The original nine-Macro
  budget is enforced. Other role combinations are not implied.

### Quality and limits

The canonical `information_necessity.py` adds the
`hybrid_world_causal_motif_v4` audit: exact Identification, single clue
relaxation, whole-Discovery omission, all-Discovery bypass, each
simulated directory becoming nondiscriminating, event eligibility
becoming nondiscriminating, and the real-fork observation being
counterfactually unreliable. Every probe must reopen **multiple
executable distinct** YAGO answers. Empty results do not count.

Summaries are in
`quality_report.json → knowledge_world_grounding.causal_hybrid_v4`.
Each Scene has the full per-source audit under
`difficulty_audit.macro_first_audit.candidate_information_necessity`.

- Both event records and existing image/audio/video observations are
  simulated media annotations, **not decoded real images/audio/video**.
- Real fork facts are independently present in the input RDF, but
  simulated source assertions are not independent historical publishers.
- The current **required real motif is a Fork**. Genuine YAGO
  Diamonds can still be mined in the v3 context layer but an entire
  independent two-hop diamond path is **not yet** an obligatory
  Verification subgraph.
- The controllable candidate design proves the stated bounded
  counterfactual properties, not global minimality of every possible
  strategy. Questions are deterministic drafts without LLM review.

## True Macro task-first generation (opt-in `graph_first`)

The experimental `graph_first` knowledge variant is the first integrated
four-stage **task-led** implementation built on the ORIGINAL canonical
`generate-questions` / Gold Program / Macro Compiler / Quality / Stage-2
pipeline. It is intentionally **not** another monolithic
`hybrid_world_v5.py`.

```bash
python -m environment generate-questions \
  --config environment/configs/questions_task_graph_first.yaml

python -m environment quality \
  --config environment/configs/questions_task_graph_first.yaml \
  --scenes environment/data/stage1_task_graph_first_pilot/constructed/scenes.jsonl
```

### Generation sequence and frozen contracts

1. **`macro_graph_sampler.py`** samples an abstract source-free,
   relation-free Macro DAG, with typed output/input information ports.
   Three currently compilable topologies are balanced in small batches:
   `parallel_association`, `serial_refinement`,
   `event_gated_association`. Each graph includes real information
   operations spanning Discovery, Identification, Investigation,
   Association, Comparison, and Synthesis. Synthesis performs actual
   source-scoped YAGO queries, not an answer-only paraphrase.
2. **`relational_task_planner.py`** converts that graph into a typed
   **Relational Task Sketch**. It specifies what relationships, ports,
   real RDF motifs, synthetic event constraints and competing evidence
   are needed without selecting any entity, predicate or answer.
3. **`world_grounder.py`** searches actual YAGO data for a valid
   multi-hop path and a genuine secondary relation per competing
   candidate. It reuses the proven v4 RDF/World sampler strictly as a
   **knowledge witness generator**; its original task graph is NOT
   adopted. It rejects motifs without complete real alternatives
   rather than fabricating real RDF.
4. **`evidence_planner.py`** instantiates synthetic projects and events,
   with a `completed` event and a competing `scheduled` event for
   each project. The fictional records and genuine YAGO evidence jointly
   determine the answer. For the event-gated topology, downstream
   directory rows carry `event_id` and require that typed upstream
   handoff. Facts, source candidate claims and context fields carry
   distinct provenance and necessity annotations.
5. **`task_graph_first.py`** freezes the above four contracts in a
   versioned `task_graph_first_v1` ProblemSpec, converts the chosen
   DAG into original source-scoped primitive Program IR, and lets the
   unchanged core execute it and create old-format WorkProducts.
   Existing Handoff/Source audits, FactRefs, strict Replay and Stage-2
   realization all apply.

### Causal and fidelity gates

The existing `information_necessity.py` now dispatches a
`task_graph_first_replanning_v1` test: each simulated identifying
clue, each whole Discovery Macro, both competing source directories,
the **actual** YAGO fork assertion and the **specific completed-event
status field** are subjected to executable re-planning interventions.
Removing any required information must leave multiple distinct
*feasible real YAGO answers*, not merely an empty result or a failed
lookup. Results and topology frequencies are summarized in
`quality_report.json -> knowledge_world_grounding.task_graph_first_v1`.

Fields such as organizer names, venue descriptions, event name and year
are kept as **context**, not falsely counted as causally necessary
information. The completed-versus-scheduled event status and genuine
YAGO fork are specifically audited at field level. The generated
Question is a deterministic, contract-checked draft unless you enable
the existing optional LLM Author/Critic.

### Compatibility and current scope

- Historical `macro_first_v2`, `typed_knowledge_v1` through
  `typed_knowledge_v4` scenes remain byte-for-byte replayable.
  `graph_first` is opt-in, with a new output directory; the default
  generation profile is not changed.
- This is an **executable constrained Macro grammar with three distinct
  DAG topologies**, not yet arbitrary task combinations or all possible
  real Diamond/Compare/Summary workflows.
- Genuine YAGO triples are frozen and labeled as RDF witnesses.
  Fictional people, venues, projects, event records and source claims
  are explicitly simulated; an image/audio/video observation is
  a programmatically generated evidence annotation, not decoded
  external media.
- The bounded counterfactual gates establish the stated in-World
  necessity constraints, not a formal proof that no shorter global
  strategy can ever recover the answer.
- The grounding layer currently leverages v4's established real
  candidate-path sampler, so its *fact-family diversity* is still
  narrower than the full macro/relationship grammar roadmap.

## Task-Graph-First v2: causal ports and locally composed relation goals

The previous `graph_first` v1 pilot demonstrated source-free DAG sampling
before YAGO grounding, but revealed a critical false handoff: all seven
`event_gated_association` scenes returned **the same candidate links for
every event_id**. Replacing event_id joins with project_id joins yielded
the exact same answer. v1 scenes are still frozen and replayable, but that
test is now a regression *negative control*.

**Use the new opt-in variant** (do not overwrite historical banks):

```bash
python -m environment generate-questions \
  --config environment/configs/questions_task_graph_first_v2.yaml

python -m environment quality \
  --config environment/configs/questions_task_graph_first_v2.yaml \
  --scenes environment/data/stage1_graph_first_v2_pilot/constructed/scenes.jsonl
```

The revised generation sequence is still
**source-free Macro DAG -> task-local Relation Sketch -> genuine YAGO &
fictional World -> event-dependent evidence -> original Gold Program &
Quality -> Question**.

### Composable local relations

`relational_task_planner.plan_relations_v2` now selects one of two typed
event information goals **before selecting any World facts**:

- **status_completed:** use an actual `filter_eq(event_status,
  "completed")` on fictional event records. Event dates are kept equal,
  so year alone does not distinguish selected and excluded events.
- **year_at_least_2025:** use an actual `filter_gte(event_year, 2025)`.
  Event statuses are intentionally identical, so status alone cannot
  reproduce the required temporal predicate.

These are different executable Relation Sketch requirements, not cosmetic
role names. Their 2 choices combine with 3 original partial-order DAG
structures; this pilot supports up to **six typed local-relation / DAG
combinations**. It is still a constrained subgrammar, **not arbitrary
DAG generation**.

### Genuine EventId Handoff and alternative-plan gate

For the event-gated shape, the two source directories now expose
**different candidate hypotheses for the qualifying and non-qualifying
event**. The correct event-specific lookup has the original competing
sets A={T,U,V}, B={T,U,W}. A supplementary, non-qualifying event
contains an additional W in source A and an additional V in source B,
each grounded in complete genuine YAGO candidate paths.

Replacing BOTH event_id joins with project_id joins therefore admits
extra, executable final YAGO answers. In contrast, the required
event_id plan uniquely returns the intended answer.

`evidence_planner.audit_information_v2` actively recompiles the
alternative primitive joins and reruns the original Gold Program. It
fails closed when removing event_id leaves the same singleton, returns
empty results, or creates no distinct valid answer. It also verifies
the actual source candidates differ by event and checks clue, Discovery,
source, chosen event field and real-fork necessity.

The public `quality_report.json -> knowledge_world_grounding.
task_graph_first_v2` section records:
`local_event_relation_distribution`, six typed graph/relation
signatures, `event_keyed_cases` and `event_keyed_bypass_blocked`.
Detailed per-scene counterfactuals are available under
`difficulty_audit.macro_first_audit.candidate_information_necessity`.

### Compatibility and limits

- The old `task_graph_first_v1`, `typed_knowledge_v1`–v4 and original
  `macro_first_v2` code paths, frozen World facts, and Replay remain
  unchanged; v2 has an independent spec, evidence hash, Quality dispatch,
  config and destination.
- The new Relation Sketch still uses a single shared YAGO-path family
  and fixed high-level role grammar; status vs year is one practical
  compositional step, not full open-ended World/role recombination.
- Real YAGO triples remain genuine frozen witnesses. Projects,
  observations, events and simulated source claims are explicitly
  fictional. Organizer and venue fields remain Context and are **not**
  credited as causally necessary.
- The alternative-plan test is a bounded, executable counterfactual in
  the frozen World. It does not prove global optimality or exclude
  shortcuts using knowledge not in the simulated corpus.

## Current experimental generator: one Task Plan (not stacked legacy Specs)

Use the opt-in configuration `environment/configs/questions_task_plan.yaml`:

```bash
python -m environment generate-questions \
  --config environment/configs/questions_task_plan.yaml

python -m environment quality \
  --config environment/configs/questions_task_plan.yaml \
  --scenes environment/data/stage1_single_task_plan_pilot/constructed/scenes.jsonl
```

The generator consists of **two current files** rather than a new stack
of complete versioned World Generators:

- `task_plan.py` first freezes the source-free typed Macro information
  tasks. It chooses the number of Discovery Macros, optional eligibility
  and cross-source verification requirements, and how a real YAGO trail
  is grouped into 1–2 Macros. **Only then** does it match complete,
  distinct real candidate paths directly from the YAGO index, and
  optional actual secondary YAGO fork facts. The two reused legacy
  functions are pure fact samplers; no earlier `make_spec()` is called.
  The authoritative DAG and relation requirements live together in
  **one** `macro_plan.macro_tasks` array.
- `task_plan_runtime.py` materializes real witnessed RDF excerpts and
  explicitly fictional project/event/source evidence, compiles those
  tasks into the **original** source-scoped Program IR, and performs
  bounded executable information-necessity interventions. It does not
  add a second execution engine; the old Compiler, Macro WorkProducts,
  FactRefs, Question Author/Critic, strict Replay and Stage-2 output
  contracts are retained.

Supported *first vertical slice*: direct project reference tracing,
a genuinely conditional event→entity lookup, and a two-claim-source
association followed by genuine YAGO fork verification. The resulting
Macro DAGs have **variable sizes, with a hard cap of nine**, rather than
enforcing nine for all tasks. These are still constrained task families,
not arbitrary compositions or all possible human research goals.

New specs do **not** contain `task_first_v1_backbone`,
`hybrid_backbone_v3` or any other embedded historical ProblemSpec.
The output path and serialized version are distinct to keep earlier
v1–v4 and Graph-first v1/v2 Replay reproducible; frozen historical
code paths are retained as compatibility handlers, not used to
generate new task banks.

Source-level and information-level audit checks remain separate:
the new `single_task_plan_counterfactual_v1` policy tests each
identifying clue, Discovery removal, event-status/year eligibility,
and (when present) the independent candidate directories and real YAGO
fork under controlled re-planning. An empty or blocked path is **not**
counted as proof of necessity; these are bounded World interventions,
not proof of globally shortest agent plans. All fictional evidence
remains simulated, while every stated genuine RDF edge is drawn from
the supplied YAGO corpus.

The quality report includes
`knowledge_world_grounding.single_task_plan` with graph mode,
number of Macro tasks, replay/necessity pass counts and a check that
the new Spec embeds no legacy backbones.


### Single Task Plan — information-substitution audit and structural calibration

The current `task_plan` generator now reuses the **Knowledge v2 shadow-case
principle** (every required identifying cue has a distractor with a COMPLETE
real YAGO answer path), plus executable filter/source counterfactual
replanning. However, those older checks alone did NOT catch a visible
`event_name = "Qualified..."` shortcut, which could replace the intended
`event_status` or `event_year` predicate.

Newly generated plans therefore freeze three narrow policy fields
**inside the existing single `macro_plan`**; there is no new generator,
backbone, Spec version, or secondary evidence framework:

- `evidence_sampling_policy=eligibility_blind_event_names_v1`: qualifying
  and nonqualifying events have the SAME neutral title within their
  fictional project. A separate `event_id` still identifies each row
  for evidence lookup, and the task's specified status/year is the only
  eligibility discriminator (the inactive field is constant).
- `difficulty_sampling_policy=calibrated_structural_difficulty_v1`:
  Medium samples shorter and simpler reference/event investigations;
  Hard samples deeper event/real-fork verification work. Selection uses
  blockwise deterministic balancing rather than correlating a mode with
  the alternating difficulty schedule. Reports explicitly describe
  **structural proxies**, not empirically measured agent hardness.
- `question_sampling_policy=grounded_relation_goal_question_v1`:
  the Question describes a human-readable relationship goal derived
  from genuine YAGO predicate witnesses rather than printing a raw
  predicate sequence or exposing Macro IDs.

`task_plan_runtime.audit_information()` additionally tries to
SUBSTITUTE the necessary event eligibility filter with `event_name` or
the *inactive* event condition field, testing every possible value for
the correctly identified fictional project. The original Gold Program
is re-executed. **Every** such alternative must leave multiple
full, executable real YAGO answers; a single-answer shortcut rejects
the scene. Deliberately injecting `Qualified`/`Unqualified` titles
back into a scene is a regression negative control and must fail.

Opaque IDs and genuine motif objects are not counted as semantic
eligibility descriptors. This is a bounded same-World substitution
audit, not a proof that arbitrary external information cannot imply
eligibility.

Important replay compatibility: **older** `task_plan` Scene Specs
without these three policy fields retain their original deterministic
sampler, event titles, Question rendering and Quality Audit report
shape. Existing complete Scene banks are intentionally protected by
the default `overwrite: false` setting. For a fresh production pilot
choose a new output directory (or explicitly opt into overwrite).

The quality report's
`knowledge_world_grounding.single_task_plan` section now adds
`non_target_field_substitution_passed`, `eligibility_cases`,
`calibrated_difficulty_profiles` and `question_topic_distribution`.

### Current Single Task Plan: Goal-constrained semantic YAGO grounding

Newly generated scenes use the **existing** `task_plan` family and
`macro_plan.macro_tasks`, rather than adding a new full generator or
building further legacy Specs. A narrow frozen policy inside the current
Task Plan, `semantic_grounding_policy: goal_constrained_yago_grounding_v1`,
switches on *interpretable relation-goal sampling*:

- The same typed Macro work is selected first. Before selecting any actual
  YAGO subject or object, the Grounder chooses a candidate **local relation
  composition with an intelligible user information goal** (e.g. someone's
  doctoral advisor's research field, or the administrative region
  containing a recorded birthplace). `location -> follows`,
  `location -> namedAfter -> location` and unsupported compositions are
  rejected rather than made plausible through invented prose.
- The existing pure YAGO sampler produces **complete, distinct real paths
  for ALL Shadow Cases**, with a new opt-in candidate filter. Old
  `typed_knowledge_v1/v2` callers do not pass that filter; their
  frozen generation and Replay behavior are unchanged.
- For the target **and every competing candidate**, each subject/predicate
  pair must have **exactly one object in the loaded bounded YAGO graph**.
  This prevents an artificial singleton created only by truncating the
  source rows. The check does NOT establish uniqueness in the complete
  global YAGO corpus, which may contain facts beyond the configured
  `knowledge_max_triples` /`knowledge_max_edges` limits.
- The selected goal is stored **only once** under the existing
  `macro_plan.goal.semantic`. It includes the witnessed predicate
  sequence, interpretable question, source domain and declared answer
  scope. No new standalone Relation Sketch/World/Spec is introduced.
  Type hints are inferred from conventional predicate semantics, not
  proven by `rdf:type`: the current streaming fact loader excludes
  RDF type assertions. Do not present inferred person/place type as
  externally verified ontology evidence.
- The deterministic reference Question presents that whole goal in
  context rather than incrementally printing raw RDF predicates. The
  existing Author/Critic can optionally rewrite it; source clues and
  eligible-event conditions must be preserved and the goal's relation
  concepts must remain expressed.
- Fictional source titles and neutral meeting names are chosen from
  academic/biographical/institutional/geographical archive vocabularies.
  They do not add historical facts or reveal which event qualifies.

**No new pipeline:** generation, primitive Gold IR, Macro WorkProducts,
FactRefs, original source/field/Shadow-Case counterfactuals and strict
Replay are still the same as in the original Single Task Plan.

The quality report now records
`knowledge_world_grounding.single_task_plan.semantic_grounding` with
goal-profile/domain distributions, count of loaded-graph singleton witnesses
and the number of attempted semantic paths in *successful* scenes. These
are conditional sampling attempts, not a global semantic validity rate for
every edge in the YAGO corpus.

```bash
python -m environment generate-questions \
  --config environment/configs/questions_task_plan.yaml \
  --output-dir environment/data/stage1_semantic_goal_pilot
```

The new policy is used **only for new scenes**: prior frozen
`task_plan` scenes without `semantic_grounding_policy` still replay with
the old path choice, source labels, question text and audit structure.
The current semantic grammar covers a deliberate subset of 2/3-hop
relation combinations; a sparse source sample may yield no accepted
combination and should be reported as such rather than filled with
fabricated RDF witnesses. Real image/audio/video processing remains
out of scope; the corresponding observations are explicitly simulated.


### Task Plan: provenance-safe fictional semantic worlds

The semantic Task Plan first searches for a complete interpretable YAGO
relationship, including all necessary shadow candidates and a witnessed fork
when required. When the bounded RDF sample cannot support that, the *same*
Task Plan instead constructs deterministic **fictional** entity relationships
with plausible semantic signatures. It does not label any synthetic relation
as a real RDF/YAGO fact, and publishes synthetic provenance in the private
witness, World observations, questions/intent, and quality report.

Synthetic scenes still require complete counterfactual candidates, qualifying
event fields when applicable, a separately documented simulated verification
link for \`verify\` mode, exact Gold replay and information-necessity audits.
The internal \`real_fork\` identifier is kept for old executor contracts,
but its provenance and natural-language source title are explicitly fictional.

The candidate loop now uses distinct Task Plan ordinals for rejected attempts
and permits bounded replacements while keeping the accepted difficulty mix.
Older frozen scenes retain their original replay semantics; this is a
simulation-first orchestration benchmark, not historical or real multimodal QA.


### Task Plan: YAGO motif composition (recommended pilot)

\`environment/configs/questions_task_plan.yaml\` selects
\`generation.knowledge_world_mode: compose\`. The default Python
\`task_plan.make_spec(...)\` behavior remains \`witnessed\` to preserve older
YAGO-grounded scenes and Replay. The original Gold Executor, Macro tasks,
Stage-2 evidence realization, and Stage-3/4 interfaces are unchanged.

In \`compose\` mode, Stage 1:

1. Samples the frozen **Macro Task Plan**, including discovery, eligibility,
   association, verification and the typed handoff contracts.
2. Extracts cached **YAGO relation motifs** from the bounded graph:
   observed predicate frequencies, short joined-relation signatures,
   outgoing-fork prevalence and safe generic research-field vocabulary.
   Schema-typed predicate roles are heuristics, not claimed RDF type proofs.
3. Chooses an interpretable 2- or 3-hop goal and composes entirely **new
   fictional entities and typed relations**. No original YAGO entity
   relationship is reassigned or asserted as a real RDF fact. The private
   canonical \`world_binding.world_model\` owns entity identities and relation
   assertions; \`candidate_paths\` is a compatibility projection, not a second
   source of truth.
4. Materializes the required evidence sources and optional secondary registry
   fork **from the same canonical World**. The original Gold Program,
   counterfactual necessity audits, and Replay must all pass.
5. Presents an **ordinary archival investigation** to the Main/Sub-Agent:
   the generated question, source titles, initial public view and readable
   source observations do not expose the generator's \`synthetic\` provenance
   markers. Those markers remain in teacher/private witness, World metadata
   and audit reports; the dataset itself must be documented as synthetic.

To use the original strict YAGO-path scheme, set
\`knowledge_world_mode: witnessed\`. Composed-world quality is not measured by
the number of original YAGO fact paths; audit
\`knowledge_world_grounding.single_task_plan.semantic_grounding\` for
\`composed_motif_worlds\`, real motif support and Replay / necessity passes.

Note: media remain generated structured observations (not decoded images or
videos), and the current Task Plan still uses three bounded Macro patterns;
these are future realism/diversity targets, not completed features.


### Shared-world evidence and required semantic clues (compose mode)

Composed Worlds keep the canonical \`world_model.context_relations\` and
\`context_by_case\`, with institution/volume and site entities shared between
dossiers. The second source of the **first existing discovery Macro** is
now a *required* association: \`affiliation\` for a person's record, or
\`subjectOf\` for a regional survey entry. The target and other candidate
shadows share an institution/volume, while one designated shadow and
background decoys belong to another.

The existing scan → filter → join Macro reads this relation from the
**authorized register**; removing its filter must introduce competing final
answers. It does not require a new planner, Macro type, Gold operator or
Stage-2/3/4 interface. Other contextual edges remain non-required background.

Earlier composed scenes emitted \`document_context\` and \`record_excerpt\`
on *every* observation. Repeating per-dossier affiliation across unrelated
sources both bloated the records and gave the agent a shortcut around the
required relation source. New scenes omit these redundant fields: each
fact appears in its authorized source only. Older frozen World specs without
\`required_context_relation\` retain their original Replay behavior.

Borrowed YAGO \`fieldOfWork\` labels are still gated as academic-field
vocabulary, with a safe fallback when the bounded sample lacks appropriate
examples. Semantic validation is lexical and typed, not an RDF type-proof.
Question openings vary while retaining the exact frozen clue and goal.

The quality report includes \`required_shared_relation_worlds\`,
\`required_shared_relation_necessity_passed\`,
\`source_records_with_context\`, \`shared_context_entity_groups\`, and
\`excluded_yago_nonacademic_field_labels\`. The new source filter's
counterfactual necessity is checked separately from merely blocking source
access. The current design still does not claim new Macro Graph skeletons,
true image/audio perception, or measured live-Agent difficulty.
