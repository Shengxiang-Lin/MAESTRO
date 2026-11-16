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
