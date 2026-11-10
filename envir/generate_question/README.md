# Stage 1 — Current generation modules

The authoritative CLI, artifact layout and running instructions are in
[../README.md](../README.md). The active entrypoint is
`python -m envir generate-questions`.

## Core source layout

| File | Responsibility |
| --- | --- |
| `spec.py` | Freeze typed user Goal and Source contracts |
| `macro_first_v2.py` | Generate current Macro-level DAG and task schemas |
| `macro_first_v2_world.py` | Synthesize typed evidence and deterministic witnesses |
| `information_necessity.py` | Verify the actual independent candidate sets/answers |
| `source_title_contract.py` | Freeze modality-accurate source titles |
| `scenario_fields.py` | Deterministic, audited local repair of presentation fields |
| `scenario.py` | Scenario Author/Critic and semantic validation |
| `semantic_evidence.py` | Applied Gold-Solver predicate and answer-slot contract |
| `question.py` | Question Author/Critic and conservative output guards |
| `compiler.py`, `program_ir.py` | Executable Gold Program and Macro WorkProducts |
| `evidence_tools.py` | Actual O1 tool registry, primitive acquisition policies and Replay validation |
| `difficulty_proxy.py`, `quality.py` | Structural workload features and strict Replay |
| `stages.py` | Durable per-round Scenario/Question stage execution |
| `quota_pipeline.py` | Accepted-level quotas, bounded retries and final publication |
| `pipeline.py` | Canonical API composition and read-only quality report |

### Web evidence paths (v2)

Fresh Macro-first V2 Scenes freeze one of three **source-acquisition**
routes for each text/table source, before World/Question realization:

| Route | Per-acquisition O1 tool sequence | When URL becomes available |
| --- | --- | --- |
| \`search_only\` | \`GoogleSearchAction\` | No page extraction needed |
| \`direct_extract\` | \`ExtractUrlContentAction\` | Known in the Main-visible source catalog |
| \`search_then_extract\` | \`GoogleSearchAction\` → \`ExtractUrlContentAction\` | Discovered by the first Sub-Agent search step |

The route is deterministic for the sampled Scene/source, and does not
change the private Gold Solver or the Macro DAG. A multi-tool sequence is
**inside the source acquisition**, not an additional Macro or Main round.
Stage 1 compiles each sequence into \`evidence_acquisition.tool_steps\`
with ordered \`depends_on\` edges. Macro \`required_tools\` remains the
deduplicated union over all its source-acquisition steps.

URLs in these offline scenarios use the reserved \`.example.invalid\` TLD:
they are explicitly synthetic, never browsed and never evidence about real
websites. Search → Extract supplies precisely its synthetic search-selected
URL as the Extract \`url\` parameter; the URL must not leak into Main's
initial prompt. Search-only snippets, page extraction and media observations
are simulated and carry the existing Oracle-supported FactRefs. Stage 2
stores this lineage in \`trace_summary.steps[].simulated_tool_calls\`,
each marked \`tool_executed: false\`; Stage 3 validates it before generating
Main turn labels.

The v2 marker is \`maestro.evidence_acquisition.o1.v2\`.
Historical v1 scenes remain supported under
\`maestro.evidence_acquisition.o1.v1\`. Existing Stage 1 / Stage 2
datasets should **not** be mixed with new-generation v2 scenes.

### O1 tools are frozen at the evidence-acquisition level

New Macro-first V2 scenes now freeze an \`evidence_access\` policy on
**each source** and an \`evidence_acquisition\` contract on **each primitive
\`scan\` or \`lookup_by_key\` node**. An ordered union of the primitive tool
names becomes each Macro's \`required_tools\`. Intermediate join/filter/
project reasoning nodes have **no automatic tool assignment**: in
particular, a source-free Macro is not automatically granted Python.

All six Action descriptions and parameter schemas are read from the
actual classes registered in \`benchmark/omnigaia/tools\`; they are not
redeclared as \`WorkspaceSearchAction\` / \`TableLookupAction\`.
Images/videos/audio map to their corresponding O1 media tools. Synthetic
text/table searches and upstream-key lookups map to
\`GoogleSearchAction\` by query (the O1 tool supports a \`query\` argument).
\`ExtractUrlContentAction\` requires an actual URL; it is intentionally
**not** assigned to invented \`workspace://\` identifiers. When future
sources contain real, valid HTTP(S) URLs, the access policy can select the
O1 URL extraction tool.

**No O1 tools are executed during dataset synthesis.** The contracts
describe what a real O1 Sub-Agent *would need*, and Stage 2 records
\`tool_executed: false\`. Stage 1 Replay validates the frozen source
policy, every lower-level acquisition and Macro tool aggregation.

Pre-contract Scene files stay replayable using their legacy schema;
Stage 3 has an explicitly marked compatibility mapping for them.
For a fully audited primitive tool contract, regenerate Stage 1
questions and Stage 2 reports into **new output directories**. Existing
Stage 3 run signatures cannot be reused with modified code.

### Invariants

1. Scene World, source permissions, Goal and Macro solver remain frozen
   before any LLM paraphrasing.
2. Source titles derive only from source ID, modality and domain. LLM
   Scenario output may repair formatting but not invent evidence.
3. Question text must request one answer slot and preserve all executed
   clue/time constraints; a Critic pass is not a formal equivalence proof.
4. Two independent multimodal branches must narrow genuinely distinct
   candidate sets; alternatives differ from the target's final answer.
5. A quota can be satisfied **only** by accepted, unique, Replay-valid
   questions with the requested difficulty label, subject to a hard
   candidate budget.

Legacy `clue_chain`, `evidence_graph` and Macro-first V1 modules are
retained exclusively for existing snapshot replay and regression tests;
they have no current public generation switches. Avoid adding a new
parallel pipeline or duplicate `README` section for each experiment.

For local regression tests:

```bash
python -m unittest discover -s envir/tests -v
```
