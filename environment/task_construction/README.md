# Task Construction

Deterministic/programmatic construction of synthetic orchestration tasks.

```text
source priors (KQA-Pro + YAGO + MultiModalQA)
        ↓
deep ProgramIR composition
        ↓
graph-grounded semantic binding
        ↓
executable symbolic world
        ↓
hardening + heterogeneous evidence allocation
        ↓
TaskDesign freeze
        ↓
P0 scenario validity
        ↓
P1 capability / progressive-discovery grounding
        ↓
frozen public authoring projection
```

The symbolic executor is authoritative for task truth. LLMs are not used in this package.

## Scenario-first acquisition and native O1 tool boundaries

Newly constructed Worlds choose an **acquisition scenario before sampling Source
modalities**. Default proposal weights are `web_search:45`,
`hybrid:40`, `provided_media:15`; these are not promised acceptance
percentages. A web-search Source can be text, image, video, audio, table or
database: *modality does not determine whether the user supplied it*.
World metadata and the accepted-world summary record the actual scenario.
Legacy Worlds without an acquisition plan retain source-type inference.

New frozen tasks identify one of three evidence scenarios:

- `web_search`: **all sources are discoverable**, including
  search-retrievable videos or images. No candidate roster or first reference
  is initially given. The public question itself is the search entry point.
- `hybrid`: at least one media Source and one text/structured Source; a
  genuine media input is initially attached and other sources are searched.
- `provided_media`: media-centric evidence with an initial media attachment;
  other media sources may require further discovery and inspection.

The resulting capability topology keeps source modality, visibility,
`search_then_inspect` or `provided_input` access mode, and O1-compatible
tool families distinct. All six native O1 tools remain globally available;
the Main agent selects a relevant subset for each Sub-Agent.

The `public_question` search bootstrap certifies only *potential* source
availability, not successful retrieval. Stage 3 independently indexes ALL
rendered public source records and verifies query-dependent discoverability
before accepting a task. Do not count this bootstrap as a real Google tool call
or as proof of real-world web discoverability. Wiki/YAGO-derived entity aliases
are synthetic, and the mock index is a bounded task-local evidence corpus.


Run from the repository root:

```bash
python environment/construct_tasks.py --reuse-source-priors --reuse-candidate-pool
```

Important outputs live under `environment/data/`, including `typed_worlds_*.jsonl`, `task_designs_*.jsonl`, `capability_worlds_*.jsonl`, `authoring_inputs_*.jsonl`, and `frozen_manifest_*.json`.

The construction policy follows the former `orchestra-o3/environment` production path, but the code is reorganized by responsibility rather than by historical P0/P1 naming.

The only production size control is `task_construction.candidate_pool_size`
(default: 2000) in `environment/config.yaml`. The program collects up to that
many semantically valid candidates and checks each one against the unchanged
world-quality and diversity gates. It freezes every passing world, without a
fixed accepted-world quota. `frozen_manifest_2000.json` records the actual
`frozen_count` (not necessarily 2000). `--reuse-source-priors` skips
the expensive source scan only after those priors have been prepared.

### Rebuild only evidence/Worlds from an existing validated candidate pool

With `environment/data/semantic_candidates_pool_2000.jsonl` already present,
do not rerun the 2000-candidate semantic binding pass:

```bash
python environment/construct_tasks.py \
  --candidate-pool-size 2000 \
  --reuse-source-priors \
  --reuse-candidate-pool
```

This re-runs World Generation, modality/scenario allocation, TaskDesign,
P0/P1 and Freeze. It **overwrites** previous `typed_worlds_2000.jsonl`,
frozen artifacts, and `build_summary.json`; back up the old files and
run Question Generation with `--restart-questions` or use a fresh output
directory. Do not combine `--reuse-candidate-pool` with
`--restart-candidates` or `--resume-downstream`.

Override the three proposal weights with `--web-search-weight`,
`--hybrid-weight`, and `--provided-media-weight`. They must be
nonnegative and have a positive total. Frozen task quality checks remain
strict; the zero-attachment exception is permitted **only for web_search
with a configured independent public query/retrieval policy**.
