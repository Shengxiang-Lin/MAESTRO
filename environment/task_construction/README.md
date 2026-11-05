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

## Web-first acquisition and native O1 tool boundaries

New frozen tasks identify one of three evidence scenarios:

- `web_search`: all sources are `text/search/table/database`. No candidate
  roster or first reference is initially given. The public question is the
  search entry point; all Source records are discoverable.
- `hybrid`: a genuine multimedia/reference clue may be initially
  attached/provided; subsequent sources are discovered via search.
- `provided_media`: media-only evidence retains an initial media input,
  rather than being falsely presented as an open-web search.

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
python environment/construct_tasks.py --reuse-source-priors
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
