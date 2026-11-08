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

## Experimental YAGO subject-anchored candidate pilot

The legacy default `--construction-mode program_composition` is unchanged.
The opt-in `semantic_anchored` mode still composes KQA-Pro reasoning
fragments and uses the existing YAGO graph-witness binder and MultiModalQA
source-layout priors. Unlike the legacy binder, it:

1. Samples a REAL YAGO subject profile from the already prepared
   `yago_entity_profiles.json` bank. Only YAGO-observed predicates with
   sufficient catalog support qualify.
2. Prioritizes those predicates during graph-grounded type / relation /
   attribute binding, then requires at least **two** chosen predicates to
   originate from that same real profile **on one logical entity scope**.
   The original YAGO subject ID and facts are *not* copied into the
   synthetic user-visible task.
3. Tries alternative typed bindings per KQA proposal
   (`--anchor-tries`, default 6) and quarantines candidates that do not
   demonstrate this local joint support.
4. Retains the current executable synthetic World, MMQA evidence allocation,
   quality audits, Gold DAG, Stage 2A / 2B, and Stage 3 unchanged.

This v1 is **local semantic anchoring**, not an extraction of a complete
real YAGO subgraph, and it does **not** guarantee global topical coherence,
raw-media grounding, natural question wording, or higher end-to-end yield.
Those must be measured empirically.

Use a NEW candidate pool capacity, distinct from the existing 2000-world
run, so the latter is not overwritten. The following assumes that
`environment/data/yago_entity_profiles.json` and other source priors
already exist:

```bash
# First run a small pilot. Does not overwrite *_2000 files.
python environment/construct_tasks.py \
  --candidate-pool-size 30 \
  --construction-mode semantic_anchored \
  --anchor-tries 6 \
  --reuse-source-priors \
  --candidate-workers 4 \
  --candidate-max-attempts 2400
```

The run uses `environment/data/semantic_candidates_pool_30.jsonl`,
`typed_worlds_30.jsonl`, `frozen_manifest_30.json`,
`capability_worlds_30.jsonl` and `authoring_inputs_30.jsonl`.
If *any* of these `*_30` files already belong to another experiment,
back them up first. Check `environment/data/build_summary.json` for
`candidate_selection.semantic_anchor_count`,
`candidate_selection.rejection_reasons`,
`worlds.semantic_anchor_worlds`, and `worlds.rejection_reasons`.
The checkpoint signature includes the YAGO profile content, mode and
anchor-tries setting. Changing these requires `--restart-candidates`
and a fresh output workspace.

Stage 2A / 2B can use the new frozen pool without editing
`environment/config.yaml` by overriding all input paths:

```bash
python environment/generate_goals.py \
  --manifest environment/data/frozen_manifest_30.json \
  --authoring-inputs environment/data/authoring_inputs_30.jsonl \
  --worlds environment/data/typed_worlds_30.jsonl \
  --capabilities environment/data/capability_worlds_30.jsonl \
  --output-dir environment/data/stage2a_anchor_30 \
  --max-records 30 --question-workers 4 --blueprint-workers 2 --flush-every 1

python environment/generate_questions.py \
  --manifest environment/data/frozen_manifest_30.json \
  --authoring-inputs environment/data/authoring_inputs_30.jsonl \
  --worlds environment/data/typed_worlds_30.jsonl \
  --capabilities environment/data/capability_worlds_30.jsonl \
  --blueprints environment/data/stage2a_anchor_30/certified_blueprints.jsonl \
  --question-brief-mode full \
  --output-dir environment/data/stage2b_anchor_full_30 \
  --max-records 30 --question-workers 4 --flush-every 1
```

For a controlled comparison, keep the same KQA files and seed, and run
`program_composition` in a **separate workspace** (do not reuse the
same `*_30` filenames). Compare candidate/world yields, signed question
rates, human-audited naturalness and semantic correctness, and the resulting
orchestration DAGs. This experimental mode is not a claim of true raw-media
perception: MultiModalQA currently informs *evidence source patterns*.
