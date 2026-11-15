# Graph-grounded Macro-first v3 pilot

An **isolated Stage-1 v3 prototype**. Existing v2 `generate-questions`,
`implement-subtasks`, main-turn generation, reward audit and strict Replay
are deliberately unchanged. It is **not** a production Stage-2 scene bank yet.

## Inputs and installation

Use the existing local YAGO 4.6 dump at `data/world-source/YAGO-4.6`.
Official YAGO 4.6 facts are Turtle files, often packaged as
`yago-4.6-facts.zip`; this parser also accepts `.ttl`, `.ttl.gz`,
`.nt` and `.nt.gz`. The loader selects **facts**, not labels/schema or
RDF-star meta logs, and streams a bounded sample into memory.

```bash
pip install 'pyoxigraph>=0.5,<0.6'
python -m environment.macro_v3 \
  --yago data/world-source/YAGO-4.6 \
  --output-dir environment/data/stage1_v3_pilot \
  --target 100 --seed 17 --max-triples 1000000
```

The v3 command is intentionally a separate Python module so the
existing `python -m environment` command surface and all v2 users stay unchanged.

If YAGO is very sparse in the first N parsed statements, increase
`--max-triples` or pass a curated Turtle facts file to `--yago`.
The full 167M-fact dump is **not** loaded in RAM.

## Output

`drafts/scenes.jsonl`: each line contains a typed abstract Macro plan, a
synthetic observation World with actual sampled YAGO predicate edges, scoped
Macro feedback and internal steps, a reproducible Gold Answer, a deterministic
Question **draft**, a dependency wave schedule and an audit.

`summary.json`: counts and explicit `stage2_compatible: false`.

## What this prototype does and does not prove

- Programmatic Macro-first structure: 2–4 independent clue investigations,
  a typed multi-input identification Macro, 1–3 genuine sampled YAGO relation
  investigations, and an optional independently sourced simulated review with
  a final synthesis. The verification branch creates a proper fork/join,
  rather than a meaningless cross-layer edge.
- Each information Macro may contain observation/search/lookup/compare steps.
  It is *not* one primitive operator per Macro.
- Each clue branch is counterfactually necessary: omit any branch and at least
  one distractor remains. The full intersection selects a unique entity.
- Synthetic records and YAGO facts retain distinct provenance. Image and
  other modalities remain **structured simulated observations**, never
  decoded media.
- A replay checks the frozen world/plan/answer and Macro WorkProducts.
- The scene has a **draft** question. It has not passed existing v2
  Author/Critic; do not pass these JSONL files directly to Stage 2.
- Real graph completeness, grounded independent fact sources, and full
  v3 compatibility with the existing Stage-2/3/4 `realize_scene` contract
  are pending an adapter. This prototype does not claim otherwise.

## Tests

```bash
python -m unittest discover -s environment/macro_v3/tests -v
```

## V3.1: coarse-grained Macro investigations (opt-in)

The original `--variant v3` behavior is **unchanged**. The new
`--variant v3.1` implements the next research iteration:

```bash
python -m environment.macro_v3 \
  --variant v3.1 \
  --yago data/world-source/YAGO-4.6 \
  --output-dir environment/data/stage1_v31_pilot \
  --target 20 --seed 17 \
  --max-triples 1000000 --max-edges 250000 \
  --min-hops 1 --max-hops 3
```

- **Coarse Macro boundary:** 2–3 investigation Macros each combine TWO
  evidence sources (image/audio/video annotation + text/table, all simulated).
  Identification reconciles their candidate WorkProducts. One relation
  investigation handles an entire sampled YAGO path, rather than one Macro
  per graph edge. Optional review uses **two independent fictional
  assessments plus a separate adjudication record**, followed by synthesis.
- **Typed structural DAG first:** investigation branches may be independent
  or progress through candidate handoffs. The identity Macro only depends
  on maximal upstream branches; the review branch genuinely forks from
  identity while YAGO-relation research proceeds in parallel.
- **Scenario grounding:** sampled visual/audio metadata describe fictional
  archive *dossiers*, not impossible colors or shapes attributed to a YAGO
  person. Fictional case-to-real-entity crosswalks are labeled synthetic.
  Research notes request exact predicates and the real RDF path is copied
  unchanged into separately tagged YAGO evidence sources.
- **Real relationship depth:** uses only connected YAGO edges for 1–3
  relations; multi-hop walks are preferred when available. Exact
  `--min-hops 2` will fail rather than invent a YAGO connection if the
  retained edge sample contains no such path.
- **Question styles:** eight contract-based drafts across several scenario
  frames; still **NOT** LLM-reviewed or suitable for production SFT/RL.
- **Audits:** all individual clue sources are counterfactually necessary;
  tasks have topological typed handoffs and source-scoped deterministic
  evidence traces. The replay hash locks Macro, World, Answer and Question.

### YAGO sampling notes

`--max-triples` and `--max-edges` are separate limits: whichever is hit
first stops the stream. In the earlier 20-task sample, `retained_edges=70000`
caused parsing to stop after 205,413 triples even with
`--max-triples 1000000`. Increase `--max-edges` to inspect more facts.
Do not interpret `real_yago_hops` as number of independent data sources.

### Compatibility boundary

The output location is
`environment/data/stage1_v31_pilot/drafts/scenes.jsonl`. Like v3, it is
**not** a Stage-2 compatible `questions/scenes.jsonl` bank. No existing
v2/v3 scenes, tests, or runtime Main-Agent interfaces are changed.


## V3.2: Programmatic Typed Macro DAG (opt-in)

This is the next **Macro-first** pilot and deliberately keeps `v3` and
`v3.1` replay semantics unchanged.

```bash
python -m environment.macro_v3 \
  --variant v3.2 \
  --yago data/world-source/YAGO-4.6 \
  --output-dir environment/data/stage1_v32_pilot \
  --target 20 --seed 17 \
  --max-triples 1000000 --max-edges 250000 \
  --min-hops 2 --max-hops 3 \
  --max-macros 12 --max-parallel 3 --max-waves 7
```

### Implementation

- `role_grammar.py`: six coarse **information-goal** roles:
  Discovery / Identification / Investigation / Verification /
  Comparison / Synthesis, with typed input/output ports.
- `graph_generator.py`: samples role graph, permitted transitions,
  branch/wave concurrency, progressive discoveries, conditional followups,
  comparisons, optional verification and *non-fake* long-range evidence edges
  BEFORE reading any YAGO data or selecting a scene domain.
- `graph_validator.py`: rejects cycles, forward dependencies,
  fabricated `depends_on`, invalid input/output types, orphan tasks,
  over-budget graphs and non-causal edges.
- `typed_world.py`: maps the frozen abstract graph onto actual YAGO
  relationship paths and independent **fictional** evidence bundles;
  each discovery owns two sources. The source-specific original witness
  is retained in its own port; Identity does not silently carry it.
- `typed_scene.py`: executes each Macro with **only its declared incoming
  ports and local sources**, builds exact provenance-bearing Trace,
  freezes Question draft and Gold Result, and replays from frozen inputs.

The typed skeleton can create direct Wave-1-to-final evidence handoffs
because `Synthesis.original_evidence` really consumes
`Discovery.evidence`. A blind extra edge is not allowed.

### What remains experimental

The Question is still a programmatic **draft**, and the multimodal sources
are *simulated observations* (not decoded actual media). V3.2 scenes are
**not Stage-2-compatible** and must not be fed to production training. The
typed sampler uses a bounded, extensible information-goal grammar rather
than mathematically arbitrary DAGs. The World/Question naturalization
pipeline and exact Stage-2/3/4 schema adapter remain separate work.

### Tests

```bash
python -m unittest discover -s environment/macro_v3/tests -p 'test_*.py' -v
```


## Canonical generator migration

The typed Role Grammar, partial-order DAG sampler, graph validator and
port-level goal/necessity audits are now maintained in
`environment/generate_question/`. Files of the same name under this
experimental pilot are compatibility imports, not separate implementations.
The v3.2 CLI and historical task versions remain usable. Old production
`generate-questions` still uses the original executable frozen-v2
compiler and existing strict quality gates, pending a versioned typed
World/Program compiler adapter.
