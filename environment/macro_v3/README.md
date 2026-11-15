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

Use `python -m environment generate-macro-v3 --yago ...` as an equivalent
command after integrating the optional entrypoint.

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
