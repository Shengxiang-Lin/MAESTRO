# Stage 1: current output-first Task Plan only

`python -m environment generate-questions` is the single supported Stage 1
generator. It constructs synthetic tasks from bounded YAGO structural motifs:
**Goal → typed evidence and task dependencies → Macro DAG → fictional World →
Gold Program and WorkProducts → grounded Question**.

Input YAGO provides predicate vocabulary and relation witnesses; simulated
image/audio/video observations are text representations derived from synthetic
structured values. The generator does **not** read original OmniGAIA media or
claim to decode real media. The LLM may rewrite Question/Scenario prose only;
Goal, World, permissions, dependency graph, Gold and answer are frozen.

## Canonical layout

| Path | Responsibility |
| --- | --- |
| `pipeline.py` | Generation command, quality aggregation and reporting |
| `stages.py` | Bounded candidate construction, resumable LLM review and ordered journals |
| `planning/` | Output-first Goal, evidence/source plan, typed Macro DAG, YAGO index and current spec |
| `grounding/` | Synthetic clue/source representations and contextual media descriptions |
| `execution/` | World materialization, typed Program IR, scoped Macro execution and acquisition permissions |
| `authoring/` | Question/Scenario Author and Critic, prompts, semantics and surface guards, shared API backend |
| `auditing/` | Strict Replay, source necessity, evidence-grounding checks and scheduling-structure metrics |

The `generate_question/` root deliberately has only three Python files:
`__init__.py`, `pipeline.py` and `stages.py`. Do not add forwarding modules
for functions already owned by subpackages.

**Retired:** `macro_first_v2`, `typed_knowledge_v1/v2/v3/v4`,
`task_graph_first_v1/v2`, `hybrid_world*`, legacy approval constructors,
and their import-compatibility facades. There is no `legacy_stage1/`
generator. `spec_issues` rejects retired reasoning patterns. Old scene files
requiring those retired compilers are no longer supported for Replay; the
intentionally supported input is the current frozen `task_plan` format.

Some source-facing data structures (typed Program IR, Gold executor, source
acquisition traces) continue to serve Stage 2–4 and are retained in
`execution/` with their consumer imports updated. Removing a *generator*
does not remove the current downstream WorkProduct protocol.

## Default generation

```bash
export DASHSCOPE_API_KEY="your-key"
python -m environment generate-questions \
  --config environment/configs/questions.yaml \
  --output-dir environment/data/stage1
```

The canonical config selects `qwen3.8-flash` for Question Author/Critic
through `config/model_config.yaml`; Question workers default to **10**.
Scenario Author/Critic is optional and off by default.

`max_revisions: 2` means at most three candidate drafts per Question.
Accept/reject decisions do not rewrite the frozen World or Gold. Use a new
output directory between incompatible model/code settings; a completed
run cannot be silently resumed under changed signatures.

Expected files include:
- `constructed/scenes.jsonl`: deterministic frozen Scenes with reference Questions
- `questions/scenes.jsonl`: accepted LLM-reviewed Questions referencing identical World/Gold
- `quality_report.json`, `summary.json`: acceptance and structural/evidence audits

Stage 2–4 consume only accepted current-Task-Plan Scenes.

## Validation

```bash
python -m unittest discover -s environment/tests -p 'test_*.py'
python -m environment.tests.stage1_pilot \
  --target 100 --output-dir environment/data/stage1_offline_100
```

The offline pilot uses a small generated RDF fixture, tests strict Replay
and Goal/World/source dependency consistency, and makes **no** live LLM calls.
Live Flash quality, acceptance rate, API costs and latency require a
separate full run with real YAGO and configured credentials.

For subsequent stage integration, the current-only fixture and pipeline
tests exercise Stage 1 → Stage 2 WorkProducts → Stage 3 trusted trajectory →
Stage 4 untrusted recovery. A passing offline test alone is not evidence of
agent robustness in the full benchmark.

The `quality_report.json` scheduling metrics describe legal partial-order
freedom only; they are not claims that Main-Agent learned adaptive policies.
