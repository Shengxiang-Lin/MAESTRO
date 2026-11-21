# Stage 1 — executable Question and evidence-scene generation

The **only maintained** Stage-1 entrypoint is:

```bash
python -m environment generate-questions --config environment/configs/questions.yaml
python -m environment quality --config environment/configs/questions.yaml
```

The default configuration creates **500 frozen candidate ProblemSpecs**
using up to 2,000 candidate seeds, then independently reviews their natural
questions. Author/Critic rejection can reduce the count of published Scenes;
do not equate `generation.target` with LLM-reviewed output size.

## Construction

`Goal → typed evidence requirements → Macro DAG → synthetic World →
executable Program/WorkProducts → independently reviewed Question`

YAGO supplies structural motifs and relation vocabulary; the synthesized
World, Gold and source permissions are code-owned. Image/audio/video evidence
is a **textual synthetic observation**, not decoded external media. Question
Author/Critic can edit question language but cannot mutate evidence, Gold or
task dependencies.

| Module | Role |
| --- | --- |
| `planning/` | Goal and scoped evidence plans; typed Macro DAG |
| `grounding/` | source vocabulary, YAGO motifs, synthetic observations |
| `execution/` | exact World, Program IR, evidence source access and Replay |
| `authoring/` | optional Scenario Author and reviewed Question language |
| `auditing/` | strict Replay and evidence/graph-quality diagnostics |
| `stages.py` | ordered candidate construction and crash-safe review journals |
| `pipeline.py` | one supported generation pipeline and quality reporting |

## Outputs and acceptance

`stage1/constructed/scenes.jsonl` is the deterministic frozen corpus.
`stage1/questions/scenes.jsonl` contains **only accepted LLM-reviewed**
questions. Downstream Stages 2–4 must consume the latter. Check
`stage1/summary.json` and `stage1/quality_report.json` for acceptance,
rejection reasons, replay correctness and structural diversity.

The defaults use the local `local-qwen3.5-9b` OpenAI-compatible server at
`http://127.0.0.1:8807/v1`; Scenario Author/Critic is disabled. Ensure
`data/world-source/YAGO-4.6` is available. No DashScope key is required
for this local default.

An offline test can be run without the 500-scene LLM batch:

```bash
python -m unittest discover -s environment/tests -p 'test_*.py'
python -m environment.tests.stage1_pilot --target 100 \
  --output-dir environment/data/stage1_offline_100
```

Keep original Scene identifiers and provenance intact across later Stages.
No older generator variants or compatibility facades are maintained.
