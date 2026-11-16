# Stage 1 — Task Plan + Question Author/Critic

The maintained data-generation route creates one executable Macro Task Plan,
composes a canonical synthetic World with source-scoped evidence, compiles
Gold/Replay and information-necessity audits, then **by default** uses a
local Question LLM Author and Critic to publish user-facing questions.

## Default 100-candidate pilot

From the repository root with the OpenAI-compatible Qwen service listening at
`http://127.0.0.1:8807/v1`:

```bash
python -m environment generate-questions \
  --config environment/configs/questions_task_plan.yaml \
  --output-dir environment/data/stage1_question_llm_100
```

The YAML defaults to 100 fixed candidates, difficulty mix 0/50/50
(Easy/Medium/Hard), `task_plan + compose`, Qwen3.5-9B Question Author
and a **separate Critic invocation using the same configured model**.
Scenario creation remains deterministic. For an offline test without a
model endpoint, explicitly set `question.model` and
`question.api_base` (and Critic) to null in a copied YAML.

### Output layout

```text
stage1_question_llm_100/
  constructed/
    scenes.jsonl      # frozen, deterministic task baseline
    questions.jsonl
    summary.json
  questions/
    scenes.jsonl      # ONLY Author+Critic accepted scenes; use downstream
    questions.jsonl
    failed.jsonl      # rejected/fallback cases with review diagnostics
    summary.json
    question_authoring_progress.json
  quality_report.json
  summary.json        # authoritative final_scenes and acceptance counts
```

`constructed/scenes.jsonl` represents the original fixed tasks.
`questions/scenes.jsonl` is the accepted, reviewed subset; it is the
**default Stage-2/3/4 input**. Do not use `constructed/scenes.jsonl`
for a production reviewed-Question experiment or assume the original
100 candidates must all pass. A candidate failing its Critic/Guard after
the configured retries is logged in `questions/failed.jsonl`, not
silently published as reviewed. The journal can resume an interrupted
Question phase with the same model/contract/code signature; use a new
output directory for changed settings.

## Frozen components versus editable surfaces

| Artifact | Frozen? | Initial Main visibility |
|---|---|---|
| Macro DAG, goal and dependencies | Yes | No |
| Canonical World and evidence facts | Yes | Only available **source descriptions**, not contents |
| Gold Program, Trace and Answer | Yes | No |
| Provided Reference Card v2 | Yes | Yes, as already supplied *textual simulated observations* |
| User Question | **LLM rewriting only** | Yes, after review |

The Author receives a compact semantic Goal, required event condition,
topic, reference-card modality summary and style request. It must express
the Goal as an ordinary user question without copying underlying evidence
values, traversals, private IDs, source titles or internal DAG steps. Its
output cannot modify World, Gold or evidence. The Critic receives the
frozen Goal, ordered relationship semantics, answer type, event condition,
frozen reference question and candidate rewrite. Deterministic guards
reject unsupported semantics or missing requirements; strict Replay
validates the final Scene again.

The **Initial Task Inputs** are descriptions of image/audio/video/table/text
observations already supplied by the user, not actual media files.
The **Available Evidence Sources** are still-unqueried simulated archives,
including browse and keyed-lookup contracts. Stage 3 renders these as
separate sections rather than advertising all sources as attached media.

## Important invariants

- The candidate count is fixed; rejected Question rewrites do not trigger
  additional Task Plan generation to hit an acceptance quota.
- Gold Replay validates the original and reviewed Scene; each accepted
  Question must preserve the original World, Macro Graph, Program and Answer.
- Information necessity includes full-vs-ablated target disambiguation,
  event-key propagation and source-scoped relation checks.
- Deduplication uses **(Question, supplied reference observation content)**,
  not Question alone or an arbitrary reference ID.
- `question_authoring.status=accepted` is required for the reviewed output.
  Deterministic offline Scene banks without Question LLM remain valid for
  their explicit, separate testing workflow.
- All user-visible Question acceptance claims reflect Critic judgments
  and program checks; they are not formal natural-language equivalence proofs.

## Inspect the 100-task pilot

```bash
python - <<'PY'
import json
from pathlib import Path

root = Path("environment/data/stage1_question_llm_100")
summary = json.loads((root / "summary.json").read_text())
review = json.loads((root / "questions/summary.json").read_text())
print("Candidate count:", summary["candidate_count"])
print("Accepted questions:", summary["accepted"])
print("Rejected questions:", summary["rejected"])
print("Review statuses:", review["question_results"])
print("Final Scene bank:", summary["final_scenes"])

for line in (root / "questions/scenes.jsonl").open():
    scene = json.loads(line)
    print(scene["task_id"], scene["question"])
PY
```

Once the reviewed 100-task output has been inspected, test Stage 2 using
`--scenes environment/data/stage1_question_llm_100/questions/scenes.jsonl`
and a **new** Stage-2 output directory, then Stage 3 on those same scenes.
Stage 4 independently builds its trusted history from Stages 1 and 2.

Historical generator families remain test fixtures; this is the sole
maintained production Stage-1 configuration profile.
