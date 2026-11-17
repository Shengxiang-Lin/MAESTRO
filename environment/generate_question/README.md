# Stage 1 — Task Plan + Question Author/Critic

The maintained data-generation route creates one executable Macro Task Plan,
composes a canonical synthetic World with source-scoped evidence, compiles
Gold/Replay and information-necessity audits, then **by default** uses a
local Question LLM Author and Critic to publish user-facing questions.

## Goal-conditioned composition (v2)

The default configuration selects `task_plan_composer: goal_conditioned_evidence_needs_v3`.
V3 samples information obligations (optional event eligibility and independent
corroboration), checks an independently sampled source inventory, and selects
only typed source producers satisfying the exact need. Each clue source's
browse/keyed access is decided before building its Macro dependency.
Existing `goal_conditioned_typed_v2` frozen Scenes replay unchanged.

The semantic relationship Goal is chosen before the information acquisition
plan. Supported source-scoped producers then form a bounded typed Macro DAG:
clue investigations may be independent or may require upstream `case_id`
keys; relationship hops may be grouped into one or several Macro WorkProducts.
Only downstream source views with a genuine upstream key become
`keyed_lookup` sources. The legacy composer remains available by setting
`task_plan_composer: legacy`, and earlier frozen Scene banks replay unchanged.

V2 intentionally retains the existing one-string, exactly-one answer contract,
three tested entity-acquisition producers, World/Gold IR and Stage 2–4
interfaces. It does **not** claim arbitrary DAG planning or new answer types.

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

### Opt-in typed Goal v4: patterns abstracted from the 60-question pretest

Run `--config environment/configs/questions_typed_goals_v4.yaml` to
enable an **opt-in** Stage-1 pilot. The stable
`questions_task_plan.yaml` continues to use v3 unchanged.

The reference split is the actual repository
`data/OmniGAIA/splits/pretest_60_ids.txt` (IDs beginning 6, 8, 10,
20 and ending 354, 355, 358, 359), **not** the older, incompatible
list beginning 5, 12, 20, 28. We manually reviewed the official
question/answer formats and grouped each of the 60 by its primary
Goal. As an exploratory, *mutually-exclusive* description (not an
official OmniGAIA taxonomy):

| Main outcome of the actual selected 60 | Count |
|---|---:|
| Formula, ratio, numerical derivation | 20 |
| Structured multi-field answer | 16 |
| Date/temporal derivation | 9 |
| Entity/attribute retrieval | 7 |
| Geospatial distance | 5 |
| Evidence judgment/explanation | 2 |
| Counting/aggregation | 1 |

The table combines final output format and operation type only to
describe benchmark coverage. The **implemented** v4 distinguishes the
primitive computation (difference, ratio, date gap, great-circle
distance, higher-of-two, group-count) from a separately modeled
structured answer. It smoothed sampling weights to keep rare Goals
testable on 50–100 synthetic questions. Cross-source explanatory
judgment is *not* yet a fully general v4 operation.

V4's relationship resolver is retained to provide a unique typed
entity key. A non-entity Goal then acquires independently attributed
synthetic observations keyed by that entity and runs the actual
arithmetic/date/geographic/aggregate/structured operation in Gold IR.
The two-source cases introduce real parallel branches and a fusion
barrier. Scene settings vary with the semantic entity domain. Image,
video and audio remain **simulated textual observations**.

The underlying benchmark questions, answer values, media and external
research results are **not included** in synthetic World facts, Author
prompts, or output labels. Synthetic-vs-real benchmark accuracy is not
established by passing Replay. See the `goal_kind_distribution` and
`typed_goal_synthesis_operations` counters in
`quality_report.json`. Future iterations should enrich media clue
dependencies and source capabilities after confirming Goal-level
execution correctness.


### Goal operation DAG × answer output (v5, opt-in)

`environment/configs/questions_goal_ops_v5.yaml` enables
`goal_conditioned_operation_dag_v5`. Stable v3 and v4 generation configs
and frozen datasets remain unchanged.

Based on the **operation patterns** in the explicitly selected OmniGAIA
pretest 60 IDs stored in `data/OmniGAIA/splits/pretest_60_ids.txt`,
v5 models two independent axes:

- **Operations**: `lookup`, `arithmetic`, `temporal_difference`,
  `temporal_midpoint`, `geographic_distance`,
  `count_aggregate`, `compare_select`, `verify`.
  Each node is typed and topologically depends on previous outputs.
- **Output contract**: `single_entity`, `single_number`, `date_time`,
  `structured_multi_field`, `evidence_based_explanation`.
  `output_spec.fields` declares each field's type;
  `output_spec.bindings` selects its producer in the operation graph.

This is a **bounded compositional grammar**, not a random concatenation of
operators. One source capability (entity relationships, two count tables,
two dated records, two coordinate records, or a dated observation ledger)
is selected; compatible operations and an independent output schema are
sampled thereafter. In multi-field cases the same acquired evidence can
be used for multiple computations (for example **difference AND ratio**,
or **date gap AND midpoint**). Gold IR executes each operation and formats
the typed answer, rather than treating `structured_answer` as an opaque
single operation. The Macro DAG acquires evidence before the synthesis
barrier; a bounded number of computations run inside the synthesis Macro.

`evidence_based_explanation` here is a **structured, source-grounded
assessment** containing a comparison conclusion and both recorded counts;
the complete source fact references remain in WorkProduct provenance.
It is **not** unrestricted natural-language explanatory reasoning or
fully general verification of contradictory real-world statements.

The reference 60 questions inform goal coverage only; their specific
answers, descriptions and media are not used as synthetic labels.
Program correctness, counterfactual source sensitivity, Stage 2
realization and replay are checked by `test_goal_ops_v5.py`.
Report fields `goal_operation_distribution`,
`goal_output_distribution` and `goal_operation_output_pairs` distinguish
operator coverage from output coverage and Macro topology.

### Output-first v6: Goal fields drive partial-order operation planning

Run `--config environment/configs/questions_output_first_v6.yaml` for
the opt-in `goal_conditioned_output_first_v6` pilot. Legacy v3/v4/v5
configs and frozen Worlds stay unchanged.

**P0 — Question-Critic semantic mismatch fixed.** v5's Author already
received the frozen `goal_ops_v5.goal_question` and `output_spec`, but
its Critic incorrectly obtained the intermediate entity relationship
question via `goal_spec.kind` (absent from v5). Now both Author and
Critic consume the true final Goal and typed output fields. This
addresses a plausible major failure mode observed in v5's 64/100
Question-acceptance pilot; any improvement in LLM acceptance must be
measured by rerunning the same seeds, not assumed from unit tests.

**P1 — Output-first backward composition.**

1. Sample one of five answer structures. Choose typed *semantic* field
   demands (e.g. `duration_days: int`, `distance_km: int`,
   `entity: str`) rather than an arbitrary Profile category.
2. Recursively expand each demanded field to valid primitive operators:
   Lookup, Arithmetic, Temporal Difference/Midpoint, Geographic Distance,
   Count/Aggregate, Compare/Select and evidence-bounded Verify.
3. Intern identical lookup and intermediate operations. The resulting
   DAG has a topological dependency order; `operation_subgraph_leq`
   defines the *non-total* typed operation-subgraph partial order across
   different Goal DAGs (independent of output format).
4. Determine precisely which canonical source fields are needed. One
   independently keyed record may contain both dates and coordinates
   when both are necessary; absent requirements are not added.
5. Bind the synthetic canonical World, create dependency-correct Macro
   handoffs, compute group counts **before** joining scalar evidence,
   execute typed Goal operations in Gold IR, format the final output,
   and run the usual replay/information-necessity/counterfactual audits.

To accommodate at most three independent evidence retrieval Macros,
v6 alone allows up to **11** Macro tasks. This is an explicit
cross-domain budget, not a relaxation of Source necessity; prior versions
retain 9. v6's bounded generator still uses the original two- or
three-hop entity-relation substrate, and generates *simulated textual
observations* rather than consuming real image/audio/video assets.
The evidence-based explanation output is a structured comparison with
two supporting counts and cited fact references, not open-ended prose.

In `quality_report.json` the operation/output cross-distribution is
recorded separately from semantic fields and source capability mixes.
As always, the 60 OmniGAIA pretest questions motivate the abstract
operation coverage only; their original prompts/answers/media are not
copied into synthesized data.

Test coverage: `environment/tests/test_output_first_v6.py` checks
output-first coverage, typed DAG partial order, multi-domain source
sharing, canonical World, final answer, Replay, source-counterfactual
necessity, Stage 2 realization and mock Author/Critic alignment.
