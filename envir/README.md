# MAESTRO envir: Subtask-First experimental environment

A separate implementation under MAESTRO/envir/. The original environment/
pipeline is not changed or used for question/world construction. The only
direct code reuse is the tested low-level YAGO Turtle stream parser.

## Concept

KQA-Pro -> logical graph skeletons (operators/dependencies only).
YAGO-4.6 -> REAL fact triples -> searchable subject pages -> parameterized
            relation/attribute subtask templates.
MultiModalQA -> official QA annotations + supporting source IDs and available
                text/table/image assets, with explicit verification tiers.

Build a large SOURCE / FACT BANK; cache concrete YAGO subtasks only when a
TaskGraph needs them. Never pre-generate 100K summaries or fake evidence.
A task-local searcher ranks PUBLIC source documents using only the query.
YAGO answers are checked against observed facts, public page content, and
public search discovery before becoming source_replay_verified.

MultiModalQA question answers are dataset_labeled (NOT equivalent to YAGO
source-replay verification). Image captions alone are caption_proxy; an
available image path is file_available_unverified and still does not prove
raw image recognition. There is no claim of image/audio/video model execution.

Build TaskGraphs with actual typed/identity-consistent dependency bindings.
Initial supported graph shapes:
  chain             YAGO relation -> YAGO attribute/relation
  parallel          two factual lookups from distinct YAGO subject pages
  media_to_search   official MMQA labeled answer -> unique exact YAGO name
Use a KQA-Pro skeleton bank as prior for allowed patterns, not an assumed
equivalent KoPL full-program proof. Invalid joins are rejected.

Produce:
  bank.sqlite             source documents, original facts, KQA motifs,
                          labeled MMQA tasks, lazy verified YAGO task cache
  subtask_templates.json  compact parameterized YAGO task templates
  prepare_summary.json    ingestion statistics and evidence-tier policies
  gold_graphs.jsonl       PRIVATE: answers, tool events, summary traces, graph
  public_questions.jsonl  PUBLIC: question, provided material IDs, tool names
  compose_summary.json    yield, rejections, coverage/diversity statistics

The natural-language Question is currently a deterministic fallback generated
from the graph's final goal. QuestionBrief stores the necessary constraints
and explicit public anchors, not the complete factual trace or Gold answer.
The new environment does not yet perform online LLM question authoring,
conditional Main decisions, noisy subagent attacks or real multimodal
perception. Those are subsequent extensions, not implicit claims of v0.1.

## Run

From the MAESTRO root:

~~~bash
python -m pip install tqdm

python -m envir.cli prepare \
  --kqa data/world-source/KQA-Pro/train.json \
  --yago data/world-source/YAGO-4.6/yago-facts.ttl \
  --mmqa data/world-source/MultiModalQA/MMQA_train.jsonl.gz \
  --output-dir envir/data/pilot \
  --max-yago-triples 300000 \
  --max-yago-subjects 100000 \
  --max-kqa-records 10000 \
  --max-mmqa-records 2000

python -m envir.cli inspect \
  --bank envir/data/pilot/bank.sqlite

python -m envir.cli compose \
  --bank envir/data/pilot/bank.sqlite \
  --output-dir envir/data/pilot \
  --target 20 --max-attempts 300 \
  --patterns chain,parallel,media_to_search \
  --seed 17

python -m envir.cli search \
  --bank envir/data/pilot/bank.sqlite \
  --query "director film"

python -m unittest discover -s envir/tests -p 'test_*.py' -v
~~~

IMPORTANT: YAGO is very large. The prototype's bounded PREFIX scan
may give few usable connected edges if early triples are skewed, so the
CLI returns no_graphs/exit code 2 rather than inventing a valid graph.
If necessary, increase max-yago-triples or run a follow-up indexed
entity-neighborhood sampler; do not blindly assume bigger Subtask count
means more composability.

Graph checks reject cycles, missing/unused required subtasks, incorrect
bindings, missing public sources, broken YAGO replay, and incorrect
two-result composition. Parallel means two independent public lookups,
not necessarily a deep AND/OR query over two sets. The current KQA motif
selector is an approximate pattern prior.

## Next improvements and evaluation

1. Measure real dataset yields, especially overlap between MMQA entity names
   and YAGO subjects. Do not force a synthetic identity match when absent.
2. Build type-aware and diversity-aware candidate indices; avoid storing
   subject-centered pages beyond the budget while retaining fact consistency.
3. Map KQA operators to certified multi-subtask relation/filter/join programs.
4. Use genuine image files with independently verified perception actions,
   never present captions as visual perception.
5. Model optional/redundant sources and cost-sensitive Main-Agent choices.
6. Audit Question semantic fidelity and single-source ablations, alongside
   human ratings of naturalness.

Maintain strict entity/evidence/task-family splits during training/evaluation,
not just random question splits. Gold graphs are never public prompts.

Optional candidate Subtasks are selected only from already cached,
independently replayed and structurally disconnected facts. They are recorded
in PRIVATE Gold metadata, never forced into required dependencies or injected
into the Question. The current prototype does not yet expose them as
interactive Main-Agent choices; that belongs to the next simulator stage.


## v0.2: MMQA-guided simulated Scene Bank

This independent pilot explicitly targets Main Agent **orchestration**, not
image/video/audio recognition accuracy. No actual media files are needed.

The new scenarios package extracts MultiModalQA question types, pseudolanguage
questions, topic hints, intermediate evidence patterns and supporting-context
modalities. These are SOURCE MOTIFS, never claims about the new world's facts.
MMQA does not natively supply video or audio: those are deliberately synthetic
extensions of cross-source task motifs.

The generator creates 9 balanced scenario domains, 20+ scene settings and
controlled observations of images, timed videos and multi-speaker audio.
For every scene it adds a reference-bearing media source, a complementary
table/text record, a simulated searchable registry record and a distracting
search result. The same source observations deterministically generate
Subtask goals, answers, event-based summary traces and the final question.

Three patterns are generated: media->search chains, media->parallel lookups->
join, and table-vs-registry cross-checks. Removing any required source makes
the complete Gold response unresolvable. Clean, partial and deliberately
misleading sub-agent variants are included to build orchestration episodes;
none is an unconditional optimal-action label.

Build 100 scenes from the installed MMQA training QA file:

    python -m envir.scenarios.cli build \
      --mmqa data/world-source/MultiModalQA/MMQA_train.jsonl.gz \
      --output-dir envir/data/sim_scenarios_v1 \
      --max-records 3000 --target 100 --seed 17

Replay-check all answers/feedback and required-source ablation:

    python -m envir.scenarios.cli replay \
      --gold envir/data/sim_scenarios_v1/scenes_gold.jsonl

The outputs include: mmqa_motifs.jsonl (source priors); scenes_gold.jsonl
(PRIVATE teacher-side source observations, Subtasks, trace and answers);
public_tasks.jsonl (answer-blind questions, visible media IDs, tools);
simulation_summary.json (domain/mode/pattern and MMQA pattern alignment);
scenario_index.json (stable identifiers and replay status).

An optional simulated tool call reveals one source to the Main Agent only
after the Agent requests it. The PRIVATE Gold record must not be supplied as
an Agent prompt. Example:

    python -m envir.scenarios.cli tool \
      --gold envir/data/sim_scenarios_v1/scenes_gold.jsonl \
      --scene-id SCENE_ID \
      --action InspectSimulatedMedia \
      --source-id MEDIA_ID

The initial author uses deterministic, varied templates; a future optional
LLM could expand noncritical narrative surface content under consistency
checking. This pilot is **parallel to** the original source-backed envir
prepare/compose and does not claim to have integrated full KQA or YAGO
semantic grounding into the new synthetic worlds.


## Macro TaskPlan pilot: independent work packages, fixed concurrency 10

The coarse-grained planner is **independent of both** the source-backed
YAGO Subtask-First composer and the synthetic MMQA Scene Bank. It does NOT
alter the legacy environment. It is a first-stage **TaskPlan generator**,
not a complete Feedback/trajectory simulator.

Design boundaries:

1. **Question Level**: one practical user need and final deliverable.
2. **Macro-Orchestration Level**: several independently delegable work
   packages (e.g., investigate a situation, audit records, review rules,
   assess consequences, research feasible responses). Each package exposes
   typed input/output handoffs; graph edges are real data requirements.
3. **Sub-Agent Execution Level**: any media inspection, KoPL relation
   traversal, OCR, search or calculation stays in PRIVATE per-package
   internal_execution_outline, where later reliable Feedback Construction
   can turn it into source-backed tool events and summary_trace.

The planner samples 9 scenario domains, 4 issue variants per domain,
5 question intents and 5 macro DAG shapes (chain, parallel_join,
fork_join, mixed, staged). Domain and topology are crossed; no domain has
a fixed DAG. No simulated errors, runtime variants, token budgets or
latency/cost distributions are generated. All tasks use the **single fixed
concurrency maximum of 10**. reference_waves is only a valid dependency
schedule, never the Main Agent's required action order.

KQA-Pro: Parse complete KoPL ProgramIR and retain structural/operator
priors. These are NOT treated as one agent per KoPL node.
MultiModalQA: Sample official question/composition roles and modality
contexts to select relevant work/evidence patterns; no real media files
are needed. Video/audio roles are marked synthetic extensions.
YAGO: Optionally read existing envir bank.sqlite in read-only mode to
extract observed *predicate name* hints. This does NOT ground the
synthetic scenario with real YAGO facts.

Generate a 100-plan pilot from the real source files:

    python -m envir.macro.cli build \
      --kqa data/world-source/KQA-Pro/train.json \
      --mmqa data/world-source/MultiModalQA/MMQA_train.jsonl.gz \
      --bank envir/data/pilot/bank.sqlite \
      --output-dir envir/data/macro_v1 \
      --target 100 --max-kqa-records 10000 \
      --max-mmqa-records 3000 --seed 17

Omit --bank if the YAGO SQLite bank is not prepared. The planner then
uses KQA + MMQA structural priors without a YAGO semantic hint.

Replay structural validation:

    python -m envir.macro.cli replay \
      --plans envir/data/macro_v1/private_taskplans.jsonl

Output files:

* private_taskplans.jsonl: PRIVATE teacher structural TaskPlans with
  source-program priors, macro DAG, typed contracts, internal execution
  outlines, coarsening stats and reference scheduling waves. **There are NO grounded Gold answers yet**.
* public_questions.jsonl: only public question, situation/resource
  descriptions and an explicit unrealized-resource notice.
* macro_summary.json: graph, intent/domain and question diversity counters.
* prior_statistics.json: dataset coverage and whether YAGO prior was supplied.

The validator proves type-connected DAG structure and public/private
separation, **not** natural-language question validity, end-to-end
executability, YAGO fact correctness or reliable sub-agent feedback.

Next stage will ground every work package in a simulated evidence world and
derive verified Answer / Summary / Summary Trace using scoped executors,
with no private future-results leaking to an earlier delegated work unit.


For more varied natural-language **public question surfaces**, you may
optionally connect an OpenAI-compatible LOCAL vLLM server (the deterministic
Macro TaskPlan itself is unchanged):

    python -m envir.macro.cli build \
      --kqa data/world-source/KQA-Pro/train.json \
      --mmqa data/world-source/MultiModalQA/MMQA_train.jsonl.gz \
      --output-dir envir/data/macro_v1_llm \
      --target 100 --seed 17 \
      --question-api-base http://localhost:8000/v1 \
      --question-model YOUR_LOCAL_MODEL_NAME \
      --question-workers 4

The Question Author sees only the public situation, user need, requested
deliverable and resource descriptions. It NEVER receives teacher-private
macro topology, fine-grained KoPL operations, execution traces or answers.
The rewritten question is checked for subject/intent preservation using
lightweight rules. Invalid outputs and model-call failures revert to the
deterministic question. This is surface-level **heuristic** validation, not
a semantic proof; downstream task authoring must still certify meaning
against the grounded world before using any candidate for training.


## Macro TaskPlan V2 — work-deliverable composition (2026-11)

V1 uses five manually authored DAGs and couples an Intent to a
single graph. V2 introduces a **backward work-product grammar** and is
invoked from a separate CLI; old macro V1, envir.scenarios, and the
legacy environment/ remain unchanged.

Key distinction:

- **Question**: an end-user information need, contextual resources,
  conditions and required final result. No agent/tool plan revealed.
- **Macro TaskPlan**: a directed graph of substantial, independently
  delegable *work products* such as investigations, record audits,
  timelines, requirements reviews, compliance assessments, response
  options or rankings. Edges name the exact producer output slot,
  consumer input slot, typed artifact and **why the handoff is needed**.
- **Subtask execution** (FUTURE stage): analysis of images/videos/audio,
  original KoPL operators, individual searches, calculations, internal
  tool observations, Answer, Summary and Summary Trace. These are not
  individual macro nodes.

The planner first selects a user intent and desired output artifact
(e.g., ImpactReport), then recursively chooses from multiple
semantically specified recipes that produce this artifact. Reused work
products are memoized; branching and joins follow from actual
input/output requirements. It does not choose one of five fixed DAGs.
A mild, independent structural sampling preference explores both serial
and parallel routes without creating artificial edges. Internal
micro-actions are coarsened to their work package's private outline.

The original data have distinct roles:

- KQA-Pro: full dependency-preserving KoPL ProgramIR saved as a
  **private internal reasoning prior**. No micro-node is automatically
  delegated as a macro task.
- MultiModalQA: official question/composition type, source modalities
  and original question hint are kept as *source-pattern priors*.
  The corresponding source contents are not claimed to exist.
- YAGO: optionally contributes observed relation-name hints from the
  existing **read-only** EvidenceStore. These are not synthetic truths.
- OmniGAIA: optional local official question metadata file contributes
  ONLY rhetorical question-style categories; neither original
  examples/answers nor annotated reasoning/media is copied into tasks.
  Without a local OmniGAIA file, the generated question grammars are
  simply benchmark-inspired patterns.

All scheduling uses ONE fixed maximum of **10 concurrent Sub-Agents**.
No varying limits, costs, latencies, fault scenarios or deceptive
Sub-Agent feedback are generated.

Build 100 structural plans from the original input files:

    python -m envir.macro.v2_cli build \
      --kqa data/world-source/KQA-Pro/train.json \
      --mmqa data/world-source/MultiModalQA/MMQA_train.jsonl.gz \
      --bank envir/data/pilot/bank.sqlite \
      --output-dir envir/data/macro_v2 \
      --target 100 \
      --max-kqa-records 10000 \
      --max-mmqa-records 3000 --seed 17

Optional: add --omnigaia /path/to/official/test_metadata.json to
sample original OmniGAIA QUESTION style only. Omit --bank if the
envir SQLite YAGO bank is not available. Neither is required to run.

Optionally use a LOCAL OpenAI-compatible model (only public Question
naturalization, not TaskPlan authoring):

    python -m envir.macro.v2_cli build \
      --kqa data/world-source/KQA-Pro/train.json \
      --mmqa data/world-source/MultiModalQA/MMQA_train.jsonl.gz \
      --output-dir envir/data/macro_v2_llm --target 100 \
      --question-api-base http://localhost:8000/v1 \
      --question-model YOUR_LOCAL_MODEL_NAME \
      --question-workers 4

The optional Question Author receives only the public Question Brief.
Its rewritten question passes conservative topic/intent/no-leak checks
or falls back to the deterministic wording. This does **not** certify
semantic equivalence; meaningful approval awaits source-grounded
semantic review.

Replay saved plans:

    python -m envir.macro.v2_cli replay \
      --plans envir/data/macro_v2/private_taskplans_v2.jsonl

Output files:
- private_taskplans_v2.jsonl — PRIVATE teacher *structural* TaskPlans,
  work recipes, dependency reasons, internal action outlines, KQA/MMQA
  priors, coarsening and reference waves. NO grounded Gold or feedback.
- public_questions_v2.jsonl — questions and prospective resource roles
  only. These source descriptions are NOT yet instantiated files.
- macro_v2_summary.json — graph/intent/work/query diversity statistics.
- macro_v2_source_priors.json — source prior coverage and policies.

V2 validates dependency types, their explicit work requirements,
contribution of each work to the terminal output, DAG acyclicity,
no fine-grained internal steps in public questions, and fixed
concurrency 10. Validation does NOT establish correct factual answers,
the necessity of every source in a realized world, or reliable
Sub-Agent traces. The next stage must ground source material, produce
source-scoped Feedback and certify question-answer/evidence equivalence.


## Macro V3 — concrete QuestionNeed → macro handoffs → reliable feedback

V3 is a separate experiment; V1/V2, envir.scenarios and the legacy
environment/ remain unchanged. The major change from V2 is that **the
specific information need is sampled before planning the DAG**.

It builds each controlled scenario in this order:

1. Read complete KQA-Pro KoPL programs and MMQA question/composition
   annotations. They are **abstract structural/answer-operator priors**,
   not copied benchmark factual content. Optional OmniGAIA JSON/JSONL
   contributes question STYLE only, not reference solutions or media.
2. Create a concrete InformationNeed: user goal, scope case IDs, date,
   comparison/filter/timing/aggregation conditions, required source roles.
3. Create a controlled simulated evidence world: observations, record
   status/quantities, effective conditions, capacity and independent
   timestamp reports. The scenario is SYNTHETIC, with simulated textual
   media observations and no actual images/videos/audio or external truth.
4. Compile independently delegable macro work packages with *exact*
   input/output handoffs. The graph is derived from the required evidence;
   it is not selected from a fixed set of DAG templates. Scopes include
   object identity, so two CaseSet operands are NOT collapsed to one.
5. Dynamically merge adjacent scoped records-audit + record-matching work
   when it has only one consumer and represents the same deliverable.
   Internal source inspection actions remain inside the task, not Main
   delegations.
6. Derive reliable Answer / Summary / Summary Trace from source-scoped
   deterministic execution. A Sub-Agent sees only its authorized sources
   plus completed upstream outputs; it cannot see private future tasks
   or Gold answers.
7. Recompute every subtask, ablate every required source, and execute
   one actual fact-value counterfactual per question. These checks prove
   consistency in the controlled world; they do NOT establish real-world
   truth or the uniqueness/necessity of every source across all worlds.

The maximum concurrency is FIXED to 10. No dynamic scheduler profiles,
deceptive feedback or attack variants are generated.

Command (uses original KQA + MMQA; no existing YAGO bank required):

    python -m envir.macro.v3_cli build \
      --kqa data/world-source/KQA-Pro/train.json \
      --mmqa data/world-source/MultiModalQA/MMQA_train.jsonl.gz \
      --output-dir envir/data/macro_v3 \
      --target 100 \
      --max-kqa-records 10000 \
      --max-mmqa-records 3000 --seed 17

Optional: append --omnigaia /path/to/official/omnigaia.json to use
rhetorical question-style priors ONLY. The script never imports the
dataset's answer, annotated solution, real media or knowledge facts.

Optional local vLLM rewriting of the already GROUNDED user question:

    python -m envir.macro.v3_cli build \
      --kqa data/world-source/KQA-Pro/train.json \
      --mmqa data/world-source/MultiModalQA/MMQA_train.jsonl.gz \
      --output-dir envir/data/macro_v3_llm \
      --target 100 --seed 17 \
      --question-api-base http://localhost:8000/v1 \
      --question-model YOUR_LOCAL_MODEL_NAME \
      --question-workers 4

The optional author receives only the public question and source
descriptions. Case IDs, dates, source names, intent and numerical
conditions are checked. Bad rewrites fall back to the original.
These are *heuristic surface checks*, not proof of semantic equivalence.

Replay generated private scenarios:

    python -m envir.macro.v3_cli replay \
      --scenes envir/data/macro_v3/private_grounded_scenes_v3.jsonl

Generated files:

- private_grounded_scenes_v3.jsonl — TEACHER ONLY: controlled synthetic
  sources, InformationNeed, macro graph, Gold result, all reliable
  feedback and counterfactual witness.
- public_questions_v3.jsonl — PUBLIC: concrete question and available
  simulated source handles, WITHOUT observations, graph, feedback or Gold.
- macro_v3_summary.json — distribution and evidence replay statistics.

YAGO's real fact-level relation binding is NOT included in V3's
synthetic-world generator yet. The older V1/V2 source-bank path remains
available. Thus V3 is a verified *simulated* compositional task pilot,
not a claim of preserving original KQA/MMQA/YAGO data semantics.


## Macro V4: Clean Task-first / Evidence-within-Task

V4 is an isolated CLEAN pilot. V1, V2, V3 and environment/ are unchanged.
The design deliberately separates **macro delegation (Main Agent)** from
**evidence collection and reasoning (Sub-Agent)**:

1. Compose complete, independently delegable Macro Tasks and typed
   result handoffs from the V2 semantic work-product grammar. The graph
   follows substantive work dependencies, not one KoPL step per agent.
2. **Only after Task composition**, derive a PUBLIC Question from the
   scenario, final Task goal/deliverable and public source categories.
   The Question Author does NOT see any concrete evidence value, private
   proof graph, feedback, hidden orchestration order or final answer.
3. Build a deterministic synthetic source world with multiple domain-aware
   case-scoped observations per source. Each source is a TEXTUAL surrogate:
   video/image/audio modality hints DO NOT indicate actual perception.
4. Execute every Macro Task with an INTERNAL EvidenceGraph:
   source_observation nodes can only access its granted sources;
   upstream_scope and upstream_finding nodes can only use COMPLETED
   predecessor WorkProducts; verified_scope_alignment checks shared case
   references; work_synthesis combines the supported evidence.
5. Generate clean Sub-Agent feedback with Answer (a *work product*),
   Summary, structured Summary Trace (one entry per evidence node),
   evidence_refs and transitive source_fact_refs.
6. Teacher-side termination checking verifies the terminal work-product's
   complete EVIDENCE SUPPORT CLOSURE. It does NOT match a Gold Answer,
   force a final answer as a separate agent Task, or require Main to
   individually orchestrate internal evidence nodes.

All required Evidence is complete and consistent. The current environment
does NOT introduce missing/conflicting evidence, retries, verification,
backend switching or new tasks created by recovery. Those are possible
future extensions, intentionally outside the clean phase.

KQA-Pro is only an INTERNAL REASONING/complexity prior and MMQA is
only a composition/source-modality prior. V4 does not import original
dataset facts or perform actual multimodal perception. No YAGO factual
grounding is claimed. The synthetic case evidence ensures deterministic
feedback replay, **not** an external correctness guarantee.

The maximum concurrent Sub-Agents remains FIXED at 10. A reference
execution schedule is retained only in the teacher-private plan.

Build from the original KQA and MMQA priors:

    python -m envir.macro.v4_cli build \
      --kqa data/world-source/KQA-Pro/train.json \
      --mmqa data/world-source/MultiModalQA/MMQA_train.jsonl.gz \
      --output-dir envir/data/macro_v4 \
      --target 100 \
      --max-kqa-records 10000 \
      --max-mmqa-records 3000 \
      --seed 17

Replay all task-level evidence and feedback:

    python -m envir.macro.v4_cli replay \
      --scenes envir/data/macro_v4/private_clean_scenes_v4.jsonl

Outputs:

- private_clean_scenes_v4.jsonl: TEACHER ONLY. Macro TaskGraph,
  Task-level Question Brief, source observations, per-Task EvidenceGraphs,
  clean Answers/Summaries/Summary Traces, and evidence-level goal coverage.
  There is NO final Gold Answer.
- public_questions_v4.jsonl: PUBLIC questions plus source descriptions
  and modality hints ONLY. There are no private DAGs, evidence facts or
  Feedback snippets.
- macro_v4_summary.json: diversity and clean evidence-coverage statistics.

V4 replay detects changed/removed synthetic evidence, tampered Feedback
or Summary Traces, broken typed handoffs, evidence leakage into public
questions, and incorrect terminal proof coverage. It cannot yet certify
naturalness of every Question, authenticity of real video/image/audio
evidence, or behavior under untrusted Sub-Agent feedback.

For the optional V4 Author-Critic Question naturalization pipeline, see
[macro/QUESTION_NATURALIZATION.md](macro/QUESTION_NATURALIZATION.md).
