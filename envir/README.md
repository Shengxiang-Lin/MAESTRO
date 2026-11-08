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
