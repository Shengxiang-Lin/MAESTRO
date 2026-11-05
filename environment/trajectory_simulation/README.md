# Skeleton-bound O1 trajectory synthesis

This stage realizes the frozen Stage-1 solution as Main/Sub-Agent interactions.
It preserves composition fragment boundaries, adds cuts for independent branches
and shared intermediate results, and dispatches every ready fragment in parallel.
The accepted Stage-2 question is preserved.

## Strict semantic boundary

The two teacher LLMs select bounded language plans. The controller renders native
O1 actions and feedback from verified operations, exact operands, completed input
bindings, source provenance and symbolic results. The LLM does not freely rewrite
predicates, comparisons, relation directions, candidate sets or conclusions.
This is gold-guided language realization, not an autonomous planning evaluation.
Currently the language options are `concise` and `stepwise`.

- Main chooses a language variant for each ordered ready task and its reasoning.
  The public output is `delegate_task` with native `params.tasks`, or `complete`.
- Each current-round Sub-Agent receives a separate LLM request to choose its
  report style. Independent requests run through `subagent_workers` concurrently.
- `summary` states the subtask's result. `trace_summary` narrates inspected/reused
  references and the performed operations in prose or progress bullets.
- Exact intermediate answers populate `finish_result.result`. The final Main
  answer is copied from completed public results and checked against the oracle.
- Projected tasks and reports must exactly match one allowed realization. A
  same-model semantic judge cannot waive the contract or modify the gold logic.

For example a follow-up instruction names the completed `Attempt 1, subtask 1`
result and its exact value. It cannot replace that set with all records. Filters
report their true input and retained output; they do not invent reasons for
discarding particular entities. Joins identify both inputs explicitly.

Latent FIND targets are expressed through their public question clues, not their
hidden entity names. A name becomes visible when a Sub-Agent identifies it.
Private graph IDs, rejected-generation commentary and teacher instructions never
appear in projected public language.

## Evidence and parallelism

Every fragment is evaluated on a reachable, decision-complete evidence witness.
Acquisition is AND over required anchors and OR over their providers. Read order
respects those anchors. Previously read evidence is marked as reused.

Uncomposed independent operands of a join are separate tasks rather than being
hidden inside one terminal task. Only the actual join and final calculation wait
for their operands. Non-root shared intermediates have explicit result bindings.
No source-by-source decomposition replaces the original reasoning skeleton.

Main's public state contains the question, provided reference descriptors,
available native tools, and cumulative O1-formatted subtask history. Simulated
records have public reference handles, not invented image paths or URLs.
The teacher language-choice prompt retains completed result bindings; narrative
history above 9000 characters uses a marked recent-history view. Saved public
states always preserve the full cumulative history.

## Scene tool mapping

`tools.py` owns `ToolProfile` and the scene registry. `--scene omnigaia` selects
the native tools used by the actual OmniGAIA implementation:

| Frozen modality | Native inspection tool |
| --- | --- |
| text | ExtractUrlContentAction |
| search | GoogleSearchAction |
| image / image_report | ImageAnalysisAction |
| audio / audio_transcript | ParseAudioAction |
| video / video_report | VideoAnalysisAction |
| table / database | ExecuteCodeAction |

Discoverable evidence additionally uses the scene's search tool. Evidence reuse
without a new read uses the scene's calculation tool. Unknown modalities/profiles
raise an error. Add another `ToolProfile` to the registry to support another
scene; compiler, prompt, public tool registry and feedback share this mapping.

## Commands

Run from the repository root, with the normal configured conda environment:

```bash
python environment/simulate_trajectories.py plan --scene omnigaia \
  --max-records 10 --output work/trajectory_plans.jsonl

python environment/simulate_trajectories.py synthesize --scene omnigaia \
  --provider passthrough --max-records 10 --output work/trajectory_offline.jsonl

python environment/simulate_trajectories.py synthesize --scene omnigaia \
  --max-records 10 --output work/trajectory_llm.jsonl

python -m unittest discover -s environment/tests
```

Main/executor model profiles, temperatures, retries and current-round worker
concurrency continue to come from `environment/config.yaml`. Different questions
remain sequential. The passthrough provider is a structural reference, not
LLM-generated training data. Synthesis overwrites its selected output paths;
use distinct paths for smoke tests. There is no cross-run resume checkpoint.

## Output and scope

Each episode has native O1 `trajectory[]` actions and `subtask_results` with
`finish_result` and `trace_summary`, followed by `complete`. `gold_supervision`
contains private graph bindings, source witnesses and exact answers: exclude it
from student prompts. Rejected audits are private too.

Metadata explicitly records LLM style selection with verified semantic projection.
The system does not execute or fabricate raw tool calls, costs or step counts.
`language_contract_verified` proves conformance to the frozen skeleton, not that
the Stage-2 natural-language question is equivalent to that skeleton.
`upstream_question_semantics_revalidated` is therefore false. An incorrectly
rewritten question must be repaired at Stage 2; Stage 3 never silently replaces it.
