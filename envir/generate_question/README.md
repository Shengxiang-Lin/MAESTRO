# Stage 1 — Current generation modules

The authoritative CLI, artifact layout and running instructions are in
[../README.md](../README.md). The active entrypoint is
`python -m envir generate-questions`.

## Core source layout

| File | Responsibility |
| --- | --- |
| `spec.py` | Freeze typed user Goal and Source contracts |
| `macro_first_v2.py` | Generate current Macro-level DAG and task schemas |
| `macro_first_v2_world.py` | Synthesize typed evidence and deterministic witnesses |
| `information_necessity.py` | Verify the actual independent candidate sets/answers |
| `source_title_contract.py` | Freeze modality-accurate source titles |
| `scenario_fields.py` | Deterministic, audited local repair of presentation fields |
| `scenario.py` | Scenario Author/Critic and semantic validation |
| `semantic_evidence.py` | Applied Gold-Solver predicate and answer-slot contract |
| `question.py` | Question Author/Critic and conservative output guards |
| `compiler.py`, `program_ir.py` | Executable Gold Program and Macro WorkProducts |
| `difficulty_proxy.py`, `quality.py` | Structural workload features and strict Replay |
| `stages.py` | Durable per-round Scenario/Question stage execution |
| `quota_pipeline.py` | Accepted-level quotas, bounded retries and final publication |
| `pipeline.py` | Canonical API composition and read-only quality report |

### Invariants

1. Scene World, source permissions, Goal and Macro solver remain frozen
   before any LLM paraphrasing.
2. Source titles derive only from source ID, modality and domain. LLM
   Scenario output may repair formatting but not invent evidence.
3. Question text must request one answer slot and preserve all executed
   clue/time constraints; a Critic pass is not a formal equivalence proof.
4. Two independent multimodal branches must narrow genuinely distinct
   candidate sets; alternatives differ from the target's final answer.
5. A quota can be satisfied **only** by accepted, unique, Replay-valid
   questions with the requested difficulty label, subject to a hard
   candidate budget.

Legacy `clue_chain`, `evidence_graph` and Macro-first V1 modules are
retained exclusively for existing snapshot replay and regression tests;
they have no current public generation switches. Avoid adding a new
parallel pipeline or duplicate `README` section for each experiment.

For local regression tests:

```bash
python -m unittest discover -s envir/tests -v
```
