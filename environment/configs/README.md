# MAESTRO Stage 1–4 canonical configuration

The four default YAML files are production configurations for **one 500-candidate
batch**, not the retired 10-scene pilot. Their paths all share the prefix
`environment/data/batches/candidate_500/`.

| Stage | Default YAML | Scene cap | Purpose |
| --- | --- | --- | --- |
| 1 | `questions.yaml` | target **500** (up to 2000 seeds) | freeze ProblemSpecs, World and executable Macro DAG; review natural questions |
| 2 | `subagent_reports.yaml` | **500** published Stage-1 Scenes | rebuild deterministic Oracle; independently review Sub-Agent feedback |
| 3 | `main_orchestration.yaml` | **500** reviewed Stage-2 Scenes | frozen legal dispatch; independently review Main instructions and completion |
| 4 | `untrusted_orchestration.yaml` | **500** accepted Stage-3 Scenes | Main-observable A/B1/B2/B3; one reviewed next-decision per fault |

## Run

Requires a local `local-qwen3.5-9b` OpenAI-compatible endpoint at
`127.0.0.1:8807/v1`, available YAGO corpus at
`data/world-source/YAGO-4.6`, and sufficient disk space for private
Stage-4A candidate journals. Set `overwrite: false` (default).

```bash
python -m environment generate-questions --config environment/configs/questions.yaml
python -m environment quality --config environment/configs/questions.yaml
python -m environment naturalize-subtasks --config environment/configs/subagent_reports.yaml
python -m environment generate-main-turns --config environment/configs/main_orchestration.yaml
python -m environment generate-untrusted-turns --config environment/configs/untrusted_orchestration.yaml
```

**Do not run all four commands blindly without checking each `summary.json`.**
In particular, LLM Question Author/Critic acceptance and Stage-3 trajectory
acceptance may be below 500. Stage 3 now creates
`stage3/accepted_scenes.jsonl` from its accepted journal rows. Stage 4 reads
that file, `stage3/episodes.jsonl`, and `stage2/subtasks.jsonl`, and rejects
any mismatch between reviewed Clean Reference and trusted evidence.

To inspect intermediate yields:

```bash
ROOT=environment/data/batches/candidate_500
jq '{status, target, generated, rejected, final_scenes}' "$ROOT/stage1/summary.json"
jq '{status, oracle_accepted_scenes, oracle_failed_scenes}' "$ROOT/stage2/summary.json"
jq '{status, accepted_scenes, rejected_scenes, fully_llm_reviewed_scenes}' "$ROOT/stage3/summary.json"
jq '{status, trusted_prefixes_rejected, available_scenarios, published_scenario_types, main_training_turns}' "$ROOT/stage4/summary.json"
```

### Stage-4 bounds

The 500-batch defaults use `max_candidates_per_round: 24`,
`max_distractors_per_fault: 8`, `max_selected_per_round: 2`,
`target_main_turns: 2000` and 4 LLM workers. They are **bounded candidate
exploration** and a positive-label budget, not 2000 guaranteed training turns
or 500 successful untrusted trajectories. B3 is prioritized among *feasible*
candidates, but every published B3 still needs real, independent,
Main-visible same-claim evidence. Increase budgets only after auditing the
30-Scene pilot and use a new output directory when changing caps.

### Stability and quality controls

- Stage 2 Oracle is regenerated from the input Scenes each run. Its
  `source_subtasks` and `source_scene_index` are derived from
  `reporting.output_dir/oracle`; neither is a user-configurable duplicate.
- Stage 3 keeps `freeze_dispatch: true`; routing, tools, DAG and output fields
  are never changed by a prose Author. No Delegate fallback is published.
- Stage 4 retains frozen prior dispatches and published reports, uses one
  atomic fault per candidate and does **not** execute future recovery actions.
  Template mode is for diagnostics only, never reviewed LLM supervision.
- `status: partial` may be legitimate after independent review rejects some
  Scenes. Never silently count rejected reports or fallback turns as accepted.
- For evaluation, split by original Scene or stronger shared-source template
  identity; do not place two perturbations of the same Scene into train/test.

Prior local pilot YAMLs were intentionally removed from `configs/`; recreate
small experiments by copying the canonical YAML to an isolated run directory.
