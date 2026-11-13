# Trusted Sub-Agent report generation

```bash
python -m environment naturalize-subtasks --config environment/configs/subagent_reports.yaml
```


## V1.4: source-scoped code-owned Result

`reporting.result_policy: code_owned` (in `environment/configs/subagent_reports.yaml`)
enables a safer V1.4 report format. The model authors **only**
`finish_result.summary` from the current Macro's frozen step/evidence
contract; code renders `finish_result.result` and `status=done` from the
complete currently delivered WorkProduct. The model is not asked to describe
the overall question; the complete frozen question can contain downstream
conditions that a current Macro has not checked.

For example, a context-log filter returns candidate case IDs, NOT
independently confirmed report-year disagreements. Its Result enumerates the
actual candidate IDs, while the LLM's Summary must describe only context-log
filtering. Both the code result and authored summary run the normal Guard and
independent Critic. Existing scenes without the V1.4 marker retain the legacy
report mode.

To repair an already completed V1.4 run with one rejected Macro **without
re-running its 576 accepted reviews**, use a new output folder:

```bash
python -m environment naturalize-subtasks \
  --config environment/configs/subagent_reports.yaml \
  --reuse-from environment/data/maestro_stage2_reports_current_v1_4_100 \
  --output-dir environment/data/maestro_stage2_reports_v1_4_repaired
```

`--reuse-from` verifies and reuses prior accepted reports; it only regenerates
the rejected Macro(s) using the new result policy. Previously reviewed reports
remain byte-for-byte as authored under the earlier policy. For a fully
consistent corpus in which **all** 577 results are code-owned, generate fresh
reports in a separate empty output directory **without** `--reuse-from`.
After a successful repair, update downstream Stage 3/4 reporting paths to the
new reviewed `subtasks.jsonl`.

The config's `implementation.scenes` is the accepted Question bank.
Without `--scenes`, the command automatically reads that path. This is
the **one-command** workflow:

1. Build exact Oracle Macro WorkProducts and operator/evidence traces from
   frozen Scenes, replaying source constraints into
   `<report-output>/oracle/`.
2. Naturalize every Macro's independent Finish Result and trace narrative
   with the configured Author/Critic models, preserving immutable Oracle
   answers, provenance and tools. Publish completed accepted Scenes into
   `<report-output>/subtasks.jsonl`.

The `reporting` YAML section owns model endpoints, concurrency, revisions,
output path, and resume behavior. Internal status, failure reasons and
source-preservation checks are written to summary and failure JSONL files.
Nothing here represents an actual Web/media tool invocation.

For research/recovery you can separately materialize an Oracle bank to the **independent** `implementation.output_dir` (by default `environment/data/maestro_subtask_oracle_100`) with:

```bash
python -m environment implement-subtasks --config environment/configs/subagent_reports.yaml
```

Or use `naturalize-subtasks --source-subtasks ... --source-scene-index ...`
to review an existing compatible Oracle snapshot without regenerating it.
Those overrides are intentionally optional in normal runs.
