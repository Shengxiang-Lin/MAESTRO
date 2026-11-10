# Trusted Sub-Agent report generation

```bash
python -m envir naturalize-subtasks --config envir/configs/subagent_reports.yaml
```

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

For research/recovery you can separately materialize an Oracle bank to the **independent** `implementation.output_dir` (by default `envir/data/maestro_subtask_oracle_100`) with:

```bash
python -m envir implement-subtasks --config envir/configs/subagent_reports.yaml
```

Or use `naturalize-subtasks --source-subtasks ... --source-scene-index ...`
to review an existing compatible Oracle snapshot without regenerating it.
Those overrides are intentionally optional in normal runs.
