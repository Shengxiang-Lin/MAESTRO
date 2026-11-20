# Legacy standalone DA-GRPO converter

The canonical MAESTRO data flow now separates environment data from
method-specific trainer inputs:

```text
python -m environment.data_export
  -> environment/data/unified/{cases.jsonl,decisions.jsonl,private/decision_labels.jsonl}
python baseline/da-sft/prepare_data.py
  -> baseline/da-sft/data/{train,eval,test}.parquet
python baseline/da-grpo/prepare_data.py
  -> baseline/da-grpo/data/{train,val,test}.parquet
```

See `environment/data_export/README.md` and the method README files for
commands and split policies. Each baseline owns its train/val/test split and
training schema; `environment.data_export` does not generate adapters.

The older `dataset.prepare_da_grpo` module remains as an isolated legacy
converter for old experiment scripts. It is **not used** by
`environment.data_export` or the canonical baseline training commands.
The generic `dataset.tokenization` helper remains in use by both trainers.
