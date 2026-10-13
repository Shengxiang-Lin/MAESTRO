# Dataset configuration

Both MAESTRO (`bench_orchestra_omnigaia.py`) and Orchestra-o1
(`baseline/orchestra-o1/bench_orchestra_o1_omnigaia.py`) use the same preparation
step before starting model backends. Their existing entrypoint names are retained.

## MM-BrowseComp

Add these fields to your experiment YAML; retain your existing model roles:

```yaml
dataset_type: MM-BrowseComp
dataset_path: data/MM-BrowseComp/data/MMBrowseComp_400_decrypted.jsonl
dataset_jsonl_path: data/MM-BrowseComp/data/test_metadata.jsonl
attachments_dir: data/MM-BrowseComp
level_filter: null
max_tasks: 10
```

`dataset_path` is the original source, and may be a `.json` record list, a JSON
object containing a `data`/`tasks` list, or a `.jsonl` file. For a JSON export,
change only this field to that JSON's actual path. The official encrypted JSONL
also works: preparation recognizes its question wrapper and uses the dataset's
canary to decrypt question, answer and checklist, as the official script does.

Preparation removes only the exact opening instruction asking for a roadmap.
It maps every `images` entry to a local `omni_modal_input`, preserving attachment
order. An empty `images` list is valid. GitHub URLs are resolved to existing files
under `MMBC_images`; no image downloads are performed. Missing/empty files, path
escapes, duplicate task IDs and malformed records fail before any model startup.

The source is preserved. Answer, checklist and source references remain in the
prepared record for evaluation and are not exposed in the agent instruction.
The shared environment provides the cleaned question, attachment paths and
ImageAnalysisAction hints. ImageAnalysisAction reads actual image bytes and calls
the configured vision tool model; Main/Sub-Agent prompts contain file paths.

If `dataset_jsonl_path` is omitted, MM-BrowseComp writes `<source-stem>.prepared.jsonl`
beside the source. On each startup conversion is validated and the resulting
content compared: unchanged files are reused, source changes regenerate output
atomically. The output must differ from the source.

Provided example configs (first 10 tasks, default API model roles):

```bash
python baseline/orchestra-o1/bench_orchestra_o1_omnigaia.py \
  --config baseline/orchestra-o1/config/benchmarks/orchestra_o1_mm_browsecomp.yaml

python bench_orchestra_omnigaia.py \
  --config config/benchmarks/orchestra_mm_browsecomp.yaml
```

These examples use qwen3.7-plus Main/Sub-Agents. Copy the model settings from your
actual experiment before using them for a different model combination.
MM-BrowseComp results use an `mm_browsecomp_` filename prefix, and model-role
manifests record dataset type, original source, generated JSONL and attachment root.

Evaluation currently uses the shared final-answer semantic scorer. Official
MM-BrowseComp checklist metrics are not implemented by this change; use a separate
checklist evaluation over the saved solution record if those metrics are needed.
Keep decrypted benchmark records out of public uploads.

## OmniGAIA

Existing configurations remain compatible; omitting `dataset_type` defaults to
OmniGAIA. Existing JSONL is passed to the shared reader unchanged. Official JSON
metadata is converted automatically:

```yaml
dataset_type: OmniGAIA
dataset_path: data/OmniGAIA/test_metadata.json
attachments_dir: data/OmniGAIA
```

The default generated filename is `test_metadata.jsonl`. `dataset_jsonl_path`
can override the output path for JSON conversion. Existing OmniGAIA JSONL needs
no conversion and does not use the optional output setting.
