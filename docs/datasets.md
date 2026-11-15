# Dataset configuration

Both MAESTRO (`bench_orchestra_omnigaia.py`) and Orchestra-o1
(`baseline/orchestra-o1/bench_orchestra_o1_omnigaia.py`) use the same preparation
step before starting model backends. Their existing entrypoint names are retained.

## Frozen pretest / final evaluation splits

Three **fixed one-ID-per-line files** are versioned under the associated
dataset folders. The original question/answer/attachment files remain ignored.

| Evaluation | ID file | Selection |
| --- | --- | --- |
| OmniGAIA pretest (60) | `data/OmniGAIA/splits/pretest_60_ids.txt` | 20 Easy + 20 Medium + 20 Hard; 30 video/audio + 30 image/audio |
| OmniGAIA final (100) | `data/OmniGAIA/splits/final_100_ids.txt` | 34 Easy + 33 Medium + 33 Hard; 50 video/audio + 50 image/audio |
| MM-BrowseComp final (100) | `data/MM-BrowseComp/splits/final_100_ids.txt` | Five categories × 20; all 22 subtasks; level 1/2 = 85/15 |

OmniGAIA pretest uses the previously approved 60 question IDs. The final
OmniGAIA selection excludes those 60 IDs **and all media IDs shared with
the pretest**. The 100 MM-BrowseComp IDs come from the **official 400-task
dataset**, not the older 224-task release. MM-BrowseComp IDs are not continuous
1–400; IDs above 1000 are valid.

`task_ids_file` is an optional YAML field accepted by MAESTRO and Orchestra-o1.
The default OmniGAIA benchmark configs now point at the **pretest 60**;
MM-BrowseComp configs point at the **final 100**. To run OmniGAIA final 100
without duplicating model settings:

```bash
python bench_orchestra_omnigaia.py \
  --config config/benchmarks/orchestra_omnigaia.yaml \
  --task_ids_file data/OmniGAIA/splits/final_100_ids.txt

python baseline/orchestra-o1/bench_orchestra_o1_omnigaia.py \
  --config baseline/orchestra-o1/config/benchmarks/orchestra_o1_omnigaia.yaml \
  --task_ids_file data/OmniGAIA/splits/final_100_ids.txt
```

Both launchers load selected tasks in **manifest order**, irrespective of
the source JSONL row ordering. Missing, duplicate, or malformed task IDs fail
before model calls. If a manifest is used with `max_tasks` smaller than its
selection, the launcher raises an error instead of silently truncating;
use `max_tasks: null` to run the entire frozen set. The existing `--tasks`
argument still works to narrow a selected split during debugging.
Each run's model manifest records the ID-file path and the selected IDs.

Use the stdlib-only script to re-create or verify the same frozen files:

```bash
python benchmark/experiment_splits.py
python benchmark/experiment_splits.py --check
python benchmark/experiment_splits.py --check --verify-data \
  --omnigaia-meta data/OmniGAIA/test_metadata.json \
  --mm-meta data/MM-BrowseComp/data/MMBrowseComp_400.jsonl
```

The `--verify-data` command checks membership, counts, category/level
balance, and OmniGAIA media overlap against local **official metadata**.
MM-BrowseComp questions and answers remain encrypted; the verifier
reads only public metadata fields and never exports plaintext.

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
