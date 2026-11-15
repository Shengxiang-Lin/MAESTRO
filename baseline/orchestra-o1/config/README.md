# Orchestra-o1 benchmark configurations

Configuration files are split by **dataset** and **experiment stage**.
Use the same model filename across OmniGAIA stages for controlled comparison.

| Directory | Benchmark | Questions | Fixed task-ID manifest |
| --- | --- | ---: | --- |
| [omnigaia/pretest](omnigaia/pretest/) | OmniGAIA pre-experiment | 60 | `data/OmniGAIA/splits/pretest_60_ids.txt` |
| [omnigaia/final](omnigaia/final/) | OmniGAIA final evaluation | 100 | `data/OmniGAIA/splits/final_100_ids.txt` |
| [mm_browsecomp/final](mm_browsecomp/final/) | MM-BrowseComp final evaluation | 100 | `data/MM-BrowseComp/splits/final_100_ids.txt` |

All configurations set `max_tasks: null` so the selected manifest is not
truncated, and use distinct output directories under `logs/` for each
dataset/stage/model combination. The model settings are unchanged across
the paired OmniGAIA pretest/final YAMLs.

From the **MAESTRO repository root**:

```bash
# OmniGAIA 60-task pretest (default)
python baseline/orchestra-o1/bench_orchestra_o1_omnigaia.py \
  --config baseline/orchestra-o1/config/omnigaia/pretest/orchestra_o1_omnigaia.yaml

# OmniGAIA 100-task final
python baseline/orchestra-o1/bench_orchestra_o1_omnigaia.py \
  --config baseline/orchestra-o1/config/omnigaia/final/orchestra_o1_omnigaia.yaml

# MM-BrowseComp 100-task final
python baseline/orchestra-o1/bench_orchestra_o1_omnigaia.py \
  --config baseline/orchestra-o1/config/mm_browsecomp/final/orchestra_o1_mm_browsecomp.yaml
```

To run a different model combination, select the corresponding YAML file
inside the relevant stage directory. No source data or model inference is
needed to validate the config paths; see `benchmark/test_orchestra_o1_config_splits.py`.
