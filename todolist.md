# Orchestra-o1：8 组模型与数据集实验

在 `/MAESTRO` 运行。OmniGAIA 跑全量 360 题，MM-BrowseComp 跑全量 400 题；每题最多重试 10 轮，并发上限 10；联网检索和网页读取统一用 `qwen3.7-flash`，评估与轨迹总结用 `qwen3.7-plus`。配置中的 `main_model`、`sub_models` 以本表为准。

运行前在当前 shell 中设置 `DASHSCOPE_API_KEY`；以下命令使用服务器已有的 `orchestra-o1` Conda 环境。每组输出到独立的 `logs/` 目录。

```bash
cd MAESTRO
conda create --prefix ./X python=3.11
conda activate ./X
pip install -r requirements.txt
mkdir -p logs/nohup
mkdir data
```
OmniGAIA
```bash
huggingface-cli download --repo-type dataset RUC-NLPIR/OmniGAIA --local-dir ./data/OmniGAIA
```

MM-BrowseComp 
```bash
cd data
git clone https://github.com/MMBrowseComp/MM-BrowseComp.git
python src/decrypt.py data/MMBrowseComp_400.jsonl data/MMBrowseComp_400_decrypted.jsonl
```


| 数据集 | Main-Agent | Sub-Agent | 配置文件 |
|---|---|---|---|
| OmniGAIA | `qwen3.8-max` | `qwen3.7-flash` | `baseline/orchestra-o1/config/benchmarks/orchestra_o1_omnigaia_qwen38_max_qwen37_flash.yaml` |
| OmniGAIA | `qwen3.8-max` | `qwen3.7-plus` | `baseline/orchestra-o1/config/benchmarks/orchestra_o1_omnigaia_qwen38_max_qwen37_plus.yaml` |
| OmniGAIA | `qwen3.7-plus` | `qwen3.8-max` | `baseline/orchestra-o1/config/benchmarks/orchestra_o1_omnigaia_qwen37_plus_qwen38_max.yaml` |
| OmniGAIA | `qwen3.7-flash` | `qwen3.8-max` | `baseline/orchestra-o1/config/benchmarks/orchestra_o1_omnigaia_qwen37_flash_qwen38_max.yaml` |
| MM-BrowseComp | `qwen3.8-max` | `qwen3.7-flash` | `baseline/orchestra-o1/config/benchmarks/orchestra_o1_mm_qwen38_max_qwen37_flash.yaml` |
| MM-BrowseComp | `qwen3.8-max` | `qwen3.7-plus` | `baseline/orchestra-o1/config/benchmarks/orchestra_o1_mm_qwen38_max_qwen37_plus.yaml` |
| MM-BrowseComp | `qwen3.7-plus` | `qwen3.8-max` | `baseline/orchestra-o1/config/benchmarks/orchestra_o1_mm_qwen37_plus_qwen38_max.yaml` |
| MM-BrowseComp | `qwen3.7-flash` | `qwen3.8-max` | `baseline/orchestra-o1/config/benchmarks/orchestra_o1_mm_qwen37_flash_qwen38_max.yaml` |

## OmniGAIA

按需执行下面的命令：

```bash
# main=qwen3.8-max, sub=qwen3.7-flash
nohup python -u baseline/orchestra-o1/bench_orchestra_o1_omnigaia.py \
  --config baseline/orchestra-o1/config/benchmarks/orchestra_o1_omnigaia_qwen38_max_qwen37_flash.yaml \
  > logs/nohup/bench_omnigaia_o1_qwen38_max_qwen37_flash_360.log 2>&1 &

# main=qwen3.8-max, sub=qwen3.7-plus
nohup python -u baseline/orchestra-o1/bench_orchestra_o1_omnigaia.py \
  --config baseline/orchestra-o1/config/benchmarks/orchestra_o1_omnigaia_qwen38_max_qwen37_plus.yaml \
  > logs/nohup/bench_omnigaia_o1_qwen38_max_qwen37_plus_360.log 2>&1 &

# main=qwen3.7-plus, sub=qwen3.8-max
nohup python -u baseline/orchestra-o1/bench_orchestra_o1_omnigaia.py \
  --config baseline/orchestra-o1/config/benchmarks/orchestra_o1_omnigaia_qwen37_plus_qwen38_max.yaml \
  > logs/nohup/bench_omnigaia_o1_qwen37_plus_qwen38_max_360.log 2>&1 &

# main=qwen3.7-flash, sub=qwen3.8-max
nohup python -u baseline/orchestra-o1/bench_orchestra_o1_omnigaia.py \
  --config baseline/orchestra-o1/config/benchmarks/orchestra_o1_omnigaia_qwen37_flash_qwen38_max.yaml \
  > logs/nohup/bench_omnigaia_o1_qwen37_flash_qwen38_max_360.log 2>&1 &

```

## MM-BrowseComp

按需执行下面的命令：

```bash
# main=qwen3.8-max, sub=qwen3.7-flash
nohup python -u baseline/orchestra-o1/bench_orchestra_o1_omnigaia.py \
  --config baseline/orchestra-o1/config/benchmarks/orchestra_o1_mm_qwen38_max_qwen37_flash.yaml \
  > logs/nohup/bench_mm_browsecomp_o1_qwen38_max_qwen37_flash_400.log 2>&1 &

# main=qwen3.8-max, sub=qwen3.7-plus
nohup python -u baseline/orchestra-o1/bench_orchestra_o1_omnigaia.py \
  --config baseline/orchestra-o1/config/benchmarks/orchestra_o1_mm_qwen38_max_qwen37_plus.yaml \
  > logs/nohup/bench_mm_browsecomp_o1_qwen38_max_qwen37_plus_400.log 2>&1 &

# main=qwen3.7-plus, sub=qwen3.8-max
nohup python -u baseline/orchestra-o1/bench_orchestra_o1_omnigaia.py \
  --config baseline/orchestra-o1/config/benchmarks/orchestra_o1_mm_qwen37_plus_qwen38_max.yaml \
  > logs/nohup/bench_mm_browsecomp_o1_qwen37_plus_qwen38_max_400.log 2>&1 &

# main=qwen3.7-flash, sub=qwen3.8-max
nohup python -u baseline/orchestra-o1/bench_orchestra_o1_omnigaia.py \
  --config baseline/orchestra-o1/config/benchmarks/orchestra_o1_mm_qwen37_flash_qwen38_max.yaml \
  > logs/nohup/bench_mm_browsecomp_o1_qwen37_flash_qwen38_max_400.log 2>&1 &

```

查看某组启动日志：`tail -f logs/nohup/bench_<对应实验名>.log`。结果 CSV、API 用量日志和轨迹保存在该配置的 `result_folder` 下。
