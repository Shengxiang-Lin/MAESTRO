"""GPU-isolated launcher for standalone TRL GRPO + vLLM Server Mode.

Use one terminal for `serve` (GPU 0) and another for `train`
(GPUs 1..train_gpus). The config alone chooses the topology; no
hand-written device lists or manual torchrun process count required.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import sys
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from train_maestro_8b.policy import grpo_runtime as runtime


def topology(cfg: dict, role: str) -> tuple[list[str], dict[str, str]]:
    """Return physical device IDs and the role-specific CUDA environment."""
    dist = cfg["distributed"]
    if dist["vllm_mode"] != "server":
        raise ValueError("Distributed launcher requires vllm_mode: server")
    if role not in ("serve", "train"):
        raise ValueError("role must be serve/train")
    if role == "serve":
        devices = [str(dist["rollout_gpu"])]
    else:
        devices = [str(g) for g in range(
            dist["train_gpu_start"],
            dist["train_gpu_start"] + dist["train_gpus"],
        )]
    env = os.environ.copy()
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = ",".join(devices)
    if role == "serve":
        env["VLLM_SERVER_DEV_MODE"] = "1"
        env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    else:
        # torchrun owns RANK, LOCAL_RANK and WORLD_SIZE.
        for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE"):
            env.pop(name, None)
    return devices, env


def command(cfg: dict, role: str, extra: list[str] | None = None) -> list[str]:
    extra = extra or []
    dist, rollout = cfg["distributed"], cfg["rollout"]
    if role == "serve":
        if extra:
            raise ValueError("serve does not accept training arguments")
        cmd = [
            "vllm", "serve", str(cfg["paths"]["model"]),
            "--host", dist["server_host"],
            "--port", str(dist["server_port"]),
            "--tensor-parallel-size", "1",
            "--gpu-memory-utilization", str(rollout["gpu_memory_utilization"]),
            "--max-model-len", str(cfg["model"]["max_seq_length"]),
            "--max-num-seqs", str(rollout["max_num_seqs"]),
            "--kv-cache-dtype", rollout["kv_cache_dtype"],
            "--weight-transfer-config", json.dumps({"backend": "nccl"}),
            "--logprobs-mode", "processed_logprobs",
            "--max-logprobs", "-1",
        ]
        if rollout["enforce_eager"]:
            cmd.append("--enforce-eager")
        return cmd
    if role == "train":
        if not extra:
            raise ValueError("train requires Stage2 model/data/output arguments")
        return [
            sys.executable, "-m", "torch.distributed.run",
            "--standalone", "--nproc_per_node", str(dist["train_gpus"]),
            "-m", "train_maestro_8b.policy.train", "da",
            *extra,
        ]
    raise ValueError("Unknown distributed launch role")


def require_healthy_server(cfg: dict) -> None:
    dist = cfg["distributed"]
    url = f"http://{dist['server_host']}:{dist['server_port']}/health"
    try:
        with urlopen(url, timeout=5) as response:
            if response.status != 200:
                raise RuntimeError(f"vLLM /health returned {response.status}")
    except (OSError, HTTPError, URLError) as error:
        raise RuntimeError(
            f"Dedicated vLLM server is not healthy at {url}. Start it first via "
            "`bash train_maestro_8b/run/pipeline.sh serve-vllm` "
            "in a separate terminal. " + str(error)
        ) from error


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--role", choices=("serve", "train"), required=True)
    p.add_argument("--config", type=Path, default=runtime.DEFAULT_CONFIG)
    p.add_argument("--model", type=Path, default=None, help="Optional model path shared by server/trainer")
    p.add_argument("--dry-run", action="store_true")
    args, extra = p.parse_known_args(argv)
    if extra and extra[0] == "--":
        extra.pop(0)
    cfg = runtime.load_config(args.config)
    if args.model is not None:
        cfg["paths"]["model"] = runtime.repo_path(args.model)
    devices, env = topology(cfg, args.role)
    cmd = command(cfg, args.role, extra)
    info = {
        "role": args.role,
        "physical_gpus": devices,
        "cuda_visible_devices": env["CUDA_VISIBLE_DEVICES"],
        "command": cmd,
        "shell_command": shlex.join(cmd),
        "batch_plan": runtime.batch_plan(cfg),
    }
    print(json.dumps(info, indent=2), flush=True)
    if args.dry_run:
        return 0
    if args.role == "train":
        require_healthy_server(cfg)
    # Replace the launcher process; preserve exit status and Ctrl-C signals.
    os.execvpe(cmd[0], cmd, env)


if __name__ == "__main__":
    raise SystemExit(main())
