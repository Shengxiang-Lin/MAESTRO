"""Lifecycle management for local vLLM OpenAI-compatible model servers."""
from __future__ import annotations

import atexit
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable

from base.engine.logs import logger


class VLLMServiceManager:
    """Start or reuse vLLM servers and expose them as normal OpenAI endpoints.

    MAESTRO's Main-Agent/Sub-Agent code does not need a separate local-model
    path. After preparation, the existing AsyncOpenAI client receives a local
    ``base_url`` and the vLLM ``served_model_name`` exactly as it would for a
    remote API model.
    """

    _instance: "VLLMServiceManager | None" = None

    def __init__(self) -> None:
        self._processes: Dict[str, subprocess.Popen] = {}
        self._runtime: Dict[str, Dict[str, Any]] = {}
        self._shutdown_on_exit_names: set[str] = set()
        self._registered_atexit = False

    @classmethod
    def default(cls) -> "VLLMServiceManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def prepare(self, name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Return a runtime-ready config, starting vLLM when necessary."""
        backend = str(config.get("backend") or config.get("provider") or "api").lower()
        if backend != "vllm":
            return dict(config)

        cfg = dict(config)
        model_path = self._expand_model_path(cfg.get("model_path"))
        if not model_path:
            raise ValueError(f"vLLM model {name!r} requires model_path")
        if model_path.startswith("$"):
            raise ValueError(
                f"vLLM model {name!r} has unresolved model_path {model_path!r}; "
                "set the referenced environment variable or write the path directly in YAML"
            )
        if model_path.startswith("/") and not Path(model_path).exists():
            raise FileNotFoundError(f"vLLM model path does not exist: {model_path}")

        served_name = str(cfg.get("served_model_name") or cfg.get("model") or name)
        host = str(cfg.get("host") or "127.0.0.1")
        port = int(cfg.get("port") or 8000)
        base_url = str(cfg.get("base_url") or f"http://{host}:{port}/v1").rstrip("/")
        api_key = str(cfg.get("api_key") or "EMPTY")
        auto_start = bool(cfg.get("auto_start", True))
        reuse_existing = bool(cfg.get("reuse_existing", True))

        if self._server_matches(base_url, served_name, api_key):
            logger.info(f"[vLLM] Reusing {name} at {base_url}")
            return self._runtime_config(
                name=name,
                config=cfg,
                model_path=model_path,
                served_name=served_name,
                base_url=base_url,
                api_key=api_key,
                pid=None,
                reused=True,
            )

        if self._port_open(host, port):
            if reuse_existing:
                raise RuntimeError(
                    f"Port {host}:{port} is already in use but does not expose "
                    f"the expected vLLM model {served_name!r}. Refusing to replace it."
                )
            raise RuntimeError(f"Port {host}:{port} is already in use")
        if not auto_start:
            raise RuntimeError(
                f"vLLM model {name!r} is not running at {base_url} and auto_start=false"
            )

        command, env = self._build_command(
            name=name,
            config=cfg,
            model_path=model_path,
            served_name=served_name,
            host=host,
            port=port,
            api_key=api_key,
        )
        log_path = self._log_path(name, cfg)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(
            f"[vLLM] Starting {name}: model={model_path}, served={served_name}, "
            f"gpus={env.get('CUDA_VISIBLE_DEVICES', '<inherited>')}, "
            f"tp={self._tensor_parallel_size(cfg)}, endpoint={base_url}"
        )
        log_handle = log_path.open("a", encoding="utf-8")
        try:
            process = subprocess.Popen(
                command,
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        finally:
            log_handle.close()
        self._processes[name] = process
        if bool(cfg.get("shutdown_on_exit", False)):
            self._shutdown_on_exit_names.add(name)
            self._register_atexit()

        timeout = float(cfg.get("startup_timeout", 900))
        interval = float(cfg.get("health_check_interval", 2))
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(
                    f"vLLM service {name!r} exited with code {process.returncode}. "
                    f"See {log_path}. Tail:\n{self._tail(log_path)}"
                )
            if self._server_matches(base_url, served_name, api_key):
                return self._runtime_config(
                    name=name,
                    config=cfg,
                    model_path=model_path,
                    served_name=served_name,
                    base_url=base_url,
                    api_key=api_key,
                    pid=process.pid,
                    reused=False,
                )
            time.sleep(max(0.2, interval))

        self._terminate_owned(name)
        raise TimeoutError(
            f"Timed out after {timeout:.0f}s waiting for vLLM model {name!r} at "
            f"{base_url}. See {log_path}. Tail:\n{self._tail(log_path)}"
        )

    def manifest(self, names: Iterable[str] | None = None) -> Dict[str, Dict[str, Any]]:
        selected = set(names) if names is not None else None
        return {
            name: dict(info)
            for name, info in self._runtime.items()
            if selected is None or name in selected
        }

    @staticmethod
    def _expand_model_path(value: Any) -> str:
        if value is None:
            return ""
        return os.path.expanduser(os.path.expandvars(str(value)))

    @staticmethod
    def _tensor_parallel_size(config: Dict[str, Any]) -> int:
        explicit = config.get("tensor_parallel_size")
        if explicit is not None:
            return max(1, int(explicit))
        gpus = config.get("gpus")
        if isinstance(gpus, (list, tuple)) and gpus:
            return len(gpus)
        return 1

    def _build_command(
        self,
        *,
        name: str,
        config: Dict[str, Any],
        model_path: str,
        served_name: str,
        host: str,
        port: int,
        api_key: str,
    ) -> tuple[list[str], Dict[str, str]]:
        python_executable = os.path.expanduser(
            os.path.expandvars(str(config.get("python_executable") or sys.executable))
        )
        command = [
            python_executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            model_path,
            "--served-model-name",
            served_name,
            "--host",
            host,
            "--port",
            str(port),
        ]

        option_map = {
            "dtype": "--dtype",
            "gpu_memory_utilization": "--gpu-memory-utilization",
            "max_model_len": "--max-model-len",
            "max_num_seqs": "--max-num-seqs",
            "max_num_batched_tokens": "--max-num-batched-tokens",
            "tensor_parallel_size": "--tensor-parallel-size",
            "pipeline_parallel_size": "--pipeline-parallel-size",
            "reasoning_parser": "--reasoning-parser",
            "quantization": "--quantization",
            "kv_cache_dtype": "--kv-cache-dtype",
            "download_dir": "--download-dir",
            "swap_space": "--swap-space",
        }
        normalized = dict(config)
        normalized["tensor_parallel_size"] = self._tensor_parallel_size(config)
        for key, flag in option_map.items():
            value = normalized.get(key)
            if value is not None:
                command.extend([flag, str(value)])

        bool_flags = {
            "enforce_eager": "--enforce-eager",
            "trust_remote_code": "--trust-remote-code",
            "enable_prefix_caching": "--enable-prefix-caching",
            "disable_log_requests": "--disable-log-requests",
        }
        for key, flag in bool_flags.items():
            if bool(config.get(key, False)):
                command.append(flag)

        if api_key and api_key != "EMPTY":
            command.extend(["--api-key", api_key])

        extra_args = config.get("extra_args") or []
        if isinstance(extra_args, dict):
            for key, value in extra_args.items():
                flag = "--" + str(key).replace("_", "-")
                if isinstance(value, bool):
                    if value:
                        command.append(flag)
                elif value is not None:
                    command.extend([flag, str(value)])
        elif isinstance(extra_args, (list, tuple)):
            command.extend(str(item) for item in extra_args)
        else:
            raise ValueError(f"vLLM model {name!r}: extra_args must be a list or mapping")

        env = dict(os.environ)
        env.setdefault("NO_PROXY", "127.0.0.1,localhost")
        env.setdefault("no_proxy", "127.0.0.1,localhost")
        env.setdefault("PYTHONUNBUFFERED", "1")
        if "omp_num_threads" in config:
            env["OMP_NUM_THREADS"] = str(config["omp_num_threads"])
        else:
            env.setdefault("OMP_NUM_THREADS", "4")
        gpus = config.get("gpus")
        if isinstance(gpus, str):
            gpu_value = gpus
        elif isinstance(gpus, (list, tuple)):
            gpu_value = ",".join(str(gpu) for gpu in gpus)
        else:
            gpu_value = ""
        if gpu_value:
            env["CUDA_VISIBLE_DEVICES"] = gpu_value
            env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        for key, value in (config.get("env") or {}).items():
            env[str(key)] = os.path.expandvars(str(value))
        return command, env

    @staticmethod
    def _log_path(name: str, config: Dict[str, Any]) -> Path:
        raw = str(config.get("log_file") or f"logs/vllm/{name}.log")
        return Path(os.path.expanduser(os.path.expandvars(raw)))

    def _runtime_config(
        self,
        *,
        name: str,
        config: Dict[str, Any],
        model_path: str,
        served_name: str,
        base_url: str,
        api_key: str,
        pid: int | None,
        reused: bool,
    ) -> Dict[str, Any]:
        runtime = dict(config)
        runtime.update(
            {
                "backend": "vllm",
                "model_path": model_path,
                "model": served_name,
                "served_model_name": served_name,
                "base_url": base_url,
                "api_key": api_key,
                "_runtime_prepared": True,
            }
        )
        self._runtime[name] = {
            "backend": "vllm",
            "model_path": model_path,
            "served_model_name": served_name,
            "base_url": base_url,
            "gpus": config.get("gpus"),
            "tensor_parallel_size": self._tensor_parallel_size(config),
            "pid": pid,
            "reused": reused,
        }
        return runtime

    @staticmethod
    def _port_open(host: str, port: int) -> bool:
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except OSError:
            return False

    @staticmethod
    def _server_matches(base_url: str, served_name: str, api_key: str) -> bool:
        headers = {}
        if api_key and api_key != "EMPTY":
            headers["Authorization"] = f"Bearer {api_key}"
        try:
            request = urllib.request.Request(
                base_url.rstrip("/") + "/models", headers=headers
            )
            with urllib.request.urlopen(request, timeout=2) as response:
                if response.status != 200:
                    return False
                payload = json.loads(response.read().decode("utf-8"))
            ids = {
                str(item.get("id"))
                for item in payload.get("data", [])
                if isinstance(item, dict) and item.get("id")
            }
            return served_name in ids
        except (OSError, ValueError, urllib.error.URLError, json.JSONDecodeError):
            return False

    @staticmethod
    def _tail(path: Path, max_lines: int = 30) -> str:
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            return "\n".join(lines[-max_lines:])
        except OSError:
            return "(log unavailable)"

    def _register_atexit(self) -> None:
        if not self._registered_atexit:
            atexit.register(self.shutdown_owned)
            self._registered_atexit = True

    def _terminate_owned(self, name: str) -> None:
        process = self._processes.get(name)
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)

    def shutdown_owned(self) -> None:
        for name in list(self._shutdown_on_exit_names):
            self._terminate_owned(name)
