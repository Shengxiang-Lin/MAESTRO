from __future__ import annotations

from typing import Any, Dict, List, ClassVar
import asyncio
import os
from pathlib import Path
import sys
import tempfile

from base.agent.base_action import BaseAction

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUNTIME_DIR = PROJECT_ROOT / "tmp"

class ExecuteCodeAction(BaseAction):
    name: str = "ExecuteCodeAction"
    description: str = (
        "Execute code in a sandboxed subprocess (python|bash). Relative output "
        f"files are created under {DEFAULT_RUNTIME_DIR}. Generated, downloaded, "
        "and intermediate files must stay in ORCHESTRA_TMP_DIR; do not write "
        "them to the repository root. Default disabled; enable via config and whitelist."
    )
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "code": {"type": "string"},
            "code_type": {"type": "string", "enum": ["python", "bash"], "default": "python"},
            "timeout_sec": {"type": "integer", "default": 10},
        },
        "required": ["code", "code_type"],
        "additionalProperties": False,
    }
    DISALLOWED_BASH: ClassVar[List[str]] = [" rm ", "sudo", "chmod", "chown", ">/", "> /", " mv /", " dd ", " mkfs", " mount "]

    async def __call__(self, **kwargs) -> Any:
        code = kwargs.get("code", "")
        code_type = kwargs.get("code_type", "python")
        timeout = int(kwargs.get("timeout_sec", 10))
        try:
            if code_type == "bash":
                return await self._exec_bash(code, timeout)
            if code_type == "python":
                return await self._exec_python(code, timeout)
            return {"success": False, "output": None, "error": f"Unsupported code_type: {code_type}", "metrics": {}}
        except Exception as e:
            return {"success": False, "output": None, "error": str(e), "metrics": {}}

    @staticmethod
    def _runtime_dir() -> Path:
        configured = os.environ.get("ORCHESTRA_TMP_DIR")
        runtime_dir = Path(configured).expanduser() if configured else DEFAULT_RUNTIME_DIR
        if not runtime_dir.is_absolute():
            runtime_dir = PROJECT_ROOT / runtime_dir
        runtime_dir = runtime_dir.resolve()
        runtime_dir.mkdir(parents=True, exist_ok=True)
        return runtime_dir

    @staticmethod
    def _subprocess_env(runtime_dir: Path) -> Dict[str, str]:
        env = os.environ.copy()
        runtime = str(runtime_dir)
        env["ORCHESTRA_TMP_DIR"] = runtime
        env["TMPDIR"] = runtime
        env["TMP"] = runtime
        env["TEMP"] = runtime
        return env

    async def _exec_bash(self, code: str, timeout: int) -> Dict[str, Any]:
        low = f" {code.strip()} ".lower()
        for bad in self.DISALLOWED_BASH:
            if bad in low:
                return {"success": False, "output": None, "error": f"disallowed command in bash: {bad.strip()}", "metrics": {}}
        runtime_dir = self._runtime_dir()
        proc = await asyncio.create_subprocess_shell(
            code, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            cwd=str(runtime_dir), env=self._subprocess_env(runtime_dir),
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            return {"success": False, "output": None, "error": "timeout"}
        ok = proc.returncode == 0
        return {"success": ok, "output": stdout.decode("utf-8", "replace"), "error": None if ok else stderr.decode("utf-8", "replace"), "metrics": {}}

    async def _exec_python(self, code: str, timeout: int) -> Dict[str, Any]:
        runtime_dir = self._runtime_dir()
        tmpdir = runtime_dir / ".exec"
        tmpdir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", delete=False, dir=tmpdir, suffix=".py") as tf:
            tf.write(code)
            path = tf.name
        proc = await asyncio.create_subprocess_exec(
            sys.executable, path, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            cwd=str(runtime_dir), env=self._subprocess_env(runtime_dir),
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            return {"success": False, "output": None, "error": "timeout"}
        ok = proc.returncode == 0
        return {"success": ok, "output": stdout.decode("utf-8", "replace"), "error": None if ok else stderr.decode("utf-8", "replace"), "metrics": {}}
