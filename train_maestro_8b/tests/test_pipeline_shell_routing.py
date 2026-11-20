"""Shell-level regression: optional empty Stage2 args cannot skip Python."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest


ROOT = Path(__file__).resolve().parents[2]
PIPELINE = ROOT / "train_maestro_8b/run/pipeline.sh"


class PipelineShellRoutingTests(unittest.TestCase):
    def test_default_unset_optional_arguments_reach_python(self):
        """Under set -e, preflight/judge/pilot/grpo must never silently exit."""
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base / "model").mkdir()
            (base / "data").mkdir()
            (base / "data/train.parquet").write_bytes(b"mock")
            bin_dir = base / "bin"
            bin_dir.mkdir()
            fake = bin_dir / "python"
            fake.write_text(
                '#!/usr/bin/env bash\n'
                'set -e\n'
                'if [[ "$1" == "-" ]]; then exec "$TEST_REAL_PYTHON" "$@"; fi\n'
                'printf "FAKE_PYTHON %s\\n" "$*"\n'
                'if [[ "$*" == *"--max_train_rows"* ]]; then\n'
                '  last=""\n'
                '  while (( "$#" )); do\n'
                '    if [[ "$1" == "--output_dir" ]]; then\n'
                '      shift\n'
                '      last="$1"\n'
                '    fi\n'
                '    shift\n'
                '  done\n'
                '  mkdir -p "$last"\n'
                '  printf "%s\\n" '
                "'{\"batch_id\":\"x\",\"group_index\":0,\"reward\":0.1,\"valid\":true,\"judge_reward\":0.1}' "
                "'{\"batch_id\":\"x\",\"group_index\":0,\"reward\":0.8,\"valid\":true,\"judge_reward\":0.8}' "
                '> "$last/reward_diagnostics.jsonl"\n'
                'fi\n',
                encoding="utf-8",
            )
            fake.chmod(0o755)
            env = os.environ.copy()
            env.update({
                "MODEL": str(base / "model"),
                "DA_DATA": str(base / "data"),
                "RUN_ROOT": str(base / "run"),
                "DA_GRPO_CKPT": str(base / "grpo"),
                "DA_PILOT_CKPT": str(base / "pilot"),
                "DA_REWARD_MODE": "global",
                "SKIP_COMPLETED": "0",
                "STAGE1_BELIEF_ADAPTER": "",
                "DA_MICRO_BATCH": "",
                "DA_GRAD_ACCUM": "",
                "TEST_REAL_PYTHON": sys.executable,
                "PATH": str(bin_dir) + os.pathsep + env.get("PATH", ""),
            })
            for action, expected in (
                ("preflight", "--check"),
                ("judge", "--test_judge"),
                ("pilot", "--max_train_rows"),
                ("grpo", "--epochs"),
            ):
                with self.subTest(action=action):
                    result = subprocess.run(
                        ["bash", str(PIPELINE), action],
                        cwd=ROOT, env=env, text=True, capture_output=True,
                        timeout=20, check=False,
                    )
                    self.assertEqual(
                        result.returncode, 0,
                        result.stdout + "\n" + result.stderr,
                    )
                    self.assertIn("FAKE_PYTHON", result.stdout)
                    self.assertIn(expected, result.stdout)
                    self.assertNotIn("--micro_batch_size", result.stdout)
                    self.assertNotIn("--gradient_accumulation_steps", result.stdout)

    def test_nonempty_overrides_are_forwarded(self):
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base / "model").mkdir()
            (base / "data").mkdir()
            (base / "data/train.parquet").write_bytes(b"mock")
            bin_dir = base / "bin"
            bin_dir.mkdir()
            fake = bin_dir / "python"
            fake.write_text(
                '#!/usr/bin/env bash\nprintf "FAKE_PYTHON %s\\n" "$*"\n',
                encoding="utf-8",
            )
            fake.chmod(0o755)
            env = os.environ.copy()
            env.update({
                "MODEL": str(base / "model"),
                "DA_DATA": str(base / "data"),
                "RUN_ROOT": str(base / "run"),
                "DA_MICRO_BATCH": "2",
                "DA_GRAD_ACCUM": "16",
                "STAGE1_BELIEF_ADAPTER": "/tmp/test-adapter",
                "PATH": str(bin_dir) + os.pathsep + env.get("PATH", ""),
            })
            result = subprocess.run(
                ["bash", str(PIPELINE), "preflight"],
                cwd=ROOT, env=env, text=True, capture_output=True, timeout=20,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("--micro_batch_size 2", result.stdout)
            self.assertIn("--gradient_accumulation_steps 16", result.stdout)
            self.assertIn("--stage1_belief_adapter /tmp/test-adapter", result.stdout)


if __name__ == "__main__":
    unittest.main()
