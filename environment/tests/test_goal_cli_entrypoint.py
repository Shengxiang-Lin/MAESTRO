"""Regression test: Stage 2A CLI must work as a direct Python script.

Imported tests alone do not exercise the sys.path that Python selects for
the documented command: python environment/generate_goals.py.
"""
from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


_REPO = Path(__file__).resolve().parents[2]


class GoalCliEntrypointTests(unittest.TestCase):
    def test_direct_script_from_repo_root(self):
        completed = subprocess.run(
            [sys.executable, "environment/generate_goals.py", "--help"],
            cwd=_REPO, capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(
            completed.returncode, 0,
            f"stdout={completed.stdout}\nstderr={completed.stderr}",
        )
        self.assertIn("--max-goal-revisions", completed.stdout)
        self.assertIn("--blueprint-workers", completed.stdout)

    def test_module_entrypoint_from_repo_root(self):
        completed = subprocess.run(
            [sys.executable, "-m", "environment.generate_goals", "--help"],
            cwd=_REPO, capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(
            completed.returncode, 0,
            f"stdout={completed.stdout}\nstderr={completed.stderr}",
        )
        self.assertIn("--max-records", completed.stdout)


if __name__ == "__main__":
    unittest.main()
