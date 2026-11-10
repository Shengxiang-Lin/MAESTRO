"""Shared environment ordered streaming, durable checkpoint, and resume regressions."""
from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path

from environment.streaming import (
    OrderedJournal, StreamingError, output_lock, run_ordered,
    stable_signature,
)


class StreamingTests(unittest.TestCase):
    def test_bounded_parallel_completion_writes_in_original_order(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            items = list(range(9))
            ids = [f"task-{i}" for i in items]
            calls = []
            committed = []

            def job(i):
                calls.append(i)
                if i == 0:
                    time.sleep(0.025)
                else:
                    time.sleep(0.002)
                return i

            def classify(i):
                return i % 3 != 0, {"value": i}, {"test": True}

            with output_lock(root, "test"):
                with OrderedJournal(root, "test", "sig", ids) as progress:
                    run_ordered(
                        items, job, classify, journal=progress, workers=3,
                        no_progress=True,
                        on_commit=lambda ix, ok, row, audit: committed.append(ix),
                    )
                    self.assertEqual(progress.completed, 9)
                    self.assertEqual(progress.accepted, 6)
                    self.assertEqual(progress.rejected, 3)
                    replay = progress.replay()
            self.assertEqual(committed, list(range(9)))
            self.assertEqual([row["value"] for _, row, _ in replay], items)
            self.assertEqual(set(calls), set(items))

    def test_interruption_resume_skips_committed_prefix(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ids = [f"scene-{i}" for i in range(6)]
            called = []
            def job(i):
                called.append(i)
                return i

            def halt(ix, ok, row, audit):
                if ix == 2:
                    raise RuntimeError("simulated interrupt after checkpoint")
            with output_lock(root, "resume"):
                with self.assertRaisesRegex(RuntimeError, "simulated interrupt"):
                    with OrderedJournal(
                        root, "resume", stable_signature({"case": 1}), ids,
                    ) as journal:
                        run_ordered(
                            list(range(6)), job,
                            lambda i: (True, {"value": i}, {}),
                            journal=journal, workers=1, no_progress=True,
                            on_commit=halt,
                        )
            # First three outcomes were durably checkpointed.
            called.clear()
            with output_lock(root, "resume"):
                with OrderedJournal(
                    root, "resume", stable_signature({"case": 1}), ids,
                ) as journal:
                    self.assertEqual(journal.completed, 3)
                    run_ordered(
                        list(range(6)), job,
                        lambda i: (True, {"value": i}, {}),
                        journal=journal, workers=2, no_progress=True,
                    )
                    self.assertEqual(journal.completed, 6)
                    self.assertEqual(
                        [row["value"] for _, row, _ in journal.replay()],
                        list(range(6)),
                    )
            self.assertEqual(set(called), {3, 4, 5})

    def test_checkpoint_truncates_uncommitted_tail(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with OrderedJournal(root, "recover", "stable", ["a", "b"]) as p:
                p.append(0, True, {"v": 1})
            target = root / "recover_accepted.jsonl"
            target.write_bytes(target.read_bytes() + b'{"uncommitted": true}\n')
            with OrderedJournal(root, "recover", "stable", ["a", "b"]) as p:
                self.assertEqual(p.completed, 1)
                self.assertEqual([row["v"] for _, row, _ in p.replay()], [1])
                p.append(1, False, {"v": 2})
            self.assertEqual(
                len((root / "recover_accepted.jsonl").read_text().splitlines()),
                1,
            )
            self.assertEqual(
                len((root / "recover_rejected.jsonl").read_text().splitlines()),
                1,
            )

    def test_changed_signature_or_tampering_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with OrderedJournal(root, "secure", "first", ["a"]) as p:
                p.append(0, True, {"answer": 3})
            with self.assertRaisesRegex(StreamingError, "signature"):
                OrderedJournal(root, "secure", "changed", ["a"])
            accepted = root / "secure_accepted.jsonl"
            accepted.write_bytes(accepted.read_bytes().replace(b'"answer": 3',
                                                               b'"answer": 4'))
            with self.assertRaisesRegex(StreamingError, "hash_mismatch"):
                OrderedJournal(root, "secure", "first", ["a"])

    def test_directory_lock_and_double_writer(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with output_lock(root, "exclusive"):
                with self.assertRaisesRegex(StreamingError, "Another process"):
                    with output_lock(root, "exclusive"):
                        pass


if __name__ == "__main__":
    unittest.main()
